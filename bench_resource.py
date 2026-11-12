# Resource benchmark: device memory and time per training iteration of different losses on random
# images of different sizes. Run from the project root:
#   python bench_resource.py --sizes 512 1024 2048 --batch-sizes 1 2 4 8 --losses bce dice spw cwmi
#
# mode "loss": loss forward + backward on a random prediction that tracks gradients (no network).
#              Isolates the cost of the loss itself, e.g. the autograd graph CWMI keeps through
#              the steerable pyramid, which SPW builds under no_grad.
# mode "full": one U_Net training step (forward, loss, backward, Adam step), as in train.py.
#
# Memory columns (MB):
#   loss_graph_MB = allocated after loss forward - allocated before it, i.e. what the loss keeps
#                   alive for backward
#   peak_MB       = peak allocated during the iteration, including model, Adam state and inputs
# epoch_s = estimated epoch time for --epoch-images images at this batch size.
import argparse
import gc
import math
import os
import time
from datetime import datetime

import pandas as pd
import torch

import model.att_unet as models
import model.loss as loss_functions
from model.CWMI_loss.CWMI_loss import CWMI_loss
from model.SPW_loss.SPW_loss import SPW_loss
from model.clDice.clDice import soft_dice_cldice
from model.rmi.rmi import RMILoss
from model.device import get_device, get_accelerator

LOSSES = {
    "bce": lambda: loss_functions.BCELoss(),
    "bce_cw": lambda: loss_functions.BCE_withClassBalance(),
    "dice": lambda: loss_functions.DiceLoss(),
    "spw": lambda: SPW_loss(spN=4, spK=4, beta=0.9, lamb=10, base_loss='bce'),
    "cwmi": lambda: CWMI_loss(complex=True, spN=4, spK=4, lamb=0.9),
    "rmi": lambda: RMILoss(num_classes=2),
    "cldice": lambda: soft_dice_cldice(),
}

parser = argparse.ArgumentParser()
parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--losses", nargs="+", default=["bce", "dice", "spw", "cwmi"], choices=list(LOSSES))
parser.add_argument("--modes", nargs="+", default=["loss", "full"], choices=["loss", "full"])
parser.add_argument("--in-ch", type=int, default=1, help="input channels of U_Net (1 = SNEMI3D, 3 = RGB)")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--epoch-images", type=int, default=100, help="images per epoch for the epoch_s estimate")
parser.add_argument("--out", default="results/benchmark")
parser.add_argument("--csv", default=None, help="append rows to this CSV instead of a new timestamped file")
parser.add_argument("--repeat", type=int, default=0, help="repeat index, stored in the CSV")
args = parser.parse_args()

device = get_device()
acc = get_accelerator(device)
assert acc is not None, "benchmark needs an XPU or CUDA device"
MB = 1024 ** 2


def sync():
    acc.synchronize()


def is_oom(e):
    msg = str(e).lower()
    # Level Zero reports exhausted device memory as OUT_OF_DEVICE_MEMORY or OUT_OF_RESOURCES
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in msg or "out_of_device_memory" in msg \
        or "out_of_resources" in msg


def cleanup():
    gc.collect()
    acc.empty_cache()


def random_batch(B, H, W):
    # random binary masks; the cost of these losses does not depend on the image content
    mask = (torch.rand(B, 1, H, W, device=device) > 0.5).float()
    class_weight = torch.ones(B, 2, 1, device=device)
    w_map = torch.empty(0, device=device)
    return mask, class_weight, w_map


def run_loss_mode(loss_fn, B, H, W):
    mask, class_weight, w_map = random_batch(B, H, W)
    logits = torch.randn(B, 1, H, W, device=device, requires_grad=True)
    times = {"t_loss_fwd": [], "t_backward": []}
    graph = []
    for it in range(args.warmup + args.iters):
        if it == args.warmup:
            sync()
            acc.reset_peak_memory_stats()
        logits.grad = None
        pred = torch.sigmoid(logits)
        sync(); m0 = acc.memory_allocated(); t0 = time.perf_counter()
        loss = loss_fn(mask, pred, w_map, class_weight, 1)  # epoch=1: CWMI returns plain BCE at epoch 0
        sync(); m1 = acc.memory_allocated(); t1 = time.perf_counter()
        loss.backward()
        sync(); t2 = time.perf_counter()
        if it >= args.warmup:
            times["t_loss_fwd"].append(t1 - t0)
            times["t_backward"].append(t2 - t1)
            graph.append(m1 - m0)
        del loss, pred
    res = {k: 1000 * sum(v) / len(v) for k, v in times.items()}
    res["t_iter"] = res["t_loss_fwd"] + res["t_backward"]
    res["loss_graph_MB"] = sum(graph) / len(graph) / MB
    res["peak_MB"] = acc.max_memory_allocated() / MB
    return res


def run_full_mode(loss_fn, B, H, W):
    mask, class_weight, w_map = random_batch(B, H, W)
    image = torch.randn(B, args.in_ch, H, W, device=device)
    net = models.U_Net(img_ch=args.in_ch, output_ch=2).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.train()
    times = {"t_model_fwd": [], "t_loss_fwd": [], "t_backward": [], "t_step": []}
    graph = []
    for it in range(args.warmup + args.iters):
        if it == args.warmup:
            sync()
            acc.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        pred = torch.softmax(net(image), 1)[:, 1:2, :, :]
        sync(); m1 = acc.memory_allocated(); t1 = time.perf_counter()
        loss = loss_fn(mask, pred, w_map, class_weight, 1)
        sync(); m2 = acc.memory_allocated(); t2 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        sync(); t3 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        sync(); t4 = time.perf_counter()
        if it >= args.warmup:
            times["t_model_fwd"].append(t1 - t0)
            times["t_loss_fwd"].append(t2 - t1)
            times["t_backward"].append(t3 - t2)
            times["t_step"].append(t4 - t3)
            graph.append(m2 - m1)
        del loss, pred
    res = {k: 1000 * sum(v) / len(v) for k, v in times.items()}
    res["t_iter"] = sum(res.values())
    res["loss_graph_MB"] = sum(graph) / len(graph) / MB
    res["peak_MB"] = acc.max_memory_allocated() / MB
    del net, optimizer
    return res


os.makedirs(args.out, exist_ok=True)
out_csv = args.csv or os.path.join(args.out, "bench_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv")
prev_rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
print("device:", device, "|", acc.get_device_name(0), "| total memory: %.1f GB" % (acc.get_device_properties(0).total_memory / 1024 ** 3))
print("results ->", out_csv)

rows = []
for mode in args.modes:
    for size in args.sizes:
        for loss_name in args.losses:
            for B in sorted(args.batch_sizes):
                row = {"repeat": args.repeat, "mode": mode, "size": size, "batch": B, "loss": loss_name}
                cleanup()
                loss_fn = None
                try:
                    loss_fn = LOSSES[loss_name]()
                    run = run_loss_mode if mode == "loss" else run_full_mode
                    res = run(loss_fn, B, size, size)
                    row.update(status="ok", **{k: round(v, 2) for k, v in res.items()})
                    row["epoch_s"] = round(math.ceil(args.epoch_images / B) * res["t_iter"] / 1000, 1)
                except Exception as e:
                    if not is_oom(e):
                        raise
                    row["status"] = "OOM"
                del loss_fn
                rows.append(row)
                print(row, flush=True)
                pd.DataFrame(prev_rows + rows).to_csv(out_csv, index=False)
                if row["status"] == "OOM":
                    break  # larger batches of this loss/size will not fit either

df = pd.DataFrame(rows)
cols = [c for c in ["repeat", "mode", "size", "batch", "loss", "status", "t_iter", "t_model_fwd", "t_loss_fwd", "t_backward",
                    "t_step", "loss_graph_MB", "peak_MB", "epoch_s"] if c in df.columns]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
print("\n===== summary (times in ms per iteration) =====")
print(df[cols].to_string(index=False))
