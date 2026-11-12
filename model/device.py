import torch


def get_device():
    # Intel XPU (TAMU ACE PVC) first, then CUDA (Colab), then CPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_accelerator(device):
    # torch.xpu / torch.cuda share the memory-stats API used in train.py; None on CPU
    return getattr(torch, device.type) if device.type in ("xpu", "cuda") else None
