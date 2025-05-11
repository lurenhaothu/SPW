
import torch
import torchvision
import math
from model.SPW_loss.ComplexSteerablePyramid import ComplexSteerablePyramid
import matplotlib.pyplot as plt

class SPW_loss(torch.nn.Module):
    def __init__(self, spN=4, spK=4, beta=0.9, lamb=10, pred_map=True):
        super(SPW_loss, self).__init__()
        self.SP = ComplexSteerablePyramid(complex=True, N=spN, K=spK)
        self.SP_real = ComplexSteerablePyramid(complex=False, N=spN, K=spK)
        self.beta = beta
        self.lamb = lamb
        self.pred_map = pred_map

    def fft_upsample_2(self, image: torch.tensor):
        B, C, H, W = image.shape

        f = torch.fft.rfft2(image, dim=(-2, -1))
        B, C, FH, FW = f.shape

        newf = torch.zeros((B, C, 2 * H, W + 1)).to(torch.cfloat).cuda()
        newf[:,:,:H // 2, :FW] = f[:,:,:H//2,:]
        newf[:,:, -H//2:, :FW] = f[:,:,-H//2:,:]

        res = torch.fft.irfft2(newf, dim=(-2, -1))
        return torch.abs(res)
    
    def fft_upsample_n(self, image, scale):
        for i in range(scale):
            image = self.fft_upsample_2(image)
        return image
    
    def get_map(self, image):
        B, C, H, W = image.shape
        sp_decomp = self.SP(image)

        res = torch.zeros((B, C, H, W)).to(torch.abs(sp_decomp[1]).dtype).cuda()
        for i in range(self.SP.N):
            i_level_feature = torch.abs(torch.sum(sp_decomp[i + 1], dim=2))
            res += math.pow(self.beta, i) * self.fft_upsample_n(i_level_feature, i)
        return res

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        with torch.no_grad():
            mask_weight_map = self.get_map(mask)
            pred_weight_map = self.get_map(pred)
        
        if self.pred_map:
            weight_map = self.lamb * mask_weight_map
        else:
            weight_map = self.lamb * (mask_weight_map + pred_weight_map)
        return -torch.mean((weight_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7)  \
            + (weight_map + class_weight[:,0:1,:]) * (1 - mask) * torch.log(1 - pred + 1e-7))

