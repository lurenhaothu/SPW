
import torch
import torchvision
import math
from model.SPW_loss.ComplexSteerablePyramid import ComplexSteerablePyramid
import matplotlib.pyplot as plt
import model.loss as loss_functions

class SPW_loss(torch.nn.Module):
    def __init__(self, spN=4, spK=4, beta=0.9, lamb=10, sigma=0.2, base_loss="bce"):
        super(SPW_loss, self).__init__()
        self.SP = ComplexSteerablePyramid(complex=True, N=spN, K=spK)
        self.sigma = sigma
        self.beta = beta
        self.lamb = lamb
        self.base_loss = base_loss
        if self.base_loss == 'dice':
            self.dice = loss_functions.DiceLoss()

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
    
    def get_map(self, mask, pred):
        B, C, H, W = mask.shape
        mask_decomp = self.SP(mask)
        pred_decomp = self.SP(pred)

        res = torch.zeros((B, C, H, W)).to(torch.abs(mask_decomp[1]).dtype).cuda()
        for i in range(self.SP.N):
            #i_level_feature = 1 - \
            #    torch.exp(-((torch.abs(torch.sum(mask_decomp[i + 1], dim=2)) - torch.abs(torch.sum(pred_decomp[i + 1], dim=2))) ** 2) \
            #              / (2 * self.sigma ** 2))
            i_level_feature = torch.sum(torch.abs(mask_decomp[i+1]), dim=2) + torch.sum(torch.abs(pred_decomp[i+1]), dim=2)
            res += math.pow(self.beta, i) * self.fft_upsample_n(i_level_feature, i)
        return res

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        with torch.no_grad():
            weight_map = self.lamb * self.get_map(mask, pred)

        if self.base_loss == 'bce':
            return -torch.mean((weight_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7)  \
                + (weight_map + class_weight[:,0:1,:]) * (1 - mask) * torch.log(1 - pred + 1e-7))
        elif self.base_loss == 'dice':
            return self.dice(mask, pred, w_map, class_weight, epoch) - torch.mean(weight_map * mask * torch.log(pred + 1e-7)  \
                + weight_map * (1 - mask) * torch.log(1 - pred + 1e-7))

