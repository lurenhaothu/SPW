
import torch
import torchvision
import math
from model.SPW_loss.SteerableDecomp import SteerableDecomp
import matplotlib.pyplot as plt

class SW_loss(torch.nn.Module):
    def __init__(self, spK=4, lamb=10, sigma=0.1):
        super(SW_loss, self).__init__()
        self.SP = SteerableDecomp(complex=True, K=spK)
        self.sigma = 0.1
        # self.SP_real = ComplexSteerablePyramid(complex=False, N=spN, K=spK)
        # self.beta = beta
        self.lamb = lamb
    
    def get_map(self, image):
        B, C, H, W = image.shape
        sp_decomp = self.SP(image).real

        return torch.sum(sp_decomp, dim=2)

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        with torch.no_grad():
            mask_weight_map = self.get_map(mask)
            pred_weight_map = self.get_map(pred)
        
        weight_map = self.lamb * (1 - torch.exp(-((mask_weight_map - pred_weight_map) ** 2) / (2 * self.sigma ** 2)))
        
        return -torch.mean(weight_map * mask * torch.log(pred + 1e-7)  \
            + weight_map * (1 - mask) * torch.log(1 - pred + 1e-7))

