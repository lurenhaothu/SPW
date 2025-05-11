import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import torchvision
from PIL import Image

class SteerableDecomp(torch.nn.Module):
    def __init__(self, complex=True, K=4, device='cuda'):
        super(SteerableDecomp, self).__init__()
        self.K = K
        self.complex = complex
        self.masks = {}
        self.device = device

    def get_grid(self, H, W):
        x = torch.linspace(-(H // 2 - 1) * np.pi / (H // 2), np.pi, H)
        x = x.reshape((H, 1)).expand((H, W))
        y = torch.linspace(-(W // 2 - 1) * np.pi / (W // 2), np.pi, W)
        y = y.reshape((1, W)).expand((H, W))
        radius = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.arctan(y / x)
        theta[x < 0] += torch.pi
        theta[(x >=0) & (y < 0)] += torch.pi * 2
        return radius, theta
    
    #def down_sample(self, fourier_domain_image):
    #    B, C, H, W = fourier_domain_image.shape
    #    return fourier_domain_image[:, :, H // 4 : H // 4 * 3, W // 4 : W // 4 * 3]
    
    #def up_sample(self, fourier_domain_image):
    #    B, C, H, W = fourier_domain_image.shape
    #    output = torch.zeros((B, C, 2 * H, 2 * W)).to(self.device).to(fourier_domain_image.dtype)
    #    output[:, :, H // 2: H // 2 * 3, W // 2: W // 2 * 3] = fourier_domain_image
    #    return output

    def get_mask(self, image_size: tuple[int]):
        if image_size in self.masks:
            return self.masks[image_size]
        #high_pass_filters = []
        #low_pass_filters = []
        #band_filters = []

        H, W = image_size

        radius, theta = self.get_grid(H, W)

        band_filters = torch.zeros((self.K, H, W))
        alpha_k = 2 ** (self.K - 1) * math.factorial(self.K - 1) / math.sqrt(self.K * math.factorial(2 * (self.K - 1)))
        for k in range(self.K):
            if self.complex:
                band_filters[k] = 2 * torch.abs(alpha_k * torch.pow(torch.nn.ReLU()(torch.cos(theta - torch.pi * k / self.K)), self.K - 1))
            else:
                band_filters[k] = torch.abs(alpha_k * torch.pow(torch.cos(theta - torch.pi * k / self.K), self.K - 1))
            band_filters[k, H // 2 - 1, W // 2 - 1] = 0
        #band_filters.append(band_i.to(self.device))

        self.masks[image_size] = band_filters.to(self.device)
        return self.masks[image_size]
    
    def forward(self, batch_images):
        B, C, H, W = batch_images.shape
        band_filters = self.get_mask((H, W))
        fourier_domain = torch.fft.fftshift(torch.fft.fft2(batch_images), dim=(2, 3))
        # high_freq_residue = fourier_domain * masks['high'][0]
        # output.append(torch.fft.ifft2(torch.fft.ifftshift(high_freq_residue, dim=(2, 3))))
        # fourier_domain = fourier_domain * masks['low'][0]
        band_signal = fourier_domain.unsqueeze(2) * band_filters
        output = torch.fft.ifft2(torch.fft.ifftshift(band_signal, dim=(3, 4)))

        if self.complex:
            return output
        else:
            return output.real
        
    '''
    def reconstruct(self, image):
        B, C, H, W = image[0].shape
        masks = self.get_mask((H, W))
        if image[0].is_complex():
            fourier_domain = [torch.fft.fftshift(torch.fft.fft2(i.real), dim=(-2, -1)) for i in image]
        else:
            fourier_domain = [torch.fft.fftshift(torch.fft.fft2(i), dim=(-2, -1)) for i in image]
        output = fourier_domain[-1]
        for i in range(self.N, 0, -1):
            output = self.up_sample(output) * masks['low'][i]
            output += torch.sum(fourier_domain[i] * masks['band'][i], dim=2) * masks['high'][i]
        output = output * masks['low'][0] + fourier_domain[0] * masks['high'][0]
        return torch.fft.ifft2(torch.fft.ifftshift(output, dim=(2,3))).real
    
    def reconstruct_map(self, image):
        B, C, H, W = image[0].shape
        masks = self.get_mask((H, W))
        fourier_domain = [torch.fft.fftshift(torch.fft.fft2(torch.abs(i)), dim=(-2, -1)) for i in image]
        output = torch.zeros_like(fourier_domain[-1])
        for i in range(self.N, 0, -1):
            output = self.up_sample(output) * masks['low'][i]
            output += torch.sum(fourier_domain[i] * masks['band'][i], dim=2) * masks['high'][i]
        output = output * masks['low'][0]# + fourier_domain[0] * masks['high'][0]
        return torch.fft.ifft2(torch.fft.ifftshift(output, dim=(2,3)))
    '''    

if __name__ == "__main__":
    a = ComplexSteerablePyramid(N=3, complex=True)
    path = "C:/Users/Renhao Lu/Desktop/dwt/test.jpg"
    imgg = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path)))
    imgg = imgg.unsqueeze(0).cuda()

    output = a(imgg)

    for i in output:
        print(i.shape)
        print(i.dtype)

    recons = a.reconstruct(output)

    fig, axe = plt.subplots(2,3)
    axe[0][0].imshow(imgg.cpu().squeeze().numpy())
    axe[0][1].imshow(recons.real.cpu().squeeze().numpy())
    axe[0][2].imshow(a.get_mask((400, 400))['high'][1].cpu().squeeze().numpy())
    axe[1][0].imshow(output[0].real.cpu().squeeze().numpy())
    axe[1][1].imshow(torch.sum(output[1], dim=2).angle().cpu().squeeze().numpy())
    axe[1][2].imshow(output[-1].real.cpu().squeeze().numpy())
    plt.show()