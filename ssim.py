import torch
import torch.nn.functional as F

def gaussian_kernel(window_size=11, sigma=1.5, channels=3):
    coords = torch.arange(window_size) - (window_size - 1)/2
    g_1d = torch.exp(-(coords**2) / (2*sigma*sigma))
    g_1d /= g_1d.sum()
    g_2d = g_1d[:, None] * g_1d[None, :]
    kernel = g_2d.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)
    return kernel

def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    # print(img1.shape)
    c, h, w = img1.shape
    kernel = gaussian_kernel(window_size, sigma, c).to(img1.device)
    
    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=c)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    
    sigma1_sq = F.conv2d(img1*img1, kernel, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, kernel, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1*img2, kernel, padding=window_size//2, groups=c) - mu1_mu2
    
    numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator
    return ssim_map.mean()

def ssim_loss(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    ssim_value = ssim(img1, img2, window_size, sigma, C1, C2)  # from the ssim function
    return 1 - ssim_value
