""" Generate Gaussian"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ssim import ssim_loss

"""CHANGES MADE:

1. Removed the exp from the cholesky_coeff to prevent only +ve values
2. Added a threshold to the cholesky_coeff to prevent it from going to zero
3. Added a threshold to the det to prevent it from going to zero
4. Added a threshold to the normalized kernel to prevent it from going to zero
5. Added tanh to cholseky_coeff to allow negative values as well
6. Removed the tanh from the mean to allow the mean from going to the edge
"""


def clamp_below_threshold(x, threshold):
    return torch.where((x.abs() < threshold), x.sign() * threshold, x)

class splats2d:
    def __init__(self, image_size:tuple, kernel_size:int, device = None):
        self.mean = None
        self.cholesky_coeff = None #NOTE: diagonal elements should be positive
        self.opacity = None # opacity of the splat
        self.colors = None # colour of the splat
        self.image_size = image_size # Necessary to restrict the position of a splat during optimization
        self.kernel_size = kernel_size
        self.device = device
        self.EPS = 1e-8
        self.THRESHOLD = 0.0001


    def rasterize(self,
            mean: torch.Tensor,
            cholesky_coeff: torch.Tensor,
            opacity: torch.Tensor,
            colors: torch.Tensor,
            kernel_size: int,
            device='cpu'
        ):

        """
        > Normalises and reshapes the kernel to RGB channels, pads to match the image size, and translates based on given coords. Basically putting the relevant kernel in the relevant coordinate position.
        > Multiplies the RGB kernels with given colours, sums up the layers, and returns the final clamped and permuted image.

        inputs:
            mean: torch.Tensor; [b, 2];
                value lies between [-1, 1], since the meshgrid assumes 0 as the center,

            cholesky_coeff: torch.Tensor; [b, 3];

            opacity: torch.Tensor; [b, 1];
                value lies between [0, 1],

            colors: torch.Tensor; [b, 3];
                value lies between [0, 1],

            kernel_size: int;
            device: device

        return:
            Composited Image
        """



        if self.device is None:
            self.device = device

        # Sending all the tensors to the provided device
        mean = mean.to(device)
        cholesky_coeff = cholesky_coeff.to(device)
        opacity = opacity.to(device)
        colors = colors.to(device)

        # Get the covariance from cholesky_coeff: [B, l1, l2, l3] torch.tensor

        # Exponenetiating so as to optimize the logarithm and hence prevent floating point
        # errors when dealing with small numbers.

        # l1 = torch.exp(cholesky_coeff[:, 0]) # Shape: [B]
        # l2 = torch.exp(cholesky_coeff[:, 1]) # Shape: [B]
        # l3 = torch.exp(cholesky_coeff[:, 2]) # Shape: [B]

        l1 = cholesky_coeff[:, 0] # Shape: [B]
        l2 = cholesky_coeff[:, 1] # Shape: [B]
        l3 = cholesky_coeff[:, 2] # Shape: [B]

        l1 = clamp_below_threshold(l1, self.THRESHOLD)
        l2 = clamp_below_threshold(l2, self.THRESHOLD)
        l3 = clamp_below_threshold(l3, self.THRESHOLD)

        assert not torch.isnan(l1).any(), "l1 contains NaN values"
        assert not torch.isnan(l2).any(), "l2 contains NaN values"
        assert not torch.isnan(l3).any(), "l3 contains NaN values"

        # The covariance matrix for each batch (2x2) should be constructed for each batch element
        cov = torch.stack([
            torch.stack([l1**2, l1*l2], dim=-1),
            torch.stack([l2*l1, l2**2 + l3**2], dim=-1)
        ], dim=-2).unsqueeze(1).unsqueeze(1)  # Shape: [B, 1, 1, 2, 2]
        assert not torch.isnan(cov).any(), "cov contains NaN values"

        # torch.set_printoptions(profile="full")  # Display full content
        # print(det)
        # torch.set_printoptions(profile="default")

        det = cov.det() + self.EPS
        assert not torch.isnan(det).any(), "det contains NaN values"


        # Check for positive semi-definiteness
        if (det <= 0).any():
            raise ValueError("Covariance matrix must be positive semi-definite")

        inv_cov = torch.inverse(cov)

        # torch.set_printoptions(profile="full")  # Display full content
        # print(inv_cov)
        # torch.set_printoptions(profile="default")
 
        assert not torch.isnan(inv_cov).any(), "inv_cov contains NaN values"

        # Choosing the range between [-5, 5] to for sampling the points in betweeen as inputs for the gaussian
        start = torch.tensor([-5.0], device=self.device).view(-1, 1) # shape: [1, 1]
        end = torch.tensor([5.0], device=self.device).view(-1, 1)
        samples = start + (end - start) * torch.linspace(0, 1, steps=kernel_size, device=self.device)

        # Expanding dims for broadcasting
        samples_x = samples.unsqueeze(-1).expand(-1, -1, kernel_size)
        samples_y = samples.unsqueeze(1).expand(-1, kernel_size, -1)

        # Creating a meshgrid using the broadcasted samples
        samples_xy = torch.stack([samples_x, samples_y], dim=-1) # shape: [1, 10, 10, 2]

        # Getting the values of the gaussian at these points
        z = torch.einsum('b...i,b...ij,b...j->b...', samples_xy, -0.5 * inv_cov, samples_xy)
        kernel = torch.exp(z)/(2 * torch.pi * torch.sqrt(det))
        assert not torch.isnan(kernel).any(), "Kernel contains NaN values"

        # Normalizing the kernels : taking max of last two dimension keeping batch dim is not directly possible
        kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
        kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last 

        kernel_normalized = kernel / (kernel_max_2+self.EPS)
        assert not torch.isnan(kernel_normalized).any(), "kernel_normalized contains NaN values"

        # Adding padding
        pad_h = self.image_size[0] - kernel_size
        pad_w = self.image_size[1] - kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Kernel size should be smaller than the image size")
        padding = (pad_w//2, pad_w//2 + pad_w%2, # left and right padding
                   pad_h//2, pad_h//2 + pad_h%2) # up and bottom padding
        
        kernel_padded = F.pad(kernel_normalized, padding, mode='constant', value=0)
        assert not torch.isnan(kernel_padded).any(), "kernel_padded contains NaN values"

        kernel_rgb_padded = torch.stack(3*[kernel_padded], dim=1)
        assert not torch.isnan(kernel_rgb_padded).any(), "kernel_rgb_padded contains NaN values"

        """ Alpha compositing !!!"""
        # Extracting shape information
        b, c, h, w = kernel_rgb_padded.shape

        # Create a batch of 2D affine matrices
        theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=self.device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, :, 2] = mean # To be expected between 0 and 1.

        # Creating grid and performing grid sampling
        grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
        kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)
        assert not torch.isnan(kernel_rgb_padded_translated).any(), "kernel_rgb_padded_translated contains NaN values"

        # Applying the gaussian image opacity to colours of kernels
        colour_weight = opacity.view(-1, 1, 1) * torch.exp(-det)
        assert not torch.isnan(colour_weight).any(), "colour weight contains NaN values"

        rgb_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1) * colour_weight.unsqueeze(-1)
        assert not torch.isnan(rgb_values_reshaped).any(), " contains rgb_values_reshapedNaN values"

        final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
        assert not torch.isnan(final_image_layers).any(), "Final Image Layer contains NaN values"

        final_image = final_image_layers.sum(dim=0)
        final_image = torch.clamp(final_image, 0, 1)
        return final_image

def extract_data(image_path: str, image_shape: tuple[int], num_splats: int):

    image = Image.open(image_path)
    mean = [0.5, 0.5, 0.5]  # Example values for mean
    std = [0.5, 0.5, 0.5]   # Example values for std
    transform = transforms.Compose([
        transforms.Resize(image_shape),  # Resize the image to the target shape
        transforms.ToTensor(),  # Convert the PIL image to a PyTorch tensor and normalize to [0, 1]
        transforms.Normalize(mean, std)
    ])
    original_image = transform(image)
    original_array = np.array(image)
    h, w = image_shape

    coords = np.random.randint(0, [h, w], size=(num_splats, 2))

    # Normalize the coordinated between [-1, 1]
    normalized_coords = torch.tensor(coords / [h,w]).float() # betweeen 0 & 1
    normalized_coords = 2 * normalized_coords - 1.0

    color_values = [original_array[coord[1], coord[0]] for coord in  coords] #TODO Check if the channels are correct here is it h, w or w, h.
    color_values_tensor = torch.tensor(color_values).float()

    return normalized_coords, color_values_tensor, original_image

def denormalize(tensor, mean, std):
    # Reverse normalization: pixel = (pixel * std) + mean
    mean = torch.tensor(mean).view(3, 1, 1)  # Reshape mean to match image shape
    std = torch.tensor(std).view(3, 1, 1)    # Reshape std to match image shape
    return tensor * std + mean

def optimize(splats: splats2d, image_path: str, image_shape: tuple[int], num_splats: int, kernel_size: int):


    # put the data into desired device before working with them
    mean, colors, target_tensor = extract_data(image_path, image_shape, num_splats)

    cholesky_coeff = torch.rand((num_splats, 3), device=splats.device) * 2 - 1 # Sample Uniformly between [-1, 1]
    opacity = torch.randn((num_splats, 1), device=splats.device)
    colors = torch.randn((num_splats, 3)).to(splats.device)
    mean = mean.to(splats.device)
    target_tensor = target_tensor.to(splats.device)

    W_values = torch.cat([mean, cholesky_coeff, opacity, colors], dim=1)
    W = nn.Parameter(W_values)
    assert not torch.isnan(W_values).any(), "W contains NaN values"

    optimizer = torch.optim.Adam(params=[W], lr=0.01, betas=(0.9, 0.999), weight_decay=1e-5)
    # optimizer = torch.optim.SGD(params=[params], lr=0.01)
    criterion = nn.MSELoss()

    num_iter = 1000
    for itr in range(num_iter):
        optimizer.zero_grad()

        # _mean = torch.tanh(W[:, :2])
        _mean = (W[:, :2]) # Since using tanh cause mean point to be less likely at edge
        _cholesky_coeff = torch.tanh(W[:, 2:5])
        _opacity = torch.sigmoid(W[:, 5:6])
        _colors = torch.sigmoid(W[:, 6:])

        raster = splats.rasterize(
            mean=_mean,
            cholesky_coeff=_cholesky_coeff,
            opacity=_opacity,
            colors=_colors,
            kernel_size=kernel_size,
            device=splats.device,
        )
        applied_mean = [0.5, 0.5, 0.5]  # Example values for mean
        applied_std = [0.5, 0.5, 0.5]   # Example values for std
        transform = transforms.Compose([
            transforms.Normalize(applied_mean, applied_std)
        ])
        raster = transform(raster)

        # Normalize the raster
        loss = 0.4*ssim_loss(raster, target_tensor) + 0.6*criterion(raster, target_tensor)
        loss.backward()
        optimizer.step()

        if itr % 100 == 0:
            print(f'Epoch {itr}, Loss: {loss.item()}')
            # Denormalizing the image
            rastertmp = denormalize(raster.detach().cpu(), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            plt.imshow(rastertmp.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            # plt.savefig(f'num{itr}')


    return W


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_shape = (256, 256)
    kernel_size = 64
    num_splats = 1000
    gauss = splats2d(image_shape, kernel_size, device=device)
    optimize(splats=gauss,image_path='test_image.jpg', image_shape=image_shape, num_splats=num_splats, kernel_size=kernel_size)