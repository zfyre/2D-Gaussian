import torch
import torch.nn.functional as F


def rasterize(
        image_size: tuple,              
        mean: torch.Tensor,
        cholesky_coeff: torch.Tensor,
        opacity: torch.Tensor,
        colors: torch.Tensor,
        kernel_size: int,
        device='cpu'
    ):
    
    if device is None:
        device = device

    # Sending all the tensors to the provided device
    mean = mean.to(device)
    cholesky_coeff = cholesky_coeff.to(device)
    opacity = opacity.to(device)
    colors = colors.to(device)

    # Get the covariance from cholesky_coeff: [B, l1, l2, l3] torch.tensor

    # Exponenetiating so as to optimize the logarithm and hence prevent floating point
    # errors when dealing with small numbers.

    l1 = torch.exp(cholesky_coeff[:, 0]) # Shape: [B]
    l2 = torch.exp(cholesky_coeff[:, 1]) # Shape: [B]
    l3 = torch.exp(cholesky_coeff[:, 2]) # Shape: [B]
    assert not torch.isnan(l1).any(), "l1 contains NaN values"
    assert not torch.isnan(l2).any(), "l2 contains NaN values"
    assert not torch.isnan(l3).any(), "l3 contains NaN values"

    # The covariance matrix for each batch (2x2) should be constructed for each batch element
    cov = torch.stack([
        torch.stack([l1**2, l1*l2], dim=-1),
        torch.stack([l2*l1, l2**2 + l3**2], dim=-1)
    ], dim=-2).unsqueeze(1).unsqueeze(1)  # Shape: [B, 1, 1, 2, 2]
    # print(cov)
    assert not torch.isnan(cov).any(), "cov contains NaN values"
    det = cov.det()
    assert not torch.isnan(det).any(), "det contains NaN values"
    # print(det)
    # assert False
    # det = torch.clamp(det, min=1e-8)  # Prevent small/negative determinants

    # for i, d in enumerate(det):
    #     if d.item() < 0:
    #         print(f"Negative determinant at index {i}:")
    #         print(f"l1: {l1[i].item()}, l2: {l2[i].item()}, l3: {l3[i].item()}")
    #         print(f"Determinant: {d.item()}")

    # Check for positive semi-definiteness
    if (det <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inv_cov = torch.inverse(cov)
    # print(inv_cov)
    assert not torch.isnan(inv_cov).any(), "inv_cov contains NaN values"

    # Choosing the range between [-5, 5] to for sampling the points in betweeen as inputs for the gaussian
    start = torch.tensor([-5.0], device=device).view(-1, 1) # shape: [1, 1]
    end = torch.tensor([5.0], device=device).view(-1, 1)
    samples = start + (end - start) * torch.linspace(0, 1, steps=kernel_size, device=device)

    # Expanding dims for broadcasting
    samples_x = samples.unsqueeze(-1).expand(-1, -1, kernel_size)
    samples_y = samples.unsqueeze(1).expand(-1, kernel_size, -1)

    # Creating a meshgrid using the broadcasted samples
    samples_xy = torch.stack([samples_x, samples_y], dim=-1) # shape: [1, 10, 10, 2]

    # Getting the values of the gaussian at these points
    z = torch.einsum('b...i,b...ij,b...j->b...', samples_xy, -0.5 * inv_cov, samples_xy)
    kernel = torch.exp(z)/(2 * torch.pi * torch.sqrt(det))
    # print(kernel)
    assert not torch.isnan(kernel).any(), "Kernel contains NaN values"

    # Normalizing the kernels : taking max of last two dimension keeping batch dim is not directly possible
    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    # Adding padding
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller than the image size")
    padding = (pad_w//2, pad_w//2 + pad_w%2, # left and right padding
                pad_h//2, pad_h//2 + pad_h%2) # up and bottom padding
    kernel_padded = F.pad(kernel_normalized, padding, mode='constant', value=0)
    kernel_rgb_padded = torch.stack(3*[kernel_padded], dim=1)

    """ Alpha compositing !!!"""
    # Extracting shape information
    b, c, h, w = kernel_rgb_padded.shape

    # Create a batch of 2D affine matrices
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = mean # To be expected between 0 and 1.


    # Creating grid and performing grid sampling
    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    # Applying the gaussian image opacity to colours of kernels
    colour_weight = opacity.view(-1, 1, 1) * torch.exp(-det)
    # print(torch.exp(-det))
    # print(opacity.view(-1, 1, 1))
    # print(colour_weight)
    assert not torch.isnan(colour_weight).any(), "colour weight contains NaN values"

    # colour_weight = opacity.view(-1, 1, 1)
    rgb_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1) * colour_weight.unsqueeze(-1)
    # print(colors)
    # print(rgb_values_reshaped)
    # assert False
    # ic(rgb_values_reshaped.max())
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    # ic(final_image_layers.max())
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    # final_image = final_image.permute(1,2,0)

    return final_image


if __name__ == "__main__":

    