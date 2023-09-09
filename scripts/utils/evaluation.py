import cv2
import numpy as np
import torch 
import matplotlib.pyplot as plt
from utils.plotting import plot_gallery

def sliding_windows(tensor: torch.Tensor, window_size: int, stride: int):
    """
    Extracts sliding windows from a 4D tensor.

    Args:
        tensor (torch.Tensor): A 4D tensor with shape (batch_size, num_channels, height, width).
        window_size (int): The size of the sliding window in both height and width dimensions.
        stride (int): The step size of the sliding window.

    Returns:
        torch.Tensor: A tensor containing extracted windows with shape (batch_size, num_windows, num_channels, window_size, window_size).

    Notes:
        - This function extracts sliding windows from the input tensor with the specified size and stride.
        - The input tensor should have a 4D shape representing a batch of images.
        - The function returns a tensor containing all extracted windows.
    """
    batch_size, num_channels, height, width = tensor.size()
    patches = []
        
    for row in range(0, height - window_size + 1, stride):  # Adjust the range
        for col in range(0, width - window_size + 1, stride):  # Adjust the range
            window = tensor[:, :, row:row+window_size, col:col+window_size]
            patches.append(window.unsqueeze(1))  # Add a patches dimension
      
            
    return torch.cat(patches, dim=1)  # Concatenate patches along patches dimension

def reconstruct_from_patches(patches: np.array, image_shape: tuple, patch_size: tuple, stride: tuple):
    """
    Reconstructs an image from a collection of image patches.

    Args:
        patches (np.array): A collection of image patches with shape (num_patches, num_channels, patch_height, patch_width).
        image_shape (tuple): A tuple containing the shape of the original image in the format (num_channels, image_height, image_width).
        patch_size (tuple): A tuple specifying the size of each patch in the format (patch_height, patch_width).
        stride (tuple): A tuple specifying the step size of the sliding window in the format (vertical_stride, horizontal_stride).

    Returns:
        np.array: The reconstructed image with the same number of channels as the patches.

    Notes:
        - The function assumes that 'patches' and 'image_shape' have compatible dimensions.
        - The function calculates the reconstructed image by summing overlapping patches and normalizing the result.
        - The result is returned as a float32 NumPy array.
    """
        
    num_patches, num_channels, patch_height, patch_width = patches.shape
    _, image_height, image_width = image_shape
    reconstructed_image = torch.zeros((num_channels, image_height, image_width), dtype=torch.float32)
    patch_count = torch.zeros((image_height, image_width), dtype=torch.float32)

    idx = 0
    for i in range(0, image_height - patch_size[0] + 1, stride[0]):
        for j in range(0, image_width - patch_size[1] + 1, stride[1]):
            patch = patches[idx]
            
            for c in range(num_channels):
                reconstructed_image[c, i:i+patch_size[0], j:j+patch_size[1]] += patch[c]
            patch_count[i:i+patch_size[0], j:j+patch_size[1]] += 1
            
            idx += 1
            if idx >= num_patches:
                break

    # Normalize each channel by the number of patches contributing to each pixel
    for c in range(num_channels):
        reconstructed_image[c] = np.divide(reconstructed_image[c], patch_count, where=patch_count != 0)

    return reconstructed_image


def generate_residuals(x, x_hat, score):
    n=12
    
    x = x[70]
    x_hat = x_hat[70]
    score = score[70]
    
    residual, mask, full_pack = residual_recons(x, x_hat, score)
    
    score = score[:n]
    comparison = torch.cat([x[:n], x_hat[:n], residual[:n], mask[:n]])
    comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
    fig, ax = plot_gallery(comparison.cpu().numpy(), ncols=n)
    ax.set_title(f"likelihood scores - {score.cpu().numpy()}")
    
    x, x_hat, hm_residual = full_pack
    x, x_hat, hm_residual = x.permute(1,2,0), x_hat.permute(1,2,0), hm_residual.permute(1,2,0)
    hm_residual /= torch.max(hm_residual)
    mask = hm_residual.clone()
    

    threshold = 0.8
    mask[mask >= threshold] = 1.
    mask[mask < threshold] = 0.
    comparison = torch.cat([x, x_hat, hm_residual, mask])

    fig2, ax2 = plt.subplots()
    ax2.imshow(comparison.cpu().numpy())
    
    # comparison = comparison.permute(0, 2, 3, 1)  # [B, H, W, C]
    # fig2, ax2 = plot_gallery(comparison.cpu().numpy(), ncols=n)    
    
    return fig, fig2


def heatmap_score(score):
    normed_score = score
    heat_patches = torch.stack([torch.full((3,32,32), i, dtype=torch.float32) for i in normed_score])
    heatmap = reconstruct_from_patches(heat_patches, image_shape=(3,128,128),patch_size=(32,32), stride=(8,8))
    heatmap = heatmap.numpy()
    # x = reconstruct_from_patches(x.numpy(), image_shape=(3,128,128),patch_size=(32,32), stride=(8,8))
    return heatmap
    # original_image = (x*255).astype(np.uint8).transpose(1,2,0)
    
    
#     # Blend the heatmap overlay with the original image
#     alpha = 0.2  # Adjust the transparency of the heatmap overlay
#     heatmap_overlay = (heatmap * 255).astype(np.uint8)
#     heatmap_colormap = cv2.applyColorMap(heatmap_overlay, cv2.COLORMAP_JET)

#     heatmap_blend = cv2.addWeighted(original_image, 1 - alpha, heatmap_colormap, alpha, 0)

#     plt.imshow(heatmap_blend)
#     plt.savefig("./full_slice_ll_heatmap")

    
def residual_recons(x, x_hat, score, threshold=None):
    residual = torch.abs(x - x_hat) + 1e-6 / x.mean(dim=1, keepdim=True) +1e-6
    hm = heatmap_score(score.cpu().numpy())
    x = reconstruct_from_patches(x.numpy(), image_shape=(3,128,128),patch_size=(32,32), stride=(8,8))
    x_hat = reconstruct_from_patches(x_hat.numpy(), image_shape=(3,128,128),patch_size=(32,32), stride=(8,8))
    full_slice = reconstruct_from_patches(residual.numpy(), image_shape=(3,128,128),patch_size=(32,32), stride=(8,8))
    hm_residual = full_slice * hm
    
    
    residual_sum = residual.mean(dim=1, keepdim=True)
    
    # norm_residual_sum = residual_sum / torch.max(residual_sum)
    mask = residual_sum.clone()
        
    if threshold is None:
        threshold = 0.5
            
    mask[mask >= threshold] = 1.
    mask[mask < threshold] = 0.
    
    residual_sum = residual_sum.repeat(1, 3, 1, 1)
    mask = mask.repeat(1, 3, 1, 1)

    return residual_sum, mask, (x, x_hat, hm_residual)