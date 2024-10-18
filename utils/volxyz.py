import torch
import matplotlib.pyplot as plt

def volxyz(data, range=None):
    # If no range is provided, calculate the mean and standard deviation
    if range is None:
        mu = torch.mean(data)
        sig = torch.std(data)
        k = 5
        range = [mu - k * sig, mu + k * sig]

    # Get the slice numbers
    sliceNb = torch.round(torch.tensor(data.shape)).long() // 2

    # Extract slices
    slice1 = data[:, :, sliceNb[2]]  # Middle slice in the third dimension
    slice2 = data[:, sliceNb[1], :]    # Middle slice in the second dimension
    slice3 = data[sliceNb[0], :, :]    # Middle slice in the first dimension

    # Permute slices
    slice2 = slice2.permute(0, 2, 1)  # Change shape from (H, W) to (H, D, W)
    slice3 = slice3.permute(2, 1, 0)  # Change shape from (W, H) to (D, W, H)

    # Create the composite image
    dimImg = (slice1.size(0) + slice3.size(0) + 1, slice1.size(1) + slice2.size(2) + 1)
    img = torch.zeros(dimImg, dtype=data.dtype)

    img[:slice1.size(0), :slice1.size(1)] = slice1
    img[:slice1.size(0), slice1.size(1)+1:] = slice2
    img[slice1.size(0)+1:, :slice1.size(1)] = slice3

    # Plotting the image
    plt.imshow(img.numpy(), cmap='gray', vmin=range[0].item(), vmax=range[1].item())
    
    # Add lines to separate different views
    plt.axvline(x=slice1.size(1), color='white', linewidth=2)
    plt.axhline(y=slice1.size(0), color='white', linewidth=2)
    
    # Draw rectangle
    rect = plt.Rectangle((slice1.size(1), slice1.size(0)), slice1.size(1), slice1.size(1),
                         facecolor='white', edgecolor='white')
    plt.gca().add_patch(rect)

    plt.show()

# Example usage:
# data = torch.rand((10, 10, 10))  # Example 3D tensor
# volxyz(data)