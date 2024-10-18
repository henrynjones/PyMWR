import torch

def perdecomp3D(u):
    # Get dimensions of the input tensor
    dim = u.shape

    # Initialize the boundary image
    v = torch.zeros(dim, dtype=u.dtype, device=u.device)

    # Compute boundary conditions
    v[0, :, :] = u[0, :, :] - u[-1, :, :]
    v[-1, :, :] = u[-1, :, :] - u[0, :, :]
    v[:, 0, :] += u[:, 0, :] - u[:, -1, :]
    v[:, -1, :] += u[:, -1, :] - u[:, 0, :]
    v[:, :, 0] += u[:, :, 0] - u[:, :, -1]
    v[:, :, -1] += u[:, :, -1] - u[:, :, 0]

    # Compute periodic elements
    idx1 = torch.arange(1, dim[0] + 1, dtype=torch.float32, device=u.device).view(-1, 1, 1)  # (dim[0], 1, 1)
    idx2 = torch.arange(1, dim[1] + 1, dtype=torch.float32, device=u.device).view(1, -1, 1)  # (1, dim[1], 1)
    idx3 = torch.arange(1, dim[2] + 1, dtype=torch.float32, device=u.device).view(1, 1, -1)  # (1, 1, dim[2])

    # Create the cosine terms
    f1 = torch.tile(torch.cos(2 * torch.pi * (idx1 - 1) / dim[0]), (1, dim[1], dim[2]))
    f2 = torch.tile(torch.cos(2 * torch.pi * (idx2 - 1) / dim[1]), (dim[0], 1, dim[2]))
    f3 = torch.tile(torch.cos(2 * torch.pi * (idx3 - 1) / dim[2]), (dim[0], dim[1], 1))

    f1[0, 0, 0] = 0  # Avoid division by 0

    # Compute the decomposition
    s = torch.fft.ifftn(torch.fft.fftn(v) / (6 - 2 * f1 - 2 * f2 - 2 * f3)).real

    # Return the result
    p = u - s
    return p

# Example usage
# u = torch.randn((depth, height, width), dtype=torch.float32)  # Example input
# p = perdecomp3D(u)