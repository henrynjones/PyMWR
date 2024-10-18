import torch

def get_3d_fft_freqs_on_grid(grid_size, device="cpu"):
    """
    Produces a 3D tensor with shape 'grid_size' whose entries are the spatial frequencies that correspond to the entries of a fourier transform computed with 'fft_3d'.
    """
    z = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[0]), device=device))
    y = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[1]), device=device))
    x = torch.fft.fftshift(torch.fft.fftfreq(int(grid_size[2]), device=device))
    grid = torch.cartesian_prod(z, y, x)
    return grid

def get_missing_wedge_mask(grid_size, mw_angle, device="cpu"):
    """
    Produces a 3D binary mask with shape 'grid_size', which can be used to zero-out Fourier components that lie inside a missing wedge with width 'mw_angle'.
    """
    grid = get_3d_fft_freqs_on_grid(grid_size=grid_size, device=device)
    # make normal vectors of two hyperplanes that bound missing wedge
    alpha = torch.deg2rad(torch.tensor(float(mw_angle))) / 2
    normal_left = torch.tensor([torch.sin(alpha), torch.cos(alpha)])
    normal_right = torch.tensor([torch.sin(alpha), -torch.cos(alpha)])
    # embed normal vectors into x-z plane
    normal_left = torch.tensor([normal_left[0], 0, normal_left[1]], device=device)
    normal_right = torch.tensor([normal_right[0], 0, normal_right[1]], device=device)
    # select all points that lie above or below both hyperplanes that bound missing wedge
    # convert to list because reshape needs list or tuple
    grid_size = [int(s) for s in grid_size]
    upper_wedge = torch.logical_or(
        grid.inner(normal_left) >= 0, grid.inner(normal_right) >= 0
    ).reshape(list(grid_size))
    lower_wedge = torch.logical_or(
        grid.inner(normal_left) <= 0, grid.inner(normal_right) <= 0
    ).reshape(list(grid_size))
    mw_mask = torch.logical_and(upper_wedge, lower_wedge).int()
    return mw_mask