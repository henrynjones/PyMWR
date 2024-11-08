import numpy as np
from utils import dynamic, perdecomp3D, volxyz
import torch
import time
import bm4d
from bm4d.profiles import BM4DProfile, BM4DProfileLC

def mwr(Vin, sigma_noise, wedge, plot_flag=0, T=300, Tb=100, beta=0.00004):
    sigma_excite = torch.tensor([sigma_noise])
    #dim = Vin.shape
    y = Vin
    y_spectrum = torch.fft.fftn(y)
    x_mmse = torch.zeros_like(Vin)
    buffer = torch.zeros_like(Vin)
    reject_hist = torch.zeros(T)

    # Add noise
    x_initial = torch.fft.fftn(y + (sigma_excite * torch.normal(mean = torch.zeros_like(Vin),
                                                std = torch.ones_like(Vin))))
    #what if we could do element-wise characterisation of the std to be non-stationary?
    #would this violate the bm4d assumptions?

    x_initial[wedge == 1] = y_spectrum[wedge == 1]
    x_initial = torch.real(torch.fft.ifftn(x_initial))
    x_current = denoise(x_initial, sigma_noise)

    # Parameters for plot_flag=1
    plotrange = [torch.mean(y) - 5 * torch.std(y), torch.mean(y) + 5 * torch.std(y)]
    plotrangeS = dynamic.dynamic(torch.log(torch.abs(torch.fft.fftn(y))))
    global mplot
    mplot = 2
    global nplot
    nplot = 3

    normFactor = 1
    start_time = time.time()

    for t in range(T):
        # Add noise
        z = x_current + torch.normal(mean = torch.zeros_like(Vin),
                                    std = sigma_excite * torch.ones_like(Vin))
        # Spectrum constraint
        z_spectrum = torch.fft.fftn(z)
        z_spectrum[wedge == 1] = y_spectrum[wedge == 1]
        z = torch.real(torch.fft.ifftn(z_spectrum))

        # Denoise
        z = denoise(z, sigma_excite)
        z = perdecomp3D.perdecomp3D(z)

        # Compute energies
        Ucurrent = compute_energy(y, x_current, wedge)
        Uproposed = compute_energy(y, z, wedge)

        deltaU = Uproposed - Ucurrent
        deltaP = np.exp(-deltaU / beta)

        # Accept/reject according to Metropolis
        #print(Ucurrent, Uproposed, deltaU, deltaP)
        ak = np.random.rand() #unknown about this
        #print(ak, 'ak')
        if deltaU < 0:
            x_current = z
        else:
            if ak < deltaP:
                x_current = z
            else:
                reject_hist[t] = 1

        # Aggregation
        if t > Tb:
            buffer += x_current
            x_mmse = buffer / normFactor
            normFactor += 1

        # Plotting (if needed)
        if (t +1)% 5 == 0:
            plot_results(y, z, x_mmse, plotrange, plotrangeS, reject_hist, t)

        print(f'Iteration {t + 1} / {T} ...')

    exec_time = time.time() - start_time
    print(f'Execution time: {exec_time:.2f} seconds.')

    return x_mmse

def compute_energy(ref, data, wedge):

    N = torch.prod(torch.tensor([k for k in ref.shape]))
    refC = torch.real(torch.fft.ifftn(torch.fft.fftn(ref) * wedge))
    dataC = torch.real(torch.fft.ifftn(torch.fft.fftn(data) * wedge))
    U = torch.sum((refC - dataC) ** 2) / N
    return U

def denoise(dataIN, sigma_noise):
    profile = BM4DProfileLC()
    #profile = bm4d.BM4DProfile() #normal profile
    return torch.Tensor(bm4d.bm4d(dataIN, sigma_noise, profile, bm4d.BM4DStages.HARD_THRESHOLDING))

def plot_results(y, z, x_mmse, plotrange, plotrangeS, reject_hist, t):
    # Implement your plotting function here using matplotlib or any other library
    import matplotlib.pyplot as plt
    #might want to change to volxyz.volxyz() instead of this slicing stuff
    plt.figure(figsize=(12, 8))
    plt.subplot(mplot, nplot, 1)
    plt.imshow(y[:, :, y.shape[2] // 2], vmin=plotrange[0], vmax=plotrange[1])
    plt.title('y: input volume')

    plt.subplot(mplot, nplot, 2)
    plt.imshow(z[:, :, z.shape[2] // 2], vmin=plotrange[0], vmax=plotrange[1])
    plt.title(f'z: generated sample (reject: {reject_hist[t]})')

    plt.subplot(2, 3, 3)
    plt.imshow(x_mmse[:, :, x_mmse.shape[2] // 2], vmin=plotrange[0], vmax=plotrange[1])
    plt.title('x_mmse: estimator')

    plt.subplot(2, 3, 4)
    plt.imshow(torch.abs(torch.fft.fftn(y)[:, :, y.shape[2] // 2]), cmap='gray', vmin = plotrangeS[0], vmax = plotrangeS[1])
    plt.title('Fourier')

    plt.subplot(2, 3, 5)
    plt.imshow(torch.abs(torch.fft.fftn(z)[:, :, z.shape[2] // 2]), cmap='gray', vmin = plotrangeS[0], vmax = plotrangeS[1])
    plt.title('Fourier')

    plt.subplot(2, 3, 6)
    plt.imshow(torch.abs(torch.fft.fftn(x_mmse)[:, :, x_mmse.shape[2] // 2]), cmap='gray', vmin = plotrangeS[0], vmax = plotrangeS[1])
    plt.title('Fourier')

    plt.tight_layout()
    plt.pause(0.005)
