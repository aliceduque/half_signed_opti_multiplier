import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
from util import plot_figure
from dict import nm,mm,um
import os

import math, torch
import torch.nn.functional as F


def _box1d_kernel(pixel_pitch_over_dx: float, device):
    """
    Return a normalized 1D box kernel for K = pixel_pitch/dx (supports non-integer K). Used for separable box-averaging to emulate a camera pixel footprint.
    Args: 
        pixel_pitch_over_dx: ratio between pixel pitch and sampling interval dx
        device: device
    Returns:
        w: convolution kernel to create averaged representation
    """    

    K = float(pixel_pitch_over_dx)
    n_full = int(math.floor(K))
    frac = K - n_full
    w = torch.ones(n_full + 2, device=device, dtype=torch.float32)
    if n_full > 0:
        w[0]  = frac * 0.5
        w[-1] = frac * 0.5
    else:
        w[0]  = frac
        w[-1] = 1.0 - frac
    w = w / w.sum()           # average, not sum
    return w.view(1, 1, -1)   # [1,1,L]

def bilinear_sample(img, y, x):
    """
    Bilinearly sample a 2D image at (y, x)
    Args:
        img: 2D input image (field grid)
        x, y: coordinates in pixel units
    Returns:
        Interpolated output
    """    
    H, W = img.shape
    y0 = torch.clamp(y, 0, H - 1)
    x0 = torch.clamp(x, 0, W - 1)
    y1 = torch.floor(y0); x1 = torch.floor(x0)
    y2 = torch.clamp(y1 + 1, 0, H - 1); x2 = torch.clamp(x1 + 1, 0, W - 1)
    wy = y0 - y1; wx = x0 - x1
    return ((1-wy)*(1-wx)*img[y1.long(), x1.long()] +
            (1-wy)*wx     *img[y1.long(), x2.long()] +
                wy   *(1-wx) *img[y2.long(), x1.long()] +
                wy   *wx     *img[y2.long(), x2.long()])

def capture_focal_intensity(field, slm, cam_pixel_pitch, pix_block, shot_noise, tau, fwc, read_noise_std=1.5):
    """
    Emulate a camera at focal plane. Assumes symmetry around origin, be it for single or multiple lenslets configuration.
    Args:
        field: Optical field (MonochromaticFieldTorch object)
        slm: SLM object (to extract lenslet information)
        cam_pixel_pitch: Size of camera pixel
        pix_block: Side length of capture Region of Interest (ROI), in pixel. Expected to be an odd integer.
        shot_noise: Enable or disable shot noise from readings
        tau: Exposure time in seconds. If tau is None, the function undesrtand it's calibration time and calculates tau needed to reach FWC
        fwc: Camera's Full Well Capacity. If not specified (None), fwc is estimated based on pixel area
        read_noise_std: Standard deviation of read noise. This is enabled when shot noise is enabled.
    
    Returns:
        electrons_signal: If shot_noise is enabled, the noise reading is returned
        electrons_mean: If shot_noise is disabled, mean value is returned
        tau: Exposure time calculated (when input tau is None) is returned. After this first run, the same input tau is returned.
    """
    if fwc is None:
        fwc = (cam_pixel_pitch *1e6)**2 * 700
    

    I = field.get_intensity().to(torch.float32)  # [H,W], W/m^2
    # field.plot_intensity(I, grid=True, square_root=False)
    # field.plot_phase(field.E.cpu().numpy())
    dx = field.dx
    device = I.device

    # --- Average over one camera pixel footprint ---
    K = cam_pixel_pitch / dx  # step in grid cells per camera pixel
    kx = _box1d_kernel(K, device)                 # [1,1,Lx], normalized
    ky = _box1d_kernel(K, device)                 # [1,1,Ly], normalized

    I4   = I[None, None]                          # [1,1,H,W]
    pad_y = (ky.shape[2] - 1) // 2
    pad_x = (kx.shape[2] - 1) // 2
    Ipad  = F.pad(I4, (pad_x, pad_x, pad_y, pad_y), mode='replicate')

    Iy   = F.conv2d(Ipad, ky[:, :, :, None])      # [1,1,H,W]
    Ibox = F.conv2d(Iy,   kx[:, :, None, :])[0, 0]# [H,W] average W/m^2 over 1 camera pixel

    # --- Microlens grid geometry (centers in simulation-grid coordinates) ---
    slm_pitch = slm.slm.pitch
    slm_pixels = slm.slm.Ny
    N = slm.num_lenslets
    H, W = Ibox.shape

    lenslet_size_m  = (slm_pixels * slm_pitch) / N
    lenslet_size_px = int(round(lenslet_size_m / dx))
    grid_px = lenslet_size_px * N
    top  = (H - grid_px) / 2
    left = (W - grid_px) / 2

    # centers of each focus (float, in sim-grid pixels)
    cy = top  + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px - 1
    cx = left + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px - 1

    # --- Build a p×p *camera-pixel* lattice around each center ---
    # We want offsets spaced by K grid cells between camera pixels.
    p = int(pix_block)
    assert p >= 1 and p % 2 == 1, "use odd pix_block so the lattice is centered"
    half = p // 2
    # Offsets in *simulation-grid units* between camera pixels
    oy = torch.arange(-half, half+1, device=device, dtype=torch.float32) * K
    ox = torch.arange(-half, half+1, device=device, dtype=torch.float32) * K

    pixel_area = cam_pixel_pitch ** 2  # [m^2]
    captured_block = torch.empty((N, N, p, p), device=device, dtype=torch.float32)

    for i in range(N):
        for j in range(N):
            cy0, cx0 = cy[i], cx[j]
            # sample each camera pixel center with bilinear interpolation on Ibox
            for yi in range(p):
                y = cy0 + oy[yi]
                # vectorize x sampling across the row
                x_row = cx0 + ox
                # bilinear_sample doesn't vectorize across x, so loop (short p)
                row_vals = [bilinear_sample(Ibox, y, x_row[k]) for k in range(p)]
                captured_block[i, j, yi, :] = torch.stack(row_vals)  * pixel_area # Watts per camera pixel
    # print(captured_block)
    
    wlen = field.λ
    efficiency = 0.7
    h = 6.626e-34
    c = 2.997e8
    
    if tau is None:
        tau = fwc*h*c/(efficiency*wlen*captured_block.max())
            
    I_to_N = efficiency * tau * wlen / (h * c)

    electrons_mean = captured_block * I_to_N
    print('Electrons Captured (Mean): \n',electrons_mean.squeeze().detach().cpu().numpy())
    if shot_noise:
        electrons_signal = torch.poisson(electrons_mean)
        electrons_signal = torch.clamp(electrons_signal, 0.0, fwc)
        print('Electrons Post Shot Noise: \n', electrons_signal.squeeze().detach().cpu().numpy())
        if read_noise_std > 0.0:
            read_noise = torch.normal(
                mean=0.0,
                std=read_noise_std,
                size=electrons_signal.shape,
                device=electrons_signal.device,
                dtype=electrons_signal.dtype,
            )
            electrons_signal = electrons_signal + read_noise
            
        print('Electrons Post Read Noise: \n', electrons_signal.squeeze().detach().cpu().numpy())
        return electrons_signal, tau
    
    else:
        return electrons_mean, tau

def normalise_to_E0(reading, E0, lo):
    """
    Calculate n from the electrons reading
    Args:
        reading: Number of captured photoelectrons
        E0: Reference intensity
        lo: Offset value correspondent to zero, obtained in calibration
    
    Returns:
        n
    """
    
    return torch.sqrt(torch.clamp((reading + lo) / (E0 + lo), min=1e-12)) - 1.0


def find_scale_and_bias(correct, contestant):
    """
    Linear fitting between obtained optical values and expected digital dot products
    Args:
        correct: Expected result (digital dot product)
        contestant: Measured result (optical dot product)
    Returns:
        scale, bias: linear fitting parameters
    """
    side = contestant.shape[-1]
    scale = np.empty((side,side))
    bias = np.empty((side,side))
    correct = correct.detach().cpu().numpy()
    contestant = contestant.detach().cpu().numpy()
    for i in range(side):
        for j in range(side):
            y = correct[:, i, j]
            x = contestant[:, i, j]
            A = np.vstack([x, np.ones_like(x)]).T
            scale[i,j], bias[i,j] = np.linalg.lstsq(A, y, rcond=None)[0]
    return torch.tensor(scale),torch.tensor(bias)