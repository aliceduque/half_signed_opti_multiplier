# from file_ops import get_unique_filename
# import cupy as cp
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

# def capture_focal_intensity(
#     field, 
#     slm,
#     cam_pixel_pitch,
#     pix_block
#     ):
    
#     I = field.get_intensity()
#     empty_cache()
#     field.plot_intensity(I, grid=True, square_root=False)
#     field.plot_phase(field.E.cpu().numpy())
#     dx = field.dx
#     slm_pitch = slm.slm.pitch
#     H, W = I.shape
#     device = I.device
#     slm_pixels = slm.slm.Ny
#     N = slm.num_lenslets

#     # --- 1) Box-convolve by the camera pixel (integrate/average over pixel area) ---
#     K = cam_pixel_pitch / dx
#     kx = _box1d_kernel(K, device)                 # [1,1,Lx]
#     ky = _box1d_kernel(K, device)                 # [1,1,Ly]

#     I = I[None, None].to(torch.float32)       # [1,1,H,W]

#     # symmetric padding to avoid phase shift for non-integer K
#     pad_y = (ky.shape[2] - 1) // 2
#     pad_x = (kx.shape[2] - 1) // 2
#     Ipad  = F.pad(I, (pad_x, pad_x, pad_y, pad_y), mode='replicate')

#     Iy   = F.conv2d(Ipad, ky[:, :, :, None])      # [1,1,H,W]
#     Ibox = F.conv2d(Iy,   kx[:, :, None, :])[0, 0]# [H,W]; pixel-avg intensity (W/m^2)

#     del I
#     # --- 2) Microlens grid geometry ---
#     lenslet_size_m  = (slm_pixels * slm_pitch) / N
#     lenslet_size_px = int(round(lenslet_size_m / dx))

#     grid_px = lenslet_size_px * N
#     top  = (H - grid_px) / 2
#     left = (W - grid_px) / 2

#     # NEW: float centers at geometric middles (no "- 1")
#     cy = top  + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px -1
#     cx = left + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px -1
#     # --- 3) Bilinear sample at those centers ---
#     neighbours = pix_block // 2
#     captured_block = torch.zeros((N,N,pix_block,pix_block))
#     result = torch.zeros((N, N), dtype=torch.float32, device=device)
#     for i in range(N):
#         for j in range(N):
#             idx_y = (cy[i]-neighbours).long()
#             idx_x = (cx[j]-neighbours).long()            
#             for py in range(pix_block):
#                 for px in range(pix_block): 
#                     # v = bilinear_sample(Ibox, cy[i], cx[j])  # W/m^2 (pixel-avg intensity)
#                     captured_block[i,j,py,px] = Ibox[idx_y+py, idx_x+px]  # W/m^2 (pixel-avg intensity)

#             # result[i, j] = v * (cam_pixel_pitch ** 2)       # Watts in the pixel
#     print(captured_block)

#     # field.plot_intensity(intensity,grid=True)   

#     return captured_block


def capture_focal_intensity(field, slm, cam_pixel_pitch, pix_block, shot_noise, tau, fwc=10000, read_noise_std=1.5):
    """
    Emulate a camera:
    1) average intensity over a *single camera pixel area* using a separable box;
    2) then sample a p×p lattice of camera pixels around each focus,
       stepping by K = cam_pixel_pitch/dx (NOT by 1 grid cell).
    Returns: captured_block [N, N, p, p] with *per-camera-pixel averages*.
    """
    fwc = (cam_pixel_pitch *1e6)**2 * 700
    print('FWC = ', fwc)
    

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
    
    wlen = 650e-9
    # tau = (fwc/10000) * 10e-3
    efficiency = 0.55 * (5e-6 / cam_pixel_pitch)**2
    h = 6.626e-34
    c = 2.997e8
    
    if tau is None:
        tau = fwc*h*c/(efficiency*wlen*captured_block.max())
            
    I_to_N = efficiency * tau * wlen / (h * c)

    electrons_mean = captured_block * I_to_N
    print('electrons captured: ',electrons_mean)
    electrons_signal = torch.poisson(electrons_mean)
    electrons_signal = torch.clamp(electrons_signal, 0.0, fwc)
    print('electrons post shot: ', electrons_signal)
    if read_noise_std > 0.0:
        read_noise = torch.normal(
            mean=0.0,
            std=read_noise_std,
            size=electrons_signal.shape,
            device=electrons_signal.device,
            dtype=electrons_signal.dtype,
        )
        electrons_signal = electrons_signal + read_noise
        
    print('electrons post read: ', electrons_signal)
    if shot_noise:
        return electrons_signal, tau # [N,N,p,p], non-overlapping camera pixels
    else:
        return electrons_mean, tau

def normalise_to_E0(reading, E0, lo):
    return torch.sqrt(torch.clamp((reading + lo) / (E0 + lo), min=1e-12)) - 1.0


def find_scale_and_bias(correct, contestant):
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



# def capture_foci_intensities(field, num_lenses, cam_pixel_pitch, dx, norm, E0, scale):
#     """
#     Fully differentiable version!
#     Can either plot the center matrix or save the plot to disk.
#     """
#     # for 5um-6.9um camera pitch, E at 8192 resolution, E0 is 1620.9489746
#     # for same 5um pitch, E at 16834, E0 is 7259.2490234
#     # for 6.5um, at E=16384, E0=8809.3759765625
    
#     # E0 = 5558.569824218

#     capture_block = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)
#     # print('capture block = ', capture_block)
#     print(field.shape)
#     b, H, W = field.shape
#     assert H % num_lenses == 0 and W % num_lenses == 0, "field must divide evenly into blocks"
#     block_h = H // num_lenses
#     block_w = W // num_lenses
#     sums = []
#     for i in range(num_lenses):
#         for j in range(num_lenses):
#             y0 = i * block_h
#             x0 = j * block_w
#             cy = y0-1 + block_h // 2
#             cx = x0-1 + block_w // 2
#             region = field[cy - capture_block: cy + capture_block+1, cx - capture_block: cx + capture_block+1]
#             region_sum = region.sum()
#             sums.append(region_sum)

#     center_pixels = torch.stack(sums).reshape(num_lenses, num_lenses)
    
#     if norm:
#         center_pixels_norm = scale * (torch.sqrt(center_pixels / E0) - 1)
#         # center_pixels_norm = center_pixels
#         # center_pixels_norm = center_pixels / torch.max(torch.abs(center_pixels))
#     else:
#         center_pixels_norm = center_pixels

#     return center_pixels_norm

# def capture_foci_intensities(field, num_lenses, cam_pixel_pitch, dx, norm, E0, scale):
#     """
#     Vectorized, batched version. Input field shape: [B, H, W]
#     Returns: [B, num_lenses, num_lenses]
#     """
#     B, H, W = field.shape
#     assert H % num_lenses == 0 and W % num_lenses == 0, "field must divide evenly into blocks"

#     capture_block = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)
#     block_h = H // num_lenses
#     block_w = W // num_lenses

#     # Compute center positions of all lenses
#     cy = torch.arange(num_lenses, device=field.device) * block_h + block_h // 2 - 1
#     cx = torch.arange(num_lenses, device=field.device) * block_w + block_w // 2 - 1
#     grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")  # [num_lenses, num_lenses]
#     offsets = torch.arange(-capture_block, capture_block + 1, device=field.device)
    
#     # Shape: [K] where K = 2*capture_block+1
#     dy, dx_ = torch.meshgrid(offsets, offsets, indexing="ij")
#     dy = dy.flatten()
#     dx_ = dx_.flatten()

#     patch_size = len(dy)
#     num_zones = num_lenses * num_lenses

#     # Expand center coordinates: [num_lenses, num_lenses, patch_size]
#     y_coords = (grid_y.unsqueeze(-1) + dy).long()  # [num_lenses, num_lenses, patch_size]
#     x_coords = (grid_x.unsqueeze(-1) + dx_).long()

#     # Flatten all lens positions
#     y_coords = y_coords.reshape(-1, patch_size)  # [num_zones, patch_size]
#     x_coords = x_coords.reshape(-1, patch_size)

#     # Gather values using advanced indexing
#     # Looping over batch dimension
#     center_pixels = []
#     for b in range(B):
#         I = field[b]  # [H, W]
#         vals = I[y_coords, x_coords]  # [num_zones, patch_size]
#         region_sums = vals.sum(dim=1).reshape(num_lenses, num_lenses)  # [num_lenses, num_lenses]
#         center_pixels.append(region_sums)

#     center_pixels = torch.stack(center_pixels, dim=0)  # [B, num_lenses, num_lenses]

#     if norm:
#         center_pixels_norm = scale * (torch.sqrt(center_pixels / E0) - 1)
#     else:
#         center_pixels_norm = center_pixels

#     return center_pixels_norm


def capture_foci_intensities(field, num_lenses, cam_pixel_pitch, dx, norm, E0, scale, bias):
    """
    Extract summed intensities from a central camera region of the field.
    
    Args:
        field: [B, H, W] intensity field
        num_lenses: number of microlenses per row/col (assumed square)
        cam_pixel_pitch: physical pitch of camera pixels (m)
        dx: physical spacing between field samples (m)
        norm: whether to apply normalization
        E0: calibration constant
        scale: linear scale factor
        Nx_cam, Ny_cam: number of camera pixels (x, y)
    
    Returns:
        [B, num_lenses, num_lenses] summed intensities per lens region
    """
    
    def generate_lens_centers(first_center, spacing, num_lenses, device):
        y0, x0 = first_center
        offsets = torch.arange(num_lenses, device=device) * spacing
        cy = y0 + offsets
        cx = x0 + offsets
        return torch.meshgrid(cy, cx, indexing="ij")  # [num_lenses, num_lenses]

    print('num lenses: ', num_lenses)

    B, H, W = field.shape
    Nx_cam = Ny_cam = 2048
    # 1. Compute physical size of camera sensor
    sensor_height_m = Ny_cam * cam_pixel_pitch
    sensor_width_m = Nx_cam * cam_pixel_pitch
    
    flat_idx = torch.argmax(field)
    coords = torch.unravel_index(flat_idx, field.shape)
    print("Indices of max:", coords)  # prints (1, 0)
    
    # 2. Convert physical size to field grid units
    sensor_height_px = int(round(sensor_height_m / dx))
    sensor_width_px = int(round(sensor_width_m / dx))

    # assert sensor_height_px <= H and sensor_width_px <= W, "Camera sensor larger than field"

    # 3. Compute top-left corner of the camera region (centered)
    top = (H - sensor_height_px) // 2
    left = (W - sensor_width_px) // 2

    # 4. Divide camera region into lens-sized blocks
    block_h = sensor_height_px // num_lenses
    block_w = sensor_width_px // num_lenses

    # 5. How many pixels to sum around center of each lens block
    capture_block_y = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)
    capture_block_x = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)

    # Center of each lens block within the camera region
    # cy = torch.arange(num_lenses, device=field.device) * block_h + block_h // 2 + top
    # cx = torch.arange(num_lenses, device=field.device) * block_w + block_w // 2 + left
    # grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")

    # grid_y, grid_x = generate_lens_centers((3328, 3327), 512, num_lenses, field.device)
    
    cy = torch.arange(num_lenses, device=field.device) * block_h + block_h // 2 - 1
    cx = torch.arange(num_lenses, device=field.device) * block_w + block_w // 2 - 1
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
    # Define offset pattern for the camera pixel integration
    offsets_y = torch.arange(-capture_block_y, capture_block_y + 1, device=field.device)
    offsets_x = torch.arange(-capture_block_x, capture_block_x + 1, device=field.device)
    dy, dx_ = torch.meshgrid(offsets_y, offsets_x, indexing="ij")
    dy = dy.flatten()
    dx_ = dx_.flatten()

    patch_size = len(dy)
    num_zones = num_lenses * num_lenses

    # Combine lens centers with offsets
    y_coords = (grid_y.unsqueeze(-1) + dy).long().reshape(-1, patch_size)
    x_coords = (grid_x.unsqueeze(-1) + dx_).long().reshape(-1, patch_size)

    # Safeguard: avoid out-of-bound reads
    y_coords = y_coords.clamp(min=0, max=H-1)
    x_coords = x_coords.clamp(min=0, max=W-1)

    # Gather intensity values
    center_pixels = []
    for b in range(B):
        I = field[b]
        vals = I[y_coords, x_coords]  # [num_zones, patch_size]
        region_sums = vals.sum(dim=1).reshape(num_lenses, num_lenses)
        center_pixels.append(region_sums)

    center_pixels = torch.stack(center_pixels, dim=0)  # [B, num_lenses, num_lenses]

    # Normalize
    if norm:
        center_pixels = scale * (torch.sqrt(center_pixels / E0) - 1) + bias

    return center_pixels


def capture_foci_intensities_from_slm_old(
    field, 
    num_lenses, 
    slm_pixels, 
    slm_pitch, 
    dx, 
    cam_pixel_pitch, 
    norm=False, 
    E0=None,     # shape [N, N]
    scale=1.0, 
    bias=0.0
):
    """
    Capture a single [N, N] map of summed intensities for microlens foci.

    Args:
        field: [H, W] intensity field (no batch)
        num_lenses: number of microlenses along one axis (N)
        slm_pixels: total number of SLM pixels (assumed square)
        slm_pitch: SLM pixel pitch (in meters)
        dx: field sample spacing (m/pixel)
        cam_pixel_pitch: physical pitch of camera pixel (meters)
        norm: apply calibration normalization
        E0: [N, N] per-focus calibration constants
        scale: shared linear scale
        bias: shared linear bias

    Returns:
        [N, N] tensor of summed intensities
    """
    H, W = field.shape
    device = field.device
    N = int(math.sqrt(num_lenses))

    # Microlens block size in pixels
    lenslet_size_m = (slm_pixels * slm_pitch) / N
    lenslet_size_px = int(round(lenslet_size_m / dx))

    # Grid size and corner position (centered)
    grid_px = lenslet_size_px * N
    top = (H - grid_px) // 2
    left = (W - grid_px) // 2


    # Integration radius for cam pixel pitch
    capture_radius_px = round((cam_pixel_pitch / dx - 1) / 2)
    offset_range = torch.arange(-capture_radius_px, capture_radius_px + 1, device=device)
    dy, dx_ = torch.meshgrid(offset_range, offset_range, indexing="ij")
    dy = dy.flatten()
    dx_ = dx_.flatten()

    result = torch.zeros((N, N), dtype=torch.float32, device=device)

    for i in range(N):
        for j in range(N):
            # Center of each focus
            y0 = top + i * lenslet_size_px + lenslet_size_px // 2 -1
            x0 = left + j * lenslet_size_px + lenslet_size_px // 2 -1

            y_coords = (y0 + dy).clamp(0, H - 1).long()
            x_coords = (x0 + dx_).clamp(0, W - 1).long()

            intensity = field[y_coords, x_coords].sum()

            if norm:
                if E0 is None:
                    raise ValueError("E0 must be provided when norm=True")
                intensity = scale * (torch.sqrt(intensity / E0[i, j]) - 1) + bias

            result[i, j] = intensity

    return result

def capture_central_focus(field, cam_pixel_pitch, dx, norm, E0, scale, bias):
    """
    Processes a single 2D field (no batch dimension).
    
    Parameters:
        field (tensor) : [H, W] real-valued field (intensity)
        cam_pixel_pitch (float)
        dx (float)
        norm (bool)
        E0 (float)
        scale (float)
        bias (float)
        
    Returns:
        scalar (tensor) : sum of the central region or normalized value
    """

    H, W = field.shape

    # Define integration window size based on pixel pitch
    capture_block = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)

    # Central pixel position
    cy = H // 2 - 1
    cx = W // 2 - 1

    # Offsets to extract patch
    offsets = torch.arange(-capture_block, capture_block + 1, device=field.device)
    dy, dx_ = torch.meshgrid(offsets, offsets, indexing="ij")
    dy = dy.flatten()
    dx_ = dx_.flatten()

    # Coordinates for patch
    y_coords = (cy + dy).clamp(min=0, max=H - 1).long()
    x_coords = (cx + dx_).clamp(min=0, max=W - 1).long()

    # Extract patch
    vals = field[y_coords, x_coords]    # [patch_size]
    region_sum = vals.sum()             # scalar

    print('raw reading: ', region_sum.item())
    if E0 is None:
        print("No E0 value provided, thus function will output raw intensity reading (not normalised)")
    if norm:
        center_pixel = scale * (torch.sqrt(region_sum / E0) - 1) + bias
    else:
        center_pixel = region_sum

    return center_pixel


def capture_foci_intensities_from_slm(
    field, 
    num_lenses, 
    slm_pixels, 
    slm_pitch, 
    dx, 
    cam_pixel_pitch,
    norm=False, 
    E0=None,     # [N,N]
    scale=1.0, 
    bias=0.0,
    pixel_value="power",  # "power" (W), "avg" (W/m^2), or "sum_like_before"
):
    H, W = field.shape
    device = field.device
    N = int(math.sqrt(num_lenses))

    # --- 1) Box-convolve by the camera pixel (integrate/average over pixel area) ---
    K = cam_pixel_pitch / dx
    kx = _box1d_kernel(K, device)                 # [1,1,Lx]
    ky = _box1d_kernel(K, device)                 # [1,1,Ly]

    I = field[None, None].to(torch.float32)       # [1,1,H,W]

    # symmetric padding to avoid phase shift for non-integer K
    pad_y = (ky.shape[2] - 1) // 2
    pad_x = (kx.shape[2] - 1) // 2
    Ipad  = F.pad(I, (pad_x, pad_x, pad_y, pad_y), mode='replicate')

    Iy   = F.conv2d(Ipad, ky[:, :, :, None])      # [1,1,H,W]
    Ibox = F.conv2d(Iy,   kx[:, :, None, :])[0, 0]# [H,W]; pixel-avg intensity (W/m^2)

    # --- 2) Microlens grid geometry ---
    lenslet_size_m  = (slm_pixels * slm_pitch) / N
    lenslet_size_px = int(round(lenslet_size_m / dx))

    grid_px = lenslet_size_px * N
    top  = (H - grid_px) / 2
    left = (W - grid_px) / 2

    # NEW: float centers at geometric middles (no "- 1")
    cy = top  + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px
    cx = left + (torch.arange(N, device=device, dtype=torch.float32) + 0.5) * lenslet_size_px

    # --- 3) Bilinear sample at those centers ---
    def bilinear_sample(img, y, x):
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

    result = torch.zeros((N, N), dtype=torch.float32, device=device)
    for i in range(N):
        for j in range(N):
            v = bilinear_sample(Ibox, cy[i], cx[j])  # W/m^2 (pixel-avg intensity)

            if pixel_value == "power":
                v = v * (cam_pixel_pitch ** 2)       # Watts in the pixel
            elif pixel_value == "sum_like_before":
                v = v * (K * K)                      # to match old “sum of cells” scale

            if norm:
                if E0 is None:
                    raise ValueError("E0 must be provided when norm=True")
                v = scale * (torch.sqrt(torch.clamp(v / E0[i, j], min=1e-12)) - 1.0) + bias

            result[i, j] = v

    return result

# def capture_central_focus(field, cam_pixel_pitch, dx, norm, E0, scale, bias):
#     B, H, W = field.shape

#     # Define integration window size based on pixel pitch
#     capture_block = round(((cam_pixel_pitch / dx) - 1) / 2 - 1e-9)

#     # Central position
#     cy = H // 2 -1
#     cx = W // 2 -1
#     # Offsets to extract patch
#     offsets = torch.arange(-capture_block, capture_block + 1, device=field.device)
#     dy, dx_ = torch.meshgrid(offsets, offsets, indexing="ij")
#     dy = dy.flatten()
#     dx_ = dx_.flatten()

#     # Coordinates for central patch
#     y_coords = (cy + dy).clamp(min=0, max=H - 1).long()
#     x_coords = (cx + dx_).clamp(min=0, max=W - 1).long()

#     patch_size = len(dy)
#     center_pixels = []

#     for b in range(B):
#         I = field[b]
#         vals = I[y_coords, x_coords]  # [patch_size]
#         region_sum = vals.sum()       # scalar
#         center_pixels.append(region_sum)

#     center_pixels = torch.stack(center_pixels, dim=0)  # [B]

#     print('raw reading: ', center_pixels[0].item())
#     # print('norm readings with sqrt: ', (torch.sqrt(center_pixels / E0) - 1))
        
#     if norm:
#         center_pixels = scale * (torch.sqrt(center_pixels / E0) - 1) + bias

#     return center_pixels

def normalize_max_abs(img):
    """
    Normalize a Torch tensor by dividing by its maximum absolute value.
    """
    max_val = torch.max(torch.abs(img))
    return img / max_val if max_val != 0 else img

# def get_psnr(img1, img2):
#     """
#     Compute Peak Signal-to-Noise Ratio (PSNR) between two normalized Torch tensors.
#     """
#     max_val = torch.maximum(
#         torch.max(torch.abs(img1)),
#         torch.max(torch.abs(img2))
#     )
#     mse = torch.mean((img1 - img2) ** 2)
#     psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-12))  # small epsilon to avoid division by zero
#     return psnr, mse

def get_psnr(img1, img2):
    """
    Compute PSNR and MSE for each image in a batch.
    Inputs:
        img1, img2: Tensors of shape [B, H, W]
    Returns:
        psnr_vals: Tensor of shape [B]
        mse_vals: Tensor of shape [B]
    """
    assert img1.shape == img2.shape, "Input shapes must match"

    max_vals = torch.maximum(
        torch.amax(torch.abs(img1), dim=(1, 2)),
        torch.amax(torch.abs(img2), dim=(1, 2))
    )  # [B]

    mse_vals = torch.mean((img1 - img2) ** 2, dim=(1, 2))  # [B]

    psnr_vals = 20 * torch.log10(max_vals / torch.sqrt(mse_vals + 1e-12))  # [B]

    return psnr_vals, mse_vals  # each of shape [B]

def plot_differences(correct, contestant, title, save = False, plot=True):
    correct_norm = normalize_max_abs(correct)
    contestant_norm = normalize_max_abs(contestant)
    diff = correct_norm - contestant_norm
    
    
    
    vmax = torch.max(torch.abs(diff))
    vmin = -vmax
    plt.figure(figsize=(6, 6))
    plt.imshow(diff.detach().cpu().numpy(), cmap='seismic', vmin=vmin, vmax=vmax)
    plt.title(f"Digital - Optical difference ({title})")
    plt.axis("off")
    plt.colorbar()
    if save:
        os.makedirs("outcomes", exist_ok=True)
        filename = f"outcomes/{title}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_or_save(x, title, filename=None, plot=False, save=False, cmap='gray', symmetric_limits=False):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        # elif isinstance(x, cp.ndarray):
        #     x = cp.asnumpy(x)
            
        if save and filename is None:
            raise ValueError ('You must provide a valid filename to save the plot')
        
        plt.figure()
        
        imshow_kwargs = {'cmap': cmap}
        if symmetric_limits:
            vmax = np.max(np.abs(x))
            vmin = -vmax
            imshow_kwargs.update({'vmin': vmin, 'vmax': vmax})
        
        plt.imshow(x, **imshow_kwargs)
        plt.title(title)
        plt.axis('off')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)          # tick numbers bigger
        # Optional, if you use a label:
        # cbar.set_label("Units", fontsize=16)  
        if save:
            filename = get_unique_filename(filename, '.png')
            plt.savefig(filename, dpi=300)
        if plot:
            plt.show()
        plt.close()
                

# def compare_convolutions(correct, contestant, plot=False, save=False):
#     diff = correct - contestant
#     psnr, mse = get_psnr(correct,contestant)
#     if plot or save:
#         plot_or_save(diff, title=f'Element-wise difference (digital - optical), PSNR = {psnr:.2f}dB', filename='difference',
#                     plot=plot, save=save, cmap='seismic', symmetric_limits=True)
#         plot_or_save(contestant, title=f'Camera readings, PSNR = {psnr:.2f}dB', filename='opticonv', plot=plot, save=save)
        
#     return mse, psnr

def compare_convolutions(correct, contestant, plot=False, save=False):
    assert correct.shape == contestant.shape, "Shape mismatch between correct and contestant"
    B = correct.shape[0]

    diff = correct - contestant
    psnr, mse = get_psnr(correct, contestant)  # Assume it supports batches

    if plot or save:
        for i in range(B):
            plot_or_save(
                diff[i].detach().cpu().numpy(),
                title=f'Diff [#{i}] PSNR = {psnr[i]:.2f} dB',
                filename=f'difference_{i:03d}',
                plot=plot,
                save=save,
                cmap='seismic',
                symmetric_limits=True
            )
            plot_or_save(
                contestant[i].detach().cpu().numpy(),
                title=f'Optical Output [#{i}] PSNR = {psnr[i]:.2f} dB',
                filename=f'opticonv_{i:03d}',
                plot=plot,
                save=save
            )

    return mse, psnr