# import cupy as cp
import torch
import gc
import sys
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn.functional as F
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_dc_random_pair(H, W, dc_max=1.0, device=device):
    N = H * W

    # Random DC offset for x in [-dc_max, dc_max]
    # c_x = (2.0 * torch.rand(1, device=device) - 1.0) * dc_max  # scalar
    c_y = (2.0 * torch.rand(1, device=device) - 1.0) * dc_max # scalar

    # print(c_x)
    print(c_y)
    # i.i.d. noise in [-1, 1] for x, [0,1] for y
    noise_x = 2.0 * torch.rand(N, device=device) - 1.0
    noise_y = torch.rand(N, device=device)

    x_flat = noise_x

    y_flat = noise_y +c_y
    y_flat = torch.clamp(y_flat, 0, 1)


    x = x_flat.view(H, W)
    y = y_flat.view(H, W)
    return x, y


def digital_dot_product(A,B):
    l, s, _ = A.shape
    out = (A * B).sum(axis=(1, 2))   
    return out.reshape(int(np.sqrt(l)), int(np.sqrt(l))) / (s*s)

def quantise(x, lo, hi, levels):
    s = (levels - 1) / (hi - lo)
    return torch.round((x.clamp(lo, hi) - lo) * s) / s + lo

def define_limits_camera_ADC(dim, bright, low_pct=0.00, high_pct=1.00):
    dim    = torch.as_tensor(dim,    device=dim.device, dtype=dim.dtype)
    bright = torch.as_tensor(bright, device=dim.device, dtype=dim.dtype)

    eps = torch.finfo(dim.dtype).eps
    a, b = float(low_pct), float(high_pct)

    # span is always positive
    span = (bright - dim).clamp_min(eps) / max(b - a, eps)

    # initial lo/hi
    lo = dim - a * span
    # cap lo at 0.0 if negative (broadcast-safe)
    lo = torch.maximum(lo, torch.zeros_like(lo))
    # keep the same span
    hi = lo + span

    return lo, hi  # in [0, 2^bits - 1]

def read_camera_ADC(x, lo, hi, bits):
    levels = (1 << int(bits)) - 1
    codes = torch.round((x.clamp(lo, hi) - lo) * levels / (hi - lo)).to(torch.int32)
    return codes # in [0, 2^bits - 1]

    
def cleanup(device=None):
    gc.collect()
    if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    

def plot_figure(x, title="", cbar_label="", cmap='gray'):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    plt.figure()
    im = plt.imshow(x, cmap=cmap)
    plt.title(title, fontsize=16)
    plt.axis('off')
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=20, length=6, width=1) 
    plt.show()
    plt.close()

def calculate_errors(y_true, y_pred, lens_wise=False):
    r,l,_ = y_true.shape
    y_true = y_true.reshape(r, l*l).detach().cpu().numpy()
    y_pred = y_pred.reshape(r, l*l).detach().cpu().numpy()
    multioutput = 'raw_values' if lens_wise else 'uniform_average'
    axis = 0 if lens_wise else None

    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    r2 = r2_score(y_true, y_pred, multioutput=multioutput)
    err = y_pred - y_true
    snr = np.sqrt(np.mean(y_true**2, axis=axis)) / np.sqrt(np.mean(err**2, axis=axis))
    snr_db = 20*np.log10(snr)

    range_diff = y_true.max(axis=axis) - y_true.min(axis=axis)
    nrmse_range = rmse / range_diff if range_diff.all() != 0 else np.nan
    nrmse_mean = rmse / y_true.mean(axis=axis) if y_true.mean(axis=axis).all() != 0 else np.nan

    return {
        'MSE': mse.reshape(l,l),
        'RMSE': rmse.reshape(l,l),
        'MAE': mae.reshape(l,l),
        'R2': r2.reshape(l,l),
        'NRMSE (range normalized)': nrmse_range.reshape(l,l),
        'NRMSE (mean normalized)': nrmse_mean.reshape(l,l),
        'SNR_dB': snr_db.reshape(l,l)
    }


def save_fitting_plot(y, y_pred):
    plt.figure()
    plt.scatter(y, y_pred, color='blue', edgecolor='k', label='Fitted')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit (y=x)')
    plt.xlabel('Digital (True)')
    plt.ylabel('Fitted (Optical scaled)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fitting_plot.png")
    plt.close()
        
def log_to_file(kwargs, y, y_pred, errors):
    with open('log.txt', "w") as file:
        # Save the exact command line if available
        file.write("Called via function:\n\n")
        file.write(f"Command Line: {' '.join(sys.argv)}\n\n")
        # Save the configurations as key-value pairs
        for key, value in kwargs.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        if y is not None and y.shape[0]!=1:
            file.write("=== Errors ===\n")            
            for metric, value in errors.items():
                file.write(f"{metric}: {value}\n")
            file.write("\n")
            
        file.write("=== Data points ===\n")
        y = y.flatten()
        y_pred = y_pred.flatten()
        file.write("\n\n(Digital, Optical):\n")
        file.write("".join(f"({y[i]:.4f}, {y_pred[i]:.4f})" for i in range(len(y))) + "\n")
    save_fitting_plot(y,y_pred)

def create_outcome_dir(title):
    # Ensure base outcomes directory exists
    base_dir = 'outcomes'
    os.makedirs(base_dir, exist_ok=True)

    # Create base name with today's date and title
    date_str = datetime.today().strftime('%Y%m%d')
    base_name = f"{date_str}_{title}"

    # Determine a unique folder name by incrementing if needed
    count = 1
    folder_name = f"{base_name}_{count}"
    while os.path.exists(os.path.join(base_dir, folder_name)):
        count += 1
        folder_name = f"{base_name}_{count}"

    # Create the new directory and change working directory to it
    full_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_path)
    os.chdir(full_path)

    print(f"[INFO] Saving outputs to: {full_path}")
    return full_path  # Optional, in case you want to keep track

# def go_to_unlabeled_tests():
#     target_dir = os.path.join('outcomes', 'unlabeled_tests')
    
#     if not os.path.isdir(target_dir):
#         raise FileNotFoundError(f"[ERROR] Directory '{target_dir}' does not exist.")
    
#     os.chdir(target_dir)
#     print(f"[INFO] Changed working directory to: {target_dir}")
#     return target_dir

def go_to_unlabeled_tests():
    target_dir = os.path.join("outcomes", "unlabeled_tests")
    os.makedirs(target_dir, exist_ok=True)   # create if missing (incl. parents)
    os.chdir(target_dir)
    print(f"[INFO] Changed working directory to: {target_dir}")
    return target_dir

def coarse_uniform_01(H, W, cells=3, device='cpu'):
    # draw on a tiny grid, then bilinear upsample
    z = torch.rand(1, 1, cells, cells, device=device)
    y = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=True)[0,0]
    # strictly [0,1]
    return y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def auto_T_for_energy(ee_target: float = 0.98) -> float:
    """Return T so a 2D Gaussian truncated at R=T·σ contains ee_target encircled energy."""
    ee_target = float(min(max(ee_target, 0.5), 0.9999))
    return math.sqrt(-2.0 * math.log(1.0 - ee_target))

def gaussian_kernel_isotropic2d(sig_x_samp: float, sig_y_samp: float,
                                ee_target: float, device, ctype):
    """Build a normalized 2D Gaussian kernel (possibly elliptical in samples), auto-sized for >ee_target energy."""
    if sig_x_samp <= 0.0 or sig_y_samp <= 0.0:
        return torch.ones((1, 1, 1, 1), device=device, dtype=ctype)  # identity

    T = auto_T_for_energy(ee_target)
    Rx = T * sig_x_samp
    Ry = T * sig_y_samp
    kx = max(3, int(2 * math.ceil(Rx) + 1) | 1)  # odd, ≥3
    ky = max(3, int(2 * math.ceil(Ry) + 1) | 1)

    ax = torch.arange(kx, device=device, dtype=torch.float32) - (kx - 1) / 2
    ay = torch.arange(ky, device=device, dtype=torch.float32) - (ky - 1) / 2
    xx, yy = torch.meshgrid(ax, ay, indexing="xy")
    K = torch.exp(-(xx**2) / (2 * sig_x_samp**2) - (yy**2) / (2 * sig_y_samp**2))
    K = K / torch.clamp(K.sum(), min=1e-12)
    return K.to(ctype).unsqueeze(0).unsqueeze(0)  # [1,1,ky,kx]

def blur_phase_crosstalk_isotropic(phase_hw: torch.Tensor,
                                   sigma_slm_px: float,   # σ in SLM pixels (isotropic). 0 => no cross-talk
                                   pitch_m: float,        # SLM pixel pitch (m)
                                   dx_m: float, dy_m: float,  # high-res sampling (m/sample)
                                   ee_target: float = 0.98) -> torch.Tensor:
    """Isotropic cross-talk: blur e^{iφ} with a Gaussian whose σ is given in SLM pixels; wrap-safe."""
    if sigma_slm_px <= 0.0:
        return phase_hw  # no cross-talk

    # Convert isotropic σ in SLM pixels to samples along each axis (isotropic in *physical* space)
    sig_x_samp = (sigma_slm_px * pitch_m) / dx_m
    sig_y_samp = (sigma_slm_px * pitch_m) / dy_m

    ctype = (torch.complex64 if phase_hw.dtype in (torch.float16, torch.bfloat16, torch.float32)
             else torch.complex128)
    E = torch.exp(1j * phase_hw.to(torch.float32)).to(ctype)          # [H,W] complex
    K = gaussian_kernel_isotropic2d(sig_x_samp, sig_y_samp, ee_target,
                                    device=phase_hw.device, ctype=ctype)

    x = E.unsqueeze(0).unsqueeze(0)                                   # [1,1,H,W]
    pad_y, pad_x = K.shape[2] // 2, K.shape[3] // 2
    x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    y = F.conv2d(x, K, padding=0)[0, 0]                               # [H,W] complex

    y = y / torch.clamp(y.abs(), min=1e-12)                           # phase-only SLM
    return torch.angle(y).to(phase_hw.dtype)