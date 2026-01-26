import torch
import math

nm = 1e-9
um = 1e-6
mm = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"


sample_image = {
    'truck': 1,
    'car': 4,
    'horse': 7,
    'ship': 8,
    'frog': 22,
    'deer': 34,
    'cat': 38,
    'dog': 40,
    'bird': 18,
    'plane': 30
}

num_classes_dict = {
    'cifar': 10,
    'fashion': 10,
    'quickdraw': 20
}


opt_config_defaults = {
    "title": None,
    "image": None,
    "kernel": None,
    "kernel_is_phase": False,
    "calibrate": False,
    "show_or_save": 'save',
    "chequered": True,
    "capture_norm": True,
    "psnr": False,
    "signal_split": False,
    "dmd_encoding":True,
    "wlen": 850e-9,
    "focal_length": 4e-3,
    "slm_pixels": 1024,
    "active_area_ratio": 0.6,
    "dx": 0.5e-6,
    "Nx": 16384,
    "slm_pitch": 8e-6,
    "cam_pitch": 5.3e-6,
    "cluster_size": 4,
    "E0": None,
    "scale": None,
}
    


alphabet3x3 = {
    0: torch.tensor([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]),
    1: torch.tensor([[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]),
    2: torch.tensor([[1, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1]]),
    3: torch.tensor([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]),
    4: torch.tensor([[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]]),        
    5: torch.tensor([[1, 0, 1],
                 [0, 1, 0],
                 [1, 0, 1]]),
    6: torch.tensor([[0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]]),    
    7: torch.tensor([[0, 1, 1],
                 [1, 1, 1],
                 [1, 1, 0]]),
    8: torch.tensor([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]]),
    9: torch.tensor([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]])
}



alphabet_4x4 = {
    0: torch.tensor([[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]),

    1: torch.tensor([[0, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]),

    2: torch.tensor([[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]),

    3: torch.tensor([[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0]]),

    4: torch.tensor([[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 0, 0, 0]]),

    5: torch.tensor([[0, 0, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 0, 0]]),

    6: torch.tensor([[0, 1, 0, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 0, 0]]),

    7: torch.tensor([[0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 0, 0]]),

    8: torch.tensor([[0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0]]),

    9: torch.tensor([[0, 1, 1, 0],
                 [1, 1, 1, 0],
                 [0, 1, 1, 0],
                 [0, 1, 1, 0]]),

    10: torch.tensor([[0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0],
                  [0, 1, 1, 0]]),

    11: torch.tensor([[0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [1, 1, 1, 0],
                  [0, 1, 1, 0]]),

    12: torch.tensor([[0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0]]),

    13: torch.tensor([[1, 1, 1, 0],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0]]),

    14: torch.tensor([[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0]]),

    15: torch.tensor([[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 0]]),

    16: torch.tensor([[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]),
}


def generate_alphabet_16x16():
    alphabet = {}
    size = 16
    num_pixels = size * size

    indices = torch.arange(num_pixels)

    for v in range(num_pixels + 1):
        block_flat = torch.zeros(num_pixels, dtype=torch.uint8)

        if v > 0:
            n_on = min(v, num_pixels)
            block_flat[indices[:n_on]] = 1
        
        block = block_flat.view(size, size)
        alphabet[v] = block

    return alphabet

def generate_bayer_matrix(n):
    """
    Recursively generate a Bayer matrix of size n x n.
    n must be a power of 2.
    """
    if n == 1:
        return torch.tensor([[0]], dtype=torch.float32)
    else:
        smaller = generate_bayer_matrix(n // 2)
        tl = 4 * smaller
        tr = 4 * smaller + 2
        bl = 4 * smaller + 3
        br = 4 * smaller + 1
        top = torch.cat((tl, tr), dim=1)
        bottom = torch.cat((bl, br), dim=1)
        return torch.cat((top, bottom), dim=0)

def generate_dither_alphabet(cluster_size):
    size = cluster_size
    num_levels = cluster_size ** 2
    
    bayer = generate_bayer_matrix(size)
    bayer_scaled = (bayer / bayer.max()) * (num_levels-1)

    alphabet = {}
    
    for v in range(num_levels + 1):
        block = (bayer_scaled < v).to(torch.uint8)
        alphabet[v] = block
    
    return alphabet

def generate_linear_alphabet(size: int):
    n = size * size
    alphabet = {}
    for v in range(n + 1):
        flat = torch.zeros(n, dtype=torch.uint8)
        flat[:v] = 1  # works even when v=0
        alphabet[v] = flat.view(size, size)
    return alphabet

@torch.no_grad()
def build_arccos_lut(b,
                     device = device,
                     dtype = torch.float32):
    """
    Build a lookup table for θ = arccos(x) with θ-uniform binning.
    Returns:
      x_edges: 1D tensor shape [L+1], decreasing from +1 -> -1
      theta_vals: 1D tensor shape [L], θ (radians) for each bin (center value)
    Notes:
      L = 2**b bins. Built offline; multiplies here are fine.
    """
    L = 1 << b  # number of bins
    # Uniform in theta ⇒ edges at theta_k = k*pi/L
    k = torch.arange(L + 1, device=device, dtype=dtype)
    theta_edges = k * (math.pi / L)           # [0, π]
    x_edges = torch.cos(theta_edges)          # decreasing: +1 → -1

    # Bin centers (in x), then θ-values at those centers
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])   # still offline
    theta_vals = torch.arccos(x_centers)             # radians

    return x_edges, theta_vals

x_edges, theta_vals = build_arccos_lut(b=8, device=device, dtype=torch.float32)

def arccos_from_lut(s,
                    x_edges = x_edges,
                    theta_vals = theta_vals):
    """
    Multiply-free runtime arccos via LUT.

    Args:
      s: tensor of values in [-1, 1] (any shape, any float dtype)
      x_edges: from build_arccos_lut (shape [L+1], decreasing)
      theta_vals: from build_arccos_lut (shape [L], radians)

    Returns:
      theta ≈ arccos(s), same shape/dtype as theta_vals (broadcasted by indexing).
    """
    # 1) Clip to valid domain (comparisons only)
    v = torch.clamp(s, -1.0, 1.0)

    # 2) Find bin indices without multiplications.
    #    bucketize expects increasing edges; use the "-x" trick.
    idx = torch.bucketize(-v, -x_edges, right=False) - 1  # 0..L-1
    idx = idx.clamp_(0, theta_vals.numel() - 1)

    # 3) Lookup θ for each bin (no interpolation; centers were precomputed)
    theta = theta_vals[idx]
    return theta