import torch, math

nm = 1e-9
um = 1e-6
mm = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    """
    Create an offline mapping of 2**b values to their corresponding mirror block representation.
    """
    size = cluster_size
    num_levels = cluster_size ** 2
    
    bayer = generate_bayer_matrix(size)
    bayer_scaled = (bayer / bayer.max()) * (num_levels-1)

    alphabet = {}
    
    for v in range(num_levels + 1):
        block = (bayer_scaled < v).to(torch.uint8)
        alphabet[v] = block
    
    return alphabet

@torch.no_grad()
def build_arccos_lut(b,
                     device = DEVICE,
                     dtype = torch.float32):
    """
    Build a lookup table for θ = arccos(x) with θ-uniform binning.
    Returns:
      x_edges: 1D tensor shape [L+1], decreasing from +1 -> -1
      theta_vals: 1D tensor shape [L], θ (radians) for each bin (center value)
    Notes:
      L = 2**b bins. Built offline; multiplies here are fine.
    """
    L = 1 << b 
    k = torch.arange(L + 1, device=DEVICE, dtype=dtype)
    theta_edges = k * (math.pi / L)
    x_edges = torch.cos(theta_edges)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:]) 
    theta_vals = torch.arccos(x_centers)

    return x_edges, theta_vals

x_edges, theta_vals = build_arccos_lut(b=8, device=DEVICE, dtype=torch.float32)

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
    v = torch.clamp(s, -1.0, 1.0)

    idx = torch.bucketize(-v, -x_edges, right=False) - 1 
    idx = idx.clamp_(0, theta_vals.numel() - 1)

    theta = theta_vals[idx]
    return theta