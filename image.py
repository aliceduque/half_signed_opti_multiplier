import os, torch, random, numpy as np, torchvision
import torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from util import quantise, make_dc_random_pair, correlated_uniform_01, plot_figure,coarse_uniform_01

def set_input_type(input_type, image_dmd, image_slm, dataset=None, fully_signed=False):
    if input_type == 'fully_lit':
        image_dmd.set_all_ones()
        image_slm.set_all_ones()
    
    if input_type == 'frame_only':
        image_dmd.set_all_zeros()
        image_slm.set_all_zeros()
        
    if input_type == 'figure':
        index = random.randint(1, 2000)
        image_dmd.set_random_figure(dataset=dataset, start_index=index, N=image_dmd.num_lenslets**2)
        image_slm.set_random_figure(dataset=dataset, start_index=index-1, N=image_dmd.num_lenslets**2)
        image_slm.expand_to_negative_values()      
    
    if input_type == 'figure_sorted':
        l = image_dmd.num_lenslets ** 2
        index = random.randint(1, 2000)
        y = image_dmd.set_random_figure(dataset=dataset, start_index=index, N=image_dmd.num_lenslets**2)
        x = image_slm.set_random_figure(dataset=dataset, start_index=index-1, N=image_dmd.num_lenslets**2)
        x = image_slm.expand_to_negative_values()
        print('x shape = ', x.shape)
        
        x_sorted = torch.empty_like(x)
        y_sorted = torch.empty_like(y)
        
        for i in range(l):
            x_sorted[i], y_sorted[i] = sort_pair_along_zigzag(x[i], y[i])
            
        image_slm.image_raw = x
        image_slm.image_out = x_sorted
        image_dmd.image_raw = y
        image_dmd.image_out = y_sorted
        
        print('x sorted shape = ', x_sorted.shape)      

    
    if input_type == 'custom':
        image_dmd.set_custom_input(value=1.0)
        image_slm.set_custom_input(value=-0.4)
        
    if input_type == 'random':
        h = w = image_dmd.image_size
        b = image_dmd.bit_depth
        l = image_dmd.num_lenslets **2
        t = np.random.uniform(-0.6,0.6)
        if fully_signed:
            # pairs = [make_pair_with_target_mean(h, w, cells=10, t=t, bit_depth=b, device='cuda') for _ in range(l)]
            pairs = [make_pair_with_target_mean_mixture(h, w, cells=10, t=t, bit_depth=b, device='cuda') for _ in range(l)]
            
        else:
            pairs = [make_pair_with_target_mean_mixture_y01(h, w, cells=4, t=t, bit_depth=b, device='cuda') for _ in range(l)]
        X_list, Y_list = zip(*pairs)            # tuples of length B
        image_slm.image_out = torch.stack(list(X_list), dim=0)         # [B, H, W]
        image_dmd.image_out = torch.stack(list(Y_list), dim=0)
        # plot_figure(image_slm.image_out[0], title='slm')
        # plot_figure(image_dmd.image_out[0], title='dmd')
    
    
    if input_type == 'random_uncorr':
        h = w = image_dmd.image_size
        l = image_dmd.num_lenslets ** 2
        device = "cuda"
        
        dmd_raw_list = []
        slm_raw_list = []
        
        for _ in range(l):
            # x,y = make_dc_random_pair(h,w)
            x = 2.0 * torch.rand(h, w, device=device) - 1.0  # SLM [-1,1]
            x = quantise(x=x, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1)
            y = torch.rand(h, w, device=device)              # DMD [0,1]
            y = quantise(x=y, lo=0, hi=1, levels=2**(image_dmd.bit_depth)+1)

            slm_raw_list.append(x)
            dmd_raw_list.append(y)

        image_slm.image_out = torch.stack(slm_raw_list, dim=0)
        image_dmd.image_out = torch.stack(dmd_raw_list, dim=0)
        # plot_figure(image_slm.image_out[0], 'slm')
        # plot_figure(image_dmd.image_out[0], 'dmd')

        
    if input_type == 'random_uncorr_sorted':
        h = w = image_dmd.image_size
        l = image_dmd.num_lenslets ** 2
        device = image_dmd.device if hasattr(image_dmd, "device") else "cuda"

        slm_raw_list = []
        slm_sorted_list = []
        dmd_raw_list = []
        dmd_sorted_list = []

        for _ in range(l):
            # x,y = make_dc_random_pair(h,w)          
            x = 2.0 * torch.rand(h, w, device=device) - 1.0  # SLM [-1,1]
            x = quantise(x=x, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1)
            y = torch.rand(h, w, device=device)              # DMD [0,1]
            y = quantise(x=y, lo=0, hi=1, levels=2**(image_dmd.bit_depth)+1)
            x_sorted, y_sorted = sort_pair_along_zigzag(x, y)

            slm_raw_list.append(x)
            slm_sorted_list.append(x_sorted)
            dmd_raw_list.append(y)
            dmd_sorted_list.append(y_sorted)

        image_slm.image_raw = torch.stack(slm_raw_list, dim=0)
        image_slm.image_out = torch.stack(slm_sorted_list, dim=0)
        image_dmd.image_raw = torch.stack(dmd_raw_list, dim=0)
        image_dmd.image_out = torch.stack(dmd_sorted_list, dim=0)
        # plot_figure(image_slm.image_out[0], 'slm')
        # plot_figure(image_dmd.image_out[0], 'dmd')
        
    if input_type == 'random_DC':
        h = w = image_dmd.image_size
        l = image_dmd.num_lenslets ** 2
        device = "cuda"
        
        dmd_raw_list = []
        slm_raw_list = []
        
        for _ in range(l):
            x,y = make_dc_random_pair(h,w)
            # x = 2.0 * torch.rand(h, w, device=device) - 1.0  # SLM [-1,1]
            x = quantise(x=x, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1)
            # y = torch.rand(h, w, device=device)              # DMD [0,1]
            y = quantise(x=y, lo=0, hi=1, levels=2**(image_dmd.bit_depth)+1)

            slm_raw_list.append(x)
            dmd_raw_list.append(y)

        image_slm.image_out = torch.stack(slm_raw_list, dim=0)
        image_dmd.image_out = torch.stack(dmd_raw_list, dim=0)
        # plot_figure(image_slm.image_out[0], 'slm')
        # plot_figure(image_dmd.image_out[0], 'dmd')
        
    if input_type == 'figure_and_random':
        h = w = image_dmd.image_size
        device="cuda"
        index = random.randint(1, 2000)
        
        
        y = image_dmd.set_random_figure(dataset=dataset, start_index=index, N=image_dmd.num_lenslets**2)
        x = 2.0 * torch.rand(h, w, device=device) - 1.0  # SLM [-1,1]
        x_sorted, y_sorted = sort_pair_along_zigzag(x, y.squeeze(0))

        image_slm.image_out = quantise(x=x_sorted, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1).unsqueeze(0)
        image_dmd.image_out = y_sorted.unsqueeze(0)
        
        
    
        
def make_pair_mean_xpm1_y01(H, W, cells, t, bit_depth, device,
                            contrast=2.0, alpha=4.0):
    """
    x in [-1,1], y in [0,1]; target mean(x*y) ~ t. Quantise only at the end.
    """
    levels = 2 ** bit_depth

    # x ∈ [-1,1], strong contrast, then center so negatives exist
    x = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    if contrast != 1.0:
        x = torch.tanh(contrast * x)
    x = x - x.mean()                              # <-- critical for negatives
    x = x.clamp(-1.0, 1.0)                        # keep range

    # nonlinear coupling so y has healthy amplitude
    g = torch.tanh(alpha * x)                     # ∈ [-1,1], odd in x
    mu_xg = float((x.flatten() * g.flatten()).mean()) + 1e-12

    # set gamma from m = mean(x*y) = 0.5 * gamma * mean(x*g)  (since mean(x)≈0)
    gamma = (2.0 * float(t)) / mu_xg

    # build y ∈ [0,1] (elementwise clamp, don't clamp gamma)
    y = 0.5 * (1.0 + gamma * g)
    y = y.clamp(0.0, 1.0)

    # ---- FINAL quantization only ----
    xq = quantise(x, -1.0, 1.0, levels)          # directly to [-1,1]
    yq = quantise(y,  0.0, 1.0, levels)          # directly to [0,1]

    # sanity checks (comment out in production)
    # print('x pre-quant min/max:', float(x.min()), float(x.max()))
    # print('y pre-quant min/max:', float(y.min()), float(y.max()))
    # print('xq min/max:', float(xq.min()), float(xq.max()))
    # print('yq min/max:', float(yq.min()), float(yq.max()))
    # assert xq.min() < -1e-6, "xq has no negatives — something is re-mapping x later."
    return yq, xq

def make_pair_with_target_mean(H, W, cells, t, bit_depth, device, contrast=3.0):
    """
    Simple, no-iter version.
    Outputs x,y in [-1,1] so that mean(x*y) ~= t.
    Uses y = alpha * x with alpha = clip(t / mean(x^2), -1, 1).
    Quantisation happens ONLY at the end (directly in [-1,1]).
    """
    # 1) draw grayscale x in [-1,1] with good contrast (big amplitudes)
    x = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    if contrast != 1.0:
        x = torch.tanh(contrast * x)  # raises mu2 toward 1, still grayscale

    # 2) closed-form alpha (no iterations)
    xf  = x.flatten()
    mu2 = float((xf * xf).mean())  # mean(x^2) in (0,1]
    alpha = float(torch.clamp(torch.tensor(t / (mu2 + 1e-12), device=device), -1.0, 1.0))
    y = (alpha * x).clamp(-1.0, 1.0)

    # 3) FINAL quantization only (both directly to [-1,1])
    levels = 2 ** bit_depth
    xq = quantise(x.clamp(-1.0, 1.0), -1.0, 1.0, levels)
    yq = quantise(y.clamp(-1.0, 1.0), -1.0, 1.0, levels)

    return xq.reshape((H, W)), yq.reshape((H, W))

def make_pair_with_target_mean_mixture(
    H, W, cells, t, bit_depth, device, *,
    contrast=2.0,           # push values toward ±1 (like your original)
    diversity=0.8,          # strength of the independent component (0.3–1.0 is reasonable)
    orthogonalize=True      # make the diversity term orthogonal to x so E[x*z]=0 exactly
):
    """
    Outputs x,y in [-1,1] so that mean(x*y) ~= t, but with y = k*x + s*z.
    Quantisation happens ONLY at the end.
    API kept compatible with your original (extra kwargs have defaults), so it's drop-in.
    """
    levels = 2 ** bit_depth

    # --- draw x, z in [-1,1] (coarse), add contrast but stay grayscale
    x = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    z = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    if contrast != 1.0:
        x = torch.tanh(contrast * x)
        z = torch.tanh(contrast * z)

    xf = x.flatten()
    zf = z.flatten()

    # --- optionally make z orthogonal to x so mean(x*z)=0 exactly
    if orthogonalize:
        proj = (zf @ xf) / (xf @ xf + 1e-12)
        zf = zf - proj * xf
        # give z comparable RMS to x so 'diversity' has a consistent visual effect
        zf = zf / (zf.norm() + 1e-12) * (xf.norm() + 1e-12)

    z = zf.view_as(x)

    # --- choose k to hit the target mean-dot with the x term
    # mean(x*y) = k*mean(x^2) + s*mean(x*z). With orthogonalize=True the second term is ~0.
    mu2 = float((xf * xf).mean())  # in (0,1]
    k = float(t / (mu2 + 1e-12))

    # --- build y with diversity; clamp to valid range
    y = (k * x + diversity * z).clamp(-1.0, 1.0)

    # --- FINAL quantization only (both directly to [-1,1])
    xq = quantise(x.clamp(-1.0, 1.0), -1.0, 1.0, levels)
    yq = quantise(y.clamp(-1.0, 1.0), -1.0, 1.0, levels)

    return xq.reshape((H, W)), yq.reshape((H, W))

def make_pair_with_target_mean_mixture_y01(
    H, W, cells, t, bit_depth, device, *, contrast=2.0, diversity=0.8, orthogonalize=True, alpha=2.0
):
    """x ∈ [-1,1], y ∈ [0,1]; y = 0.5 + diversity*zc + 0.5*gamma*tanh(alpha*(x-mean(x)))."""
    levels = 2 ** bit_depth

    x = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    z = 2.0 * coarse_uniform_01(H, W, cells=cells, device=device) - 1.0
    if contrast != 1.0:
        x = torch.tanh(contrast * x)
        z = torch.tanh(contrast * z)

    x_c = x - x.mean()
    xcf, zf = x_c.flatten(), z.flatten()
    if orthogonalize:
        zf = zf - (zf @ xcf) / (xcf @ xcf + 1e-12) * xcf
    zf = zf - zf.mean()
    zf = zf / (zf.norm() + 1e-12) * (xcf.norm() + 1e-12)
    zc = zf.view_as(x_c)

    g = torch.tanh(alpha * x_c)
    mu_xg = float((x_c.flatten() * g.flatten()).mean()) + 1e-12
    gamma = float(2.0 * t / mu_xg)

    y = (0.5 + diversity * zc + 0.5 * gamma * g).clamp(0.0, 1.0)

    xq = quantise(x, -1.0, 1.0, levels)
    yq = quantise(y,  0.0, 1.0, levels)
    return xq.view(H, W), yq.view(H, W)

def make_pair_with_cosine(H, W, cells, c, bit_depth, device):
    # sample x on your coarse grid, quantize, map to [-1,1]
    x = coarse_uniform_01(H, W, cells=cells, device=device)
    levels = 2 ** bit_depth
    x = quantise(x, 0.0, 1.0, levels)
    x = (2*x - 1).flatten()

    # sample orthogonal-ish noise
    eps = coarse_uniform_01(H, W, cells=cells, device=device)
    eps = quantise(eps, 0.0, 1.0, levels)
    eps = (2*eps - 1).flatten()

    # remove x component from eps
    eps = eps - (eps @ x) / (x @ x + 1e-12) * x
    # combine to get desired cosine
    y = c * (x / (x.norm() + 1e-12)) + ( (1 - c**2) ** 0.5 ) * (eps / (eps.norm() + 1e-12))
    return x.reshape((H,W)), y.reshape((H,W))
        
def set_convolution_input(image_dmd, image_slm, kernel, dataset):
    index = random.randint(1, 2000)
    image_dmd.unwrap_convolution_windows(image_dmd.set_random_figure(dataset=dataset, start_index=index, N=1), kernel_size = kernel.shape[0])
    image_slm.set_kernel(kernel=kernel)

class QuickDrawDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        class_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])
        for class_idx, fname in enumerate(class_files):
            class_name = os.path.splitext(fname)[0]
            self.class_to_idx[class_name] = class_idx
            
            full_path = os.path.join(root_dir, fname)
            data = np.load(full_path)
            
            if max_per_class:
                data = data[:max_per_class]

            for img in data:
                self.samples.append((img, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_array, label = self.samples[idx]
        img_array = img_array.reshape(28, 28).astype(np.uint8)
        img = PILImage.fromarray(img_array, mode='L')  # use PILImage here
        if self.transform:
            img = self.transform(img)
        return img, label


def zigzag_path_indices(H, W, device=None):
    """
    Generate a 1D tensor of indices (length H*W) for a zigzag / diagonal-snake path
    from (0,0) to (H-1,W-1).

    We walk along diagonals with constant (row+col) = s, reversing every other
    diagonal to keep the path continuous.
    """
    indices = []

    for s in range(H + W - 1):
        diag = []
        for r in range(H):
            c = s - r
            if 0 <= c < W:
                diag.append(r * W + c)

        # reverse every other diagonal to make a continuous zigzag path
        if s % 2 == 1:
            diag.reverse()

        indices.extend(diag)

    return torch.tensor(indices, dtype=torch.long, device=device)


def sort_pair_along_zigzag(x, y):
    """
    x, y: 2D tensors of shape [H, W], on the same device.

    Returns:
        x_sorted, y_sorted: same shape, where
        - x is sorted ascending,
        - both x and y are placed along a zigzag path from top-left to bottom-right,
          and the *same* permutation is applied to y so pairings are preserved.
    """
    assert x.shape == y.shape, "x and y must have same shape"
    H, W = x.shape
    device = x.device

    # flatten
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    # sort x ascending, apply same index permutation to y
    idx_sort = torch.argsort(x_flat)
    x_sorted_vals = x_flat[idx_sort]
    y_sorted_vals = y_flat[idx_sort]

    # build zigzag path (indices into flattened image)
    path = zigzag_path_indices(H, W, device=device)  # [H*W]

    # place sorted values along the zigzag path
    x_out_flat = torch.empty_like(x_flat)
    y_out_flat = torch.empty_like(y_flat)
    x_out_flat[path] = x_sorted_vals
    y_out_flat[path] = y_sorted_vals

    # reshape back to [H, W]
    x_sorted = x_out_flat.view(H, W)
    y_sorted = y_out_flat.view(H, W)

    return x_sorted, y_sorted

class Image:
    def __init__(self, image_size, num_lenslets, bit_depth):
        self.image_size = image_size
        self.num_lenslets = num_lenslets
        self.bit_depth = bit_depth
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cifar = None
        self.quickdraw = None
        self.image_raw = None

    def get_image_raw(self):
        if self.image_raw is None:
            return self.image_out
        else:
            return self.image_raw

    def get_cifar(self):
        if self.cifar is None:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size),
                                  interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # -> [1,H,W] float32 in [0,1]
            ])
            self.cifar = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=tfm
            )
        return self.cifar

    def get_quickdraw(self, root="./data/QuickDraw", max_per_class=10):
        """
        Lazy-load and cache the QuickDraw dataset with the same preprocessing
        as CIFAR: grayscale -> resize -> ToTensor() -> [1,H,W] in [0,1].
        """
        if getattr(self, "quickdraw", None) is None:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # QuickDraw is already 'L', keeps it consistent
                transforms.Resize((self.image_size, self.image_size),
                                interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),                        # -> [1,H,W], float32 in [0,1]
            ])
            self.quickdraw = QuickDrawDataset(
                root_dir=root,
                transform=tfm,
                max_per_class=max_per_class
            )
        return self.quickdraw
    
    def get_celeb(self):
        if getattr(self, "celeb", None) is None:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size),
                                interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # [1,H,W] in [0,1]
            ])
            self.celeb = torchvision.datasets.CelebA(
                root=".//",
                split="train",      # "train" | "valid" | "test"
                download=False,
                transform=tfm,
                target_type=None
            )
        return self.celeb
    
    def get_celeb(self):
        if getattr(self, "celeb", None) is None:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size),
                                interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
            self.celeb = torchvision.datasets.ImageFolder(
                root="./data/celeba/img_align_celeba",
                transform=tfm
            )
        return self.celeb

    @staticmethod
    def expand_per_image(x):  # x: (N,H,W), min-max per image to [0,1]
        # per-image min/max
        mins = x.amin(dim=(1,2), keepdim=True)
        maxs = x.amax(dim=(1,2), keepdim=True)
        den  = (maxs - mins).clamp_min(1e-12)
        return ((x - mins) / den).clamp_(0.0, 1.0)
    
    def set_random_vector(self):
        B=self.num_lenslets**2
        vec = torch.stack(
            [coarse_uniform_01(self.image_size, self.image_size, cells=3, device=self.device)
            for _ in range(B)],
            dim=0)
        vec = self.expand_per_image(vec)
        levels = 2 ** self.bit_depth
        vec = quantise(vec,0.0, 1.0, levels)
        self.image_out = vec.to(self.device, non_blocking=True) 
        # plot_figure(self.image_out[0], title='raw')
        return self.image_out

    def set_random_figure(self, dataset, start_index, N):
        if dataset == "cifar":
            ds = self.get_cifar()
        elif dataset == 'quickdraw':
            ds = self.get_quickdraw()
        elif dataset == "celeb":
            ds = self.get_celeb()
        
        # N  = self.num_lenslets ** 2
        # make a contiguous block of indices, wrapping if needed
        idxs = [(start_index + i) % len(ds) for i in range(N)]
        # fetch all N samples, stack to (N,1,H,W)
        imgs = torch.stack([ds[i][0] for i in idxs], dim=0)  # CPU, float32
        imgs = imgs.squeeze(1)                               # (N,H,W)

        # optional per-image expand + quantize to bit_depth
        imgs = self.expand_per_image(imgs)
        levels = 2 ** self.bit_depth
        imgs = quantise(imgs,0.0, 1.0, levels)
        # imgs = torch.round(imgs * (levels - 1)) / (levels - 1)

        self.image_out = imgs.to(self.device, non_blocking=True)  # (N,H,W)
        # plot_figure(self.image_out[0])
        return self.image_out

    def set_all_zeros(self):
        self.image_out = torch.zeros((self.num_lenslets**2, self.image_size, self.image_size),
                                     device=self.device)
        return self.image_out

    def set_all_ones(self):
        self.image_out = torch.ones((self.num_lenslets**2, self.image_size, self.image_size),
                                    device=self.device)
        return self.image_out

    def set_custom_input(self, value):
        self.image_out = torch.zeros((self.num_lenslets**2, self.image_size, self.image_size),
                                     device=self.device)
        if value is not None:
            self.image_out[:,:,:] = value
        
        return self.image_out
    
    def set_kernel(self, kernel):
        self.image_out = kernel.unsqueeze(0).repeat(self.num_lenslets**2, 1, 1) 

    def expand_to_negative_values(self):
        self.image_out = self.image_out * 2 - 1.0
        # plot_figure(self.image_out[0], title='negatives')
        return self.image_out
    

    def unwrap_convolution_windows(self, img, kernel_size):
        self.img_full = img
        k = kernel_size
        # accept (H,W) or (1,H,W)
        x = img if img.dim() == 2 else img[0]            # (H,W)
        H, W = x.shape

        # SAME output size with stride=1 (asymmetric pad if k is even)
        pad_left  = k // 2
        pad_right = k - 1 - pad_left
        pad_top   = k // 2
        pad_bot   = k - 1 - pad_top

        # replicate-pad and extract all k×k windows centered at each pixel
        x4   = x.unsqueeze(0).unsqueeze(0)               # (1,1,H,W)
        xpad = F.pad(x4, (pad_left, pad_right, pad_top, pad_bot), mode='replicate')
        patches = xpad.unfold(2, k, 1).unfold(3, k, 1)   # (1,1,H,W,k,k)
        patches = patches.squeeze(0).squeeze(0)          # (H,W,k,k)
        patches = patches.contiguous().view(H*W, k, k)   # (N,k,k) with N = H*W

        self.image_out = patches
        return patches