import os, torch, random, torchvision, numpy as np, torch.nn.functional as F
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from util import quantise, plot_figure
from dict import DEVICE
def set_input_type(input_type, image_dmd, image_slm, dataset=None):

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
        
        x_sorted = torch.empty_like(x)
        y_sorted = torch.empty_like(y)
        
        for i in range(l):
            x_sorted[i], y_sorted[i] = sort_pair_along_zigzag(x[i], y[i])
            
        image_slm.image_raw = x
        image_slm.image_out = x_sorted
        image_dmd.image_raw = y
        image_dmd.image_out = y_sorted 
    
    if input_type == 'custom':
        # Set arbitrary uniform vectors for testing purposes
        image_dmd.set_custom_input(value=1.0)
        image_slm.set_custom_input(value=-0.4)
        
    if input_type == 'random_uncorr':
        h = w = image_dmd.image_size
        l = image_dmd.num_lenslets ** 2
        
        dmd_raw_list = []
        slm_raw_list = []
        
        for _ in range(l):
            x = 2.0 * torch.rand(h, w, device=DEVICE) - 1.0  # SLM [-1,1]
            x = quantise(x=x, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1)
            y = torch.rand(h, w, device=DEVICE)              # DMD [0,1]
            y = quantise(x=y, lo=0, hi=1, levels=2**(image_dmd.bit_depth)+1)

            slm_raw_list.append(x)
            dmd_raw_list.append(y)

        image_slm.image_out = torch.stack(slm_raw_list, dim=0)
        image_dmd.image_out = torch.stack(dmd_raw_list, dim=0)

    if input_type == 'random_uncorr_sorted':
        h = w = image_dmd.image_size
        l = image_dmd.num_lenslets ** 2

        slm_raw_list = []
        slm_sorted_list = []
        dmd_raw_list = []
        dmd_sorted_list = []

        for _ in range(l):
            x = 2.0 * torch.rand(h, w, device=DEVICE) - 1.0  # SLM [-1,1]
            x = quantise(x=x, lo=-1, hi=1, levels=2**(image_dmd.bit_depth)+1)
            y = torch.rand(h, w, device=DEVICE)              # DMD [0,1]
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

    return torch.tensor(indices, dtype=torch.long, device=DEVICE)


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
        self.device = DEVICE
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
   
    def get_celeb(self):
        if getattr(self, "celeb", None) is None:
            tfm = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.image_size, self.image_size),
                                interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
            
            self.celeb = torchvision.datasets.ImageFolder(
                root="./data/celeba/",
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

    def set_random_figure(self, dataset, start_index, N):
        if dataset == "cifar":
            ds = self.get_cifar()
        elif dataset == "celeb":
            ds = self.get_celeb()
        
        idxs = [(start_index + i) % len(ds) for i in range(N)]
        imgs = torch.stack([ds[i][0] for i in idxs], dim=0)  # CPU, float32
        imgs = imgs.squeeze(1)                               # (N,H,W)

        imgs = self.expand_per_image(imgs)
        levels = 2 ** self.bit_depth
        imgs = quantise(imgs,0.0, 1.0, levels)

        self.image_out = imgs.to(self.device, non_blocking=True)  # (N,H,W)
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
    

    def expand_to_negative_values(self):
        self.image_out = self.image_out * 2 - 1.0
        # plot_figure(self.image_out[0], title='negatives')
        return self.image_out
    