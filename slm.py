import torch
import math
import numpy as np
from dict import arccos_from_lut
from util import plot_figure, blur_phase_crosstalk_isotropic
from diffractsim.diffractive_elements import DOE
import torch.nn.functional as F


class SLM_Processing():
    def __init__(self, slm, image):
        self.slm = slm
        self.image = image
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        
    def verify_feasibility(self, vector_side, cluster_size, active_area_ratio, num_lenslets):
        # self.cluster_size = cluster_size
        self.num_lenslets = num_lenslets
        self.cluster_size = cluster_size
        self.vector_side = vector_side
        self.active_area_window = vector_side * cluster_size
        self.total_area_window = int(self.active_area_window / math.sqrt(active_area_ratio))
        self.frame_width = (self.total_area_window - self.active_area_window) // 2
        self.total_area = num_lenslets * self.total_area_window
        if self.total_area > self.slm.Ny:
            raise ValueError (f"Given configuration would need a SLM of size >={self.total_area}. The one provided has {self.slm.Ny} pixels")
        else:
            print(f'[SLM] Feasibility verified. Frame width = {self.frame_width}px, total area occupied = {self.total_area}px')
            
    def set_phase_mask(self):        
        phase_mask = self.tile_windows((self.create_lenslet() + self.image_to_phase_and_frame(self.image)))
        # plot_figure(torch.remainder(phase_mask, 2 * torch.pi)-torch.pi, "Phase Mask", "Phase (rad)")
        self.slm.set_mask(phase_mask)        

    def image_to_phase_and_frame(self, image):
        assert image.image_out.any() <= 1 and image.image_out.any() >= -1, \
            "Vector to be encoded in SLM falls outside [-1,1] limits"

        phase  = arccos_from_lut(image.image_out)
        device = image.image_out.device
        K, R, C = phase.shape
        # Base pattern
        half = self.cluster_size // 2
        row = torch.arange(self.cluster_size, device=device)
        row_sign = torch.where(row < half, 1.0, -1.0).view(-1, 1)
        base_pattern = row_sign.repeat(1, self.cluster_size)     # horizontal stripes ±1
        # Expand scalar kernel values into m×m blocks
        blocks = phase[..., None, None] * base_pattern  # [K, R, C, m, m]

        # [K, R, C, m, m] → [K, R*m, C*m]
        blocks = blocks.permute(0, 1, 3, 2, 4).reshape(K, R * self.cluster_size, C * self.cluster_size)

        if blocks.shape[-1] + 2 * self.frame_width != self.total_area_window:
            raise ValueError(
                f"Mismatch in modulation size: got {blocks.shape[-1] + 2 * self.frame_width}, "
                f"expected {self.total_area_window}"
            )

        # Pad with frame
        padded = torch.nn.functional.pad(
            blocks,
            (self.frame_width, self.frame_width, self.frame_width, self.frame_width),
            value=0.0
        ).to(device)
        

        return padded
    
    def create_lenslet(self):
        
        N = self.total_area_window

        k = 2 * torch.pi / self.slm.wlen

        x = torch.linspace(-(N - 1) / 2 * self.slm.pitch, (N - 1) / 2 * self.slm.pitch, N, device=self.device)
        y = torch.linspace(-(N - 1) / 2 * self.slm.pitch, (N - 1) / 2 * self.slm.pitch, N, device=self.device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')

        r2  = xx*xx + yy*yy

        phase_mask = -k* (torch.sqrt(self.slm.focal_length**2 + r2) - self.slm.focal_length)   
        
        replicas = phase_mask.unsqueeze(0).repeat(self.num_lenslets**2, 1, 1)  
        return replicas

    
    def tile_windows(self, framed_windows):
        B, h, w = framed_windows.shape
        G = int(math.isqrt(B))
        assert G*G == B, "B must be a perfect square"
        grid = framed_windows.view(G, G, h, w).permute(0, 2, 1, 3).reshape(G*h, G*w)
        return grid
    
    def get_device(self):
        return self.slm
    
    
class SLM(DOE):
    def __init__(self, Nx, Ny, pitch, fill_factor, wlen, focal_length, x0=0, y0=0, crosstalk_sigma=0.0):
        """
        Nx, Ny : number of SLM pixels in x and y
        pitch : physical size of each SLM pixel
        phase_mask : phase values (2D torch tensor, radians)
        fill_factor : fractional active area (0 to 1)
        x0, y0 : center offset (default 0,0)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.x0 = x0
        self.y0 = y0
        self.wlen = wlen
        self.focal_length = focal_length
        self.Nx = Nx
        self.Ny = Ny
        self.pitch = pitch
        self.fill_factor = fill_factor
        self.width = Nx * pitch
        self.height = Ny * pitch
        self.crosstalk_sigma_slm_px = crosstalk_sigma
        self.crosstalk_enabled = False if crosstalk_sigma == 0.0 else True

    def fast_generate_slm_grid_with_fill(self, dx, dy):
        """
        Create a high-resolution SLM grid using Kronecker tiling.
        Each SLM pixel is a square of '1's, surrounded by 0s (gaps) based on fill factor.
        """
        pitch_px_x = int(self.pitch // dx)
        pitch_px_y = int(self.pitch // dy)

        slit_px_x = int(pitch_px_x * np.sqrt(self.fill_factor))
        slit_px_y = int(pitch_px_y * np.sqrt(self.fill_factor))

        cell = torch.zeros((pitch_px_y, pitch_px_x), dtype=torch.float32, device=self.device)

        y0 = (pitch_px_y - slit_px_y) // 2
        x0 = (pitch_px_x - slit_px_x) // 2
        cell[y0:y0 + slit_px_y, x0:x0 + slit_px_x] = 1.0

        layout = torch.ones((self.Ny, self.Nx), dtype=torch.float32, device=self.device)

        # Kronecker product (tiling)
        slm_grid = torch.kron(layout, cell)

        return slm_grid

    def upscale_phase_mask(self, target_shape, phase_mask):
        Ny, Nx = phase_mask.shape
        H_target, W_target = target_shape

        scale_y = H_target // Ny
        scale_x = W_target // Nx

        expanded = phase_mask.repeat_interleave(scale_y, dim=0).repeat_interleave(scale_x, dim=1)
        return expanded[:H_target, :W_target]

    def embed_slm_into_field(self, slm_transmittance, full_shape):
        """
        Embed SLM region into the center of a larger zero-filled field.
        """
        H_full, W_full = full_shape
        H_slm, W_slm = slm_transmittance.shape

        field = torch.zeros((H_full, W_full), dtype=torch.complex64, device=self.device)

        y0 = (H_full - H_slm) // 2
        x0 = (W_full - W_slm) // 2

        field[y0:y0 + H_slm, x0:x0 + W_slm] = slm_transmittance

        return field
    
    def set_mask(self, phase_mask):
        self.phase_mask = phase_mask
        # plot_figure(torch.remainder(phase_mask, 2 * torch.pi), 'inside')
    
    def get_transmittance(self, xx, yy, λ, **kwargs):
        # dx = torch.abs(xx[0, 1] - xx[0, 0])
        # dy = torch.abs(yy[0, 0] - yy[1, 0])
        
        dx = float((xx[0, 1] - xx[0, 0]).abs().item())
        dy = float((yy[1, 0] - yy[0, 0]).abs().item())
        dx = round(dx, 7)
        dy = round(dy, 7)
        # dx = torch.floor(dx * 1e7) / 1e7
        # dy = torch.floor(dy * 1e7) / 1e7

        slm_grid = self.fast_generate_slm_grid_with_fill(dx, dy)

        # Accept externally passed phase mask, or fall back to internal one
        # phase_mask = kwargs["phase_mask"]
        expanded_phase = self.upscale_phase_mask(slm_grid.shape, self.phase_mask)
        if self.crosstalk_enabled and self.crosstalk_sigma_slm_px >= 0.0:
            expanded_phase = blur_phase_crosstalk_isotropic(
                expanded_phase,
                sigma_slm_px=self.crosstalk_sigma_slm_px,  # e.g., 0.0, 0.3, 0.5, 1.5
                pitch_m=self.pitch,
                dx_m=dx, dy_m=dy,
                ee_target=0.98   # auto-sizes kernel; >98% energy included
            )


        slm_transmittance = self.fast_generate_slm_grid_with_fill(dx, dy) * torch.exp(1j * expanded_phase)
        t = self.embed_slm_into_field(slm_transmittance, xx.shape)
        
        ## Uncomment for focal-spot intensity plot ##
        # plot_figure(torch.angle(t), title="SLM phase mask")

        return t
    
    
