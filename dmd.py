import torch
import random
import math
import torch.nn.functional as F
from util import plot_figure

class DMD:
    alphabet = None
    def __init__(self, pitch, num_pixels, frame_of_zeros, x0, y0):
        self.pitch = pitch
        self.num_pixels = num_pixels
        self.frame_of_zeros = frame_of_zeros
        self.x0 = x0
        self.y0 = y0


class DMD_Processing:
    def __init__(self, dmd, image):
        self.dmd = dmd
        self.image = image

    def verify_feasibility(self, vector_side, cluster_size, active_area_ratio, num_lenslets):
        # self.cluster_size = cluster_size
        self.num_lenslets = num_lenslets
        self.vector_side = vector_side
        self.active_area_window = vector_side * cluster_size
        self.total_area_window = int(self.active_area_window / math.sqrt(active_area_ratio))
        self.frame_width = (self.total_area_window - self.active_area_window) // 2
        self.total_area = num_lenslets * self.total_area_window
        if self.total_area > self.dmd.num_pixels:
            raise ValueError (f"Given configuration would need a DMD of size >={self.total_area}. The one provided has {self.dmd.num_pixels} pixels")
        else:
            print(f'[DMD] Feasibility verified. Frame width = {self.frame_width}px, total area occupied = {self.total_area}px')
    
    def print_image_to_dmd(self, field_dx, field_size):
        dmd_out = self.flip_image(self.tile_windows(self.frame_windows(self.modulate_clusters())))
        # plot_figure(dmd_out)
        return self.dmd_into_field(dmd_out, field_dx, field_size)
    
    def quantisation_levels(self, tensor, bit_depth):
        N = 2**bit_depth
        levels = torch.round(tensor * N).to(torch.uint16)
        return levels
    
    def flip_image(self, image):
        return torch.flip(image, dims=[0, 1])
    
    def modulate_clusters(self):
        # in:  self.image.image_out  -> (B, H, W)
        # out: (B, H*k, W*k) where k = self.cluster_size
        x = self.image.image_out
        B, H, W = x.shape
        A = torch.stack([self.dmd.alphabet[i] for i in sorted(self.dmd.alphabet)], dim=0).to(x.device)  # (L, k, k)
        levels = self.quantisation_levels(x, self.image.bit_depth).long()                               # (B, H, W)
        tiles = A[levels]                                                                               # (B, H, W, k, k)
        out = tiles.permute(0, 1, 3, 2, 4).reshape(B, self.active_area_window, self.active_area_window).float()                                 # (B, H*k, W*k)
        return out

    # def frame_windows(self, mod_image):
    #     # in:  (B, H, W)   (already expanded with clusters)
    #     # out: (B, H, W)   borders of width self.frame_width set to 1
    #     fw = int(self.frame_width)
    #     print('fw = ', fw)
    #     frame = 0 if self.dmd.frame_of_zeros else 1
        
    #     if fw <= 0:
    #         return mod_image
    #     out = mod_image.clone()
        
    #     out[:, :fw, :]  = frame
    #     out[:, -fw:, :] = frame
    #     out[:, :, :fw]  = frame
    #     out[:, :, -fw:] = frame
    #     return out
    
    def frame_windows(self, mod_image):
        """
        Append a frame of width self.frame_width around each (B,H,W) image.
        Output shape: (B, H + 2*fw, W + 2*fw).
        """
        fw = int(self.frame_width)
        if fw <= 0:
            return mod_image

        frame_val = 0 if getattr(self.dmd, "frame_of_zeros", False) else 1
        B, H, W = mod_image.shape

        out = mod_image.new_full((B, H + 2*fw, W + 2*fw), frame_val)
        out[:, fw:fw+H, fw:fw+W] = mod_image
        return out    
        

    def tile_windows(self, framed_windows):
        # in:  (B, h, w) with B a perfect square
        # out: (1, G*h, G*w) where G = sqrt(B)
        B, h, w = framed_windows.shape
        G = int(math.isqrt(B))
        assert G*G == B, "B must be a perfect square"
        grid = framed_windows.view(G, G, h, w).permute(0, 2, 1, 3).reshape(G*h, G*w)
        return grid
    
    def dmd_into_field(self, image, field_dx, field_size):
        # image: (H_dmd, W_dmd)  ->  field: (H_field, W_field)
        H_field = W_field = int(field_size)

        H_dmd, W_dmd = image.shape
        s = float(self.dmd.pitch) / float(field_dx)          # pixels->samples scale
        H_up = int(round(H_dmd * s))
        W_up = int(round(W_dmd * s))
        assert H_up <= H_field and W_up <= W_field, "Upsampled DMD exceeds field size."

        img_up = F.interpolate(image.unsqueeze(0).unsqueeze(0).float(),
                            size=(H_up, W_up), mode='nearest')[0, 0].to(image.dtype)

        field = image.new_zeros((H_field, W_field))
        yc = (H_field - H_up) // 2
        xc = (W_field - W_up) // 2
        
        
        y0 = yc + int(self.dmd.y0/field_dx)
        x0 = xc + int(self.dmd.x0/field_dx)
        
        field[y0:y0+H_up, x0:x0+W_up] = img_up
        return field
    
    # def modulate_clusters(self):
    #     B, H, W = self.image.image_out.shape
    #     alphabet = self.dmd.alphabet
    #     alphabet_stack = torch.stack([alphabet[i] for i in sorted(alphabet)]).to(self.image.image_out.device)
    #     levels = self.quantisation_levels(self.image.image_out, self.image.bit_depth).long()
    #     clusters = alphabet_stack[levels]  # [B, H, W, k, k]
    #     clusters = clusters.permute(0, 1, 3, 2, 4).reshape(B, H * self.cluster_size, W * self.cluster_size)
    #     return clusters.float()
    
    # def frame_windows(self, mod_image):
    #     B, H, W = mod_image.shape
    #     pixels_per_window = self.dmd.height // n
    #     print('frame width: ', frame_width)
    #     # print(f'pixels per window: {pixels_per_window}, frame width: {frame_width}')
    #     self.effective_area_ratio = (1 - 2 * frame_width / pixels_per_window) ** 2

    #     active_window_size = (pixels_per_window - 2 * frame_width,) * 2
    #     # print('active window size: ', active_window_size)
    #     out = torch.ones((self.dmd.height, self.dmd.width), dtype=mod_image.dtype, device=mod_image.device)
    #     if self.frame_of_zeros:
    #         out = torch.zeros((self.dmd.height, self.dmd.width), dtype=mod_image.dtype, device=mod_image.device)

    #     for i in range(self.num_lenslets):
    #         for j in range(self.num_lenslets):
    #             y0 = i * self.total_area_window + self.frame_width
    #             x0 = j * self.total_area_window + self.frame_width
    #             block = mod_image[:,
    #                          i * self.cluster_size * self.kernel_size : (i + 1) * self.cluster_size * self.kernel_size,
    #                          j * self.cluster_size * self.kernel_size : (j + 1) * self.cluster_size * self.kernel_size]
    #             block_up = self.upsample_to_shape(block, active_window_size)
    #             out[:, y0:y0 + block_up.shape[1], x0:x0 + block_up.shape[2]] = block_up
    #     return out


    
    