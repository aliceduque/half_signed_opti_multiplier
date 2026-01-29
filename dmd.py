import torch, math, torch.nn.functional as F 
from util import plot_figure


class DMD:
    alphabet = None
    def __init__(self, pitch, num_pixels, x0, y0):
        self.pitch = pitch
        self.num_pixels = num_pixels
        self.x0 = x0
        self.y0 = y0


class DMD_Processing:
    def __init__(self, dmd, image):
        self.dmd = dmd
        self.image = image

    def verify_feasibility(self, vector_side, cluster_size, active_area_ratio, num_lenslets):
        """
        Verify feasibility of input configurations. Checks if the number of pixels in the DMD is sufficient.
        Args:
            vector_side: Square root of input vector size
            cluster_size: Block of cluster_size x cluster_size DMD pixels used to encode each vector element
            active_area_ratio: Fraction of DMD pixels used for modulation
            num_lenslets: number LxL of independent vectors to be encoded in the DMD at a time
        """
    
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
        
        ## Uncomment to plot DMD modulation
        # plot_figure(dmd_out)
        
        return self.dmd_into_field(dmd_out, field_dx, field_size)
    
    def quantisation_levels(self, tensor, bit_depth):
        """
            Levels of quantisation
        """    
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

    
    def frame_windows(self, mod_image):

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
        s = float(self.dmd.pitch) / float(field_dx)
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
