import torch
from diffractsim.diffractive_elements.lens import Lens_torch
from diffractsim.diffractive_elements.circular_aperture import CircularAperture_torch
from diffractsim.light_sources.gaussian_beam import GaussianBeamTorch
from diffractsim.light_sources.plane_wave import PlaneWaveTorch
from dict import nm,um,mm
from util import cleanup
from dmd import DMD_Processing
import os

def focal_dist_correction(f1,f2,a2,a3,f_slm):
    f_def = f1*f2/(a2*f1+a3*f2)
    f_eff = f_def*f_slm/(f_def+f_slm)
    return f_eff

def propagation(field, dmd, slm, waist, pupil_radii, misalign):

    # dmd = DMD_processing(dmd_dev, image_dmd)
    # dmd.verify_feasibiliy()
    # slm = SLM_processing(slm_dev, image_slm)
    # misalign = [-0.00, -0.00, -0.00, -0.00]
    a1, a2, a3, a4 = misalign
    # print("misalign: ", misalign)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = PlaneWaveTorch(device) if waist is None else GaussianBeamTorch(w0=waist, device=device)

    field.add(source)
    if dmd is None:
        slm.set_composite_phase_mask()
    
    else:
        
        if pupil_radii is not None:
            pupil = CircularAperture_torch(radius=pupil_radii)    
        
        f_lens1 = 120*mm
        radius_l1 = radius_l2 = 15*mm
        f_lens2 = 120*mm
        
        image_dmd_proc = dmd.print_image_to_dmd(field.dx, field.Nx)
        
        
        
        field.E = field.E* image_dmd_proc.to(dtype=torch.complex64, device=device)

        # I = field.get_intensity()
        # field.plot_intensity(I,grid=True)
        del image_dmd_proc
        cleanup(device)
        
        field.propagate(f_lens1*(1+a1))

        field.add(Lens_torch(f=f_lens1, radius=radius_l1, x0=dmd.dmd.x0, y0=dmd.dmd.y0))
        field.propagate(f_lens1*(1+a2))

        if pupil_radii is not None:
            field.add(pupil)
                
        field.propagate(f_lens2*(1+a3))
        
        field.add(Lens_torch(f=f_lens2, radius=radius_l2, x0=dmd.dmd.x0, y0=dmd.dmd.y0))
        
        field.propagate(f_lens2*(1+a4))
        # I = field.get_intensity()
        # field.plot_intensity(I,grid=True)
        # field.plot_phase(field.E.cpu().numpy())
        slm.set_phase_mask()

    

    field.add(slm.get_device())
    
    slm_to_cam = slm.slm.focal_length if a2 == 0 and a3 == 0 else focal_dist_correction(f_lens1, f_lens2, a2, a3, slm.slm.focal_length)

    print(slm_to_cam)
    # field.propagate(slm.slm.focal_length)
    field.propagate(slm_to_cam)


    
    # I = field.get_intensity()
    # field.plot_intensity(I,grid=True, slice_y_pos=0)

    return field
    
    
    
