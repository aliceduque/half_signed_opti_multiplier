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
    """
    Calculate the new effective focal distance, from the SLM, after lens spacing is disturbed by a2 and a3
    Args:
        f1, f2: focal distances os L1 and L2
        a2: lens placement deviation from L1 to pupil, if present, or Fourier plane
        a3: lens placement deviation from L2 to pupil, if present, or Fourier plane
        f_SLM: original programmed focal distance of SLM
    
    Returns:
        f_eff: New effective focal length
    """
    f_def = f1*f2/(a2*f1+a3*f2)
    f_eff = f_def*f_slm/(f_def+f_slm)
    return f_eff

def propagation(field, dmd, slm, waist, pupil_radius, misalign):
    """
    Propagate field from source to capture point, going through 4f system, DMD and SLM.
    Args:
        field: MonochromaticFieldTorch object
        dmd: DMD object
        slm: SLM object
        waist: beam waist at the point where it illuminates the DMD. If none, plane wave is assumed.
        pupil_radius: radius of a pupil to be placed on the Fourier plane (between L1 and L2). If none, no pupil is added.
        misalign: misalignment vector (a1,a2,a3,a4), corresponding to the 4 segments of the 4f-relay.
    
    Returns:
        f_eff: New effective focal length
    """

    a1, a2, a3, a4 = misalign
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = PlaneWaveTorch(device) if waist is None else GaussianBeamTorch(w0=waist, device=device)
    field.add(source)
    
    f_lens1 = f_lens2 = 120*mm
    radius_l1 = radius_l2 = 15*mm
    
    image_dmd_proc = dmd.print_image_to_dmd(field.dx, field.Nx)       
    field.E = field.E* image_dmd_proc.to(dtype=torch.complex64, device=device)

    del image_dmd_proc
    cleanup(device)
    
    field.propagate(f_lens1*(1+a1))

    field.add(Lens_torch(f=f_lens1, radius=radius_l1, x0=dmd.dmd.x0, y0=dmd.dmd.y0))
    field.propagate(f_lens1*(1+a2))

    if pupil_radius is not None:
        pupil = CircularAperture_torch(radius=pupil_radius)  
        field.add(pupil)
            
    field.propagate(f_lens2*(1+a3))
    
    field.add(Lens_torch(f=f_lens2, radius=radius_l2, x0=dmd.dmd.x0, y0=dmd.dmd.y0))
    
    field.propagate(f_lens2*(1+a4))

    slm.set_phase_mask()
    field.add(slm.get_device())
    
    slm_to_cam = slm.slm.focal_length if a2 == 0 and a3 == 0 else focal_dist_correction(f_lens1, f_lens2, a2, a3, slm.slm.focal_length)

    field.propagate(slm_to_cam)

    ## Uncomment to plot focal-spot field intensity ##
    # I = field.get_intensity()
    # field.plot_intensity(I,grid=True)

    return field
    
    
    
