import argparse
import math
import torch
import time
import random
import os
import gc
# import psutil
from dict import generate_linear_alphabet, generate_dither_alphabet
from slm import SLM, SLM_Processing
from dmd import DMD, DMD_Processing
from image import Image, set_input_type
from util import digital_dot_product, set_seed, log_to_file, go_to_unlabeled_tests, create_outcome_dir, calculate_errors, read_camera_ADC, define_limits_camera_ADC
from capture_and_post_processing import capture_focal_intensity, normalise_to_E0, find_scale_and_bias
from propagation import propagation
from diffractsim.monochromatic_simulator import MonochromaticFieldTorch 
torch.set_grad_enabled(False)


    
def opti_product(kwargs):
    title             = kwargs.get("title")
    lenslets          = kwargs.get("lenslets")
    input_type        = kwargs.get("input_type")
    dataset           = kwargs.get("dataset")
    input_size        = kwargs.get("input_size")
    chequered         = kwargs.get("chequered")
    wlen              = kwargs.get("wlen")
    focal_length      = kwargs.get("focal_length")
    dmd_pixels        = kwargs.get("dmd_pixels")
    slm_pixels        = kwargs.get("slm_pixels")
    active_area_ratio = kwargs.get("active_area_ratio")
    dx                = kwargs.get("dx")
    Nx                = kwargs.get("Nx")
    slm_pitch         = kwargs.get("slm_pitch")
    dmd_pitch         = kwargs.get("dmd_pitch")
    cam_adc_bits      = kwargs.get("cam_adc_bits")
    pix_block_size    = kwargs.get("pix_block_size")
    cam_pitch         = kwargs.get("cam_pitch")
    cluster_size      = kwargs.get("cluster_size")
    scale             = kwargs.get("scale")
    bias              = kwargs.get("bias")
    frame_of_zeros    = kwargs.get("frame_of_zeros")
    runs              = kwargs.get("runs")
    find_linear_fit   = kwargs.get("find_linear_fit")
    seed              = kwargs.get("seed")
    E0                = kwargs.get("E0")
    pupil             = kwargs.get("pupil")
    waist             = kwargs.get("waist")
    crosstalk         = kwargs.get("crosstalk")
    xy_misalign       = kwargs.get("xy_misalign")
    zoom_misalign     = kwargs.get("zoom_misalign")
    shot_noise        = kwargs.get("shot_noise")
    fwc               = kwargs.get("fwc")
    tau               = kwargs.get("tau")

    scale = torch.tensor(scale)
    bias = torch.tensor(bias)
    DMD.alphabet = generate_linear_alphabet(cluster_size) if chequered else generate_dither_alphabet(cluster_size)
    tau = None
    if shot_noise is None:
        shot_noise = False
    if xy_misalign is None:
        xy_misalign = [0,0]
    if zoom_misalign is None:
        zoom_misalign = [0,0,0,0]
    x0_dmd = xy_misalign[0]*dmd_pixels*dmd_pitch
    y0_dmd = xy_misalign[1]*dmd_pixels*dmd_pitch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        set_seed(seed)
            
    dmd_device = DMD(pitch=dmd_pitch,
                     num_pixels=dmd_pixels,
                     frame_of_zeros=frame_of_zeros,
                     x0=x0_dmd,
                     y0=y0_dmd)
    
    slm_device = SLM(pitch=slm_pitch,
                    Nx=slm_pixels,
                    Ny=slm_pixels,
                    fill_factor=0.956,
                    wlen=wlen,
                    focal_length=focal_length,
                    chequered=chequered,
                    crosstalk_sigma=crosstalk)
    
    image_dmd = Image(image_size = input_size,
                      num_lenslets = lenslets,
                      bit_depth = 2*int(math.log2(cluster_size)))
    
    image_slm = Image(image_size = input_size,
                      num_lenslets = lenslets,
                      bit_depth = 2*int(math.log2(cluster_size)))
    
        
    field = MonochromaticFieldTorch(wavelength=wlen,
                                extent_x=dx*Nx,
                                extent_y=dx*Nx,
                                Nx=Nx,
                                Ny=Nx,
                                intensity=0.275)    
    
    dmd_proc = DMD_Processing (dmd_device, image_dmd)
    slm_proc = SLM_Processing (slm_device, image_slm)
    
    
    dmd_proc.verify_feasibility(vector_side=input_size,
                                cluster_size=cluster_size,
                                active_area_ratio=active_area_ratio,
                                num_lenslets=lenslets)

    slm_proc.verify_feasibility(vector_side=input_size,
                                cluster_size=cluster_size,
                                active_area_ratio=active_area_ratio,
                                num_lenslets=lenslets)
    
    
    def propagate_and_read(quantised_reading=False, norm_to_E0=False, misalign=zoom_misalign):
        with torch.inference_mode():
            field.reset()
            propagation(field=field,
                        dmd = dmd_proc,
                        slm = slm_proc,
                        waist = waist,
                        pupil_radii = pupil,
                        misalign = misalign)
            print('shot noise: ', shot_noise)
            reading, tau_out = capture_focal_intensity(field = field,
                                        slm = slm_proc,
                                        tau=tau,
                                        fwc=fwc,
                                        shot_noise=shot_noise,
                                        cam_pixel_pitch = cam_pitch,
                                        pix_block = pix_block_size)
            if not quantised_reading:
                return reading, tau_out
            else:
                reading_quant = read_camera_ADC(reading,cam_low,cam_high,bits=cam_adc_bits)
                if norm_to_E0:
                    opti_result = normalise_to_E0(reading_quant.sum(dim=(-2, -1)), E0, lo=DV_offset)
                    digi_result = digital_dot_product(image_dmd.get_image_raw(), image_slm.get_image_raw())
                    return opti_result, digi_result
                else:
                    return reading_quant
               
                
    calibration_rep = 5 if shot_noise else 1
    
    
# Camera calibration: defining BRIGHT level
    set_input_type("fully_lit", image_dmd, image_slm)
    image_dmd.set_all_ones()
    image_slm.set_all_ones()


    E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=device, dtype=torch.float32)
    tau_acc = 0
    for _ in range(calibration_rep):
        E_run, tau_run = propagate_and_read()
        E_capt += E_run
        tau_acc += tau_run
        

    E_bright = (E_capt / calibration_rep).max()
    tau = tau_acc / calibration_rep
    print("Exposure time tau = ", tau)
    kwargs.update({"tau": tau})
    print(f"Upper threshrold E_bright = {E_bright.detach().cpu().tolist()}")
        
# Camera calibration: defining BLACK level
    image_dmd.set_all_ones()
    image_slm.set_custom_input(value=-1.0)
    E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=device, dtype=torch.float32)
    for _ in range(calibration_rep):
        E_capt += propagate_and_read()[0]

    E_dim = (E_capt / calibration_rep).min()
    # E_dim = torch.tensor(10.0, device='cuda')
    # print("E_dim pushed to 10 electrons")
    print(f"Lower threshrold E_dim = {E_dim.detach().cpu().tolist()}")

    cam_low, cam_high = define_limits_camera_ADC(dim=E_dim, bright=E_bright)
    
    print(f'cam high: {cam_high}, cam low: {cam_low}')
    DV_offset = (2**cam_adc_bits - 1) * cam_low / (cam_high - cam_low + 1e-9)
    print('DV offset: ',DV_offset)
    
# System calibration: defining reference intensity (E0)

    if E0 is None:
        set_input_type("frame_only", image_dmd, image_slm)
        E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=device, dtype=torch.float32)
        for _ in range(calibration_rep):
            E_capt += propagate_and_read(quantised_reading=True)

        E0 = (E_capt / calibration_rep).sum(dim=(-2,-1))
        
        print(f"Calibration complete. E0 = {E0.detach().cpu().tolist()}")
        kwargs.update({"E0": E0})

# Finding linear parameters (least squares)
    if find_linear_fit:
        correct    = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        contestant = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        for run in range(runs):
            set_input_type(input_type, image_dmd, image_slm, fully_signed=False, dataset=dataset)
            opti_result, digi_result = propagate_and_read(quantised_reading=True, norm_to_E0=True)
            contestant[run] = opti_result
            correct[run] = digi_result
            print(f'[Run {run+1}/{runs}] \nOptical result = \n{opti_result.detach().cpu().tolist()} \nDigital result = \n{digi_result.detach().cpu().tolist()}')
            
        scale, bias = find_scale_and_bias(correct,contestant)
        kwargs.update({"scale": scale, "bias": bias})
        print(f'Linear fitting done: scale = {scale} | bias = {bias}')
        fitted_optical = scale * contestant + bias
        error_metrics = calculate_errors(correct, fitted_optical, lens_wise=True)
        print(error_metrics)      
        
# If scale and biases pre-defined, then run:
    else:
        correct = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        fitted_optical = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        for run in range(runs):
            set_input_type(input_type, image_dmd, image_slm, dataset, fully_signed=False)
            opti_result, digi_result = propagate_and_read(quantised_reading=True, norm_to_E0=True)
            opti_result = scale*opti_result + bias
            fitted_optical[run] = opti_result
            correct[run] = digi_result
            # print(f'Opti result: {opti_result} // Digit result: {digi_result}')        

    if title is not None:
        create_outcome_dir(title)
    else:
        go_to_unlabeled_tests()
    log_to_file(kwargs, correct, fitted_optical)
    
    return error_metrics

        
if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(description="Perform a convolution using free space optics")
    parser.add_argument("--title", default=None, type=str, help="Test title (outcomes folder name). If left blank, outcomes will be saved in \'unlabeled tests\' folder")
    parser.add_argument("--lenslets", default=1, type=int, help="Number of parallel dot products L**2 to be calculated")
    parser.add_argument("--input_type", default="random_uncorr", type=str, help="Options: \'fully_lit\', \'frame_only\', \'custom\', \'random\'")
    parser.add_argument("--dataset", default="cifar", type=str, help="Dataset for Random input type (options: CIFAR and QuickDraw)")
    parser.add_argument("--input_size", default=45, type=int, help="Vector size to be operated (per side)")
    parser.add_argument("--chequered", default=False, type=bool, help="If True, chequered phase mask is used; if False, striped (ordered dithering) is used")  
    parser.add_argument("--wlen", default=650e-9, type=float, help="Beam wavelength")  
    parser.add_argument("--focal_length", default=425e-3, type=float, help="Focal length of SLM lens phase profile")
    parser.add_argument("--dmd_pixels", default=1024, type=int, help="Number NxN of pixels in the DMD (per dimension)")
    parser.add_argument("--slm_pixels", default=1024, type=int, help="Number NxN of pixels in the SLM (per dimension)")
    parser.add_argument("--active_area_ratio", default=0.4943, type=float, help="Active pixel area / total pixel area. Defines frame width")
    parser.add_argument("--dx", default=1.0e-6, type=float, help="Field grid resolution")
    parser.add_argument("--Nx", default=16384, type=int, help="Field grid size (number of cells Nx x Nx)")
    parser.add_argument("--slm_pitch", default=8e-6, type=float, help="SLM pixel pitch")
    parser.add_argument("--dmd_pitch", default=8e-6, type=float, help="DMD pixel pitch")
    parser.add_argument("--cam_adc_bits", default=20, type=int, help="Bit resolution of camera's ADC")    
    parser.add_argument("--pix_block_size", default=1, type=int, help="Camera ROI size, in pixels KxK")
    parser.add_argument("--cam_pitch", default=5.0e-6, type=float, help="Camera pixel pitch")    
    parser.add_argument("--cluster_size", default=16, type=int, help="Cluster size used on DMD for image encoding")
    parser.add_argument('--scale', default=1.0, type=float, help="Pre-determined scale")
    parser.add_argument('--bias', default=0.0, type=float, help="Pre-determined bias")
    parser.add_argument('--frame_of_zeros', default=False, type=bool, help="For testing, omit the frame")
    parser.add_argument('--runs', default=100, type=int, help="Number of runs performed")
    parser.add_argument('--find_linear_fit', default=False, type=bool, help="Do many runs and find linear parameters to fit data")
    parser.add_argument('--seed', default=None, type=int, help="Random seed to be used")
    parser.add_argument('--E0', default=None, type=float, help="Intensity of reference beam. If not specified, calibration is performed to calculate E0.")
    parser.add_argument('--pupil', default=None, type=float, help="Pupil radius to be included in the 4f-system fourier plane")
    parser.add_argument('--waist', default=None, type=float, help="Beam waist after beam expansion")
    parser.add_argument('--xy_misalign', default=None, type=float, help="XY deviation of SLM with respect to beam central position. (x0,y0) tuple expected in meters")
    parser.add_argument('--zoom_misalign', default=None, type=float, help="Longitudinal deviation in the 4f relay. (d1,d2,d3,d4) tuple expected in meters")   
    parser.add_argument('--crosstalk', default=0.0, type=float, help="SLM crosstalk sigma (as a factor of pixel pitch)" )
    parser.add_argument('--shot_noise', default=False, type=bool, help="Enable shot noise")
    parser.add_argument('--fwc', default=10000, type=int, help="Full Well Capacity of camera")
    p = parser.parse_args()

    kwargs = vars(p)      
    opti_product(kwargs)
            
    
