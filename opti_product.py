import argparse, math, torch, time, random, os, gc
from pathlib import Path
from dict import DEVICE, generate_dither_alphabet
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
    runs              = kwargs.get("runs")
    find_linear_fit   = kwargs.get("find_linear_fit")
    seed              = kwargs.get("seed")
    pupil             = kwargs.get("pupil")
    waist             = kwargs.get("waist")
    crosstalk         = kwargs.get("crosstalk")
    xy_misalign       = kwargs.get("xy_misalign")
    zoom_misalign     = kwargs.get("zoom_misalign")
    shot_noise        = kwargs.get("shot_noise")
    fwc               = kwargs.get("fwc")

    scale = torch.tensor(scale)
    bias = torch.tensor(bias)
    DMD.alphabet = generate_dither_alphabet(cluster_size)
    tau = None
    if shot_noise is None:
        shot_noise = False
    if xy_misalign is None:
        xy_misalign = [0,0]
    if zoom_misalign is None:
        zoom_misalign = [0,0,0,0]
    x0_dmd = xy_misalign[0]*dmd_pixels*dmd_pitch
    y0_dmd = xy_misalign[1]*dmd_pixels*dmd_pitch


    print(f"Simulation running on {DEVICE}")
    
    if seed is not None:
        set_seed(seed)
            
    dmd_device = DMD(pitch=dmd_pitch,
                     num_pixels=dmd_pixels,
                     x0=x0_dmd,
                     y0=y0_dmd)
    
    slm_device = SLM(pitch=slm_pitch,
                    Nx=slm_pixels,
                    Ny=slm_pixels,
                    fill_factor=0.956,
                    wlen=wlen,
                    focal_length=focal_length,
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
        """
        Operate vectors by propagating field through modulated devices and performing focal plane detection
        Args:
            quantised_reading: If enabled, returns in digital value; else, returns in number of electrons
            norm_to_E0: If enabled, performs the non-linear operation to extract n from Nr, and returns n
            misalign: (a1,a2,a3,a4) tuple referent to 4f relay misalignment
        Returns:
            During bright/dark level calibration (quantised reading = False, norm_to_E0 = False):
                reading: number of electrons
                tau_out: exposure time, adjusted to reach FWC
            During E0 calibration (quantised reading = True, norm_to_E0 = False):
                reading_quant: number of electrons, expressed as digital value, placed in dark-to-bright digital scale
            During normal operation (quantised reading = True, norm_to_E0 = True):
                opti_result: optical reading, already normalised to E0 and quantised
                digi_result: expected digital result, from input vectors
        """
        with torch.inference_mode():
            field.reset()
            propagation(field=field,
                        dmd = dmd_proc,
                        slm = slm_proc,
                        waist = waist,
                        pupil_radius = pupil,
                        misalign = misalign)
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
    
    print("==== STARTING CALIBRATION ====")
    
    print("Defining brightest pixel level")
    
# Camera calibration: defining BRIGHT level
    set_input_type("fully_lit", image_dmd, image_slm)
    image_dmd.set_all_ones()
    image_slm.set_all_ones()


    E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=DEVICE, dtype=torch.float32)
    tau_acc = 0
    for _ in range(calibration_rep):
        E_run, tau_run = propagate_and_read()
        E_capt += E_run
        tau_acc += tau_run
        

    E_bright = (E_capt / calibration_rep).max()
    tau = tau_acc / calibration_rep
    print(f"Exposure time tau = {tau.detach().cpu().numpy()} seconds")
    kwargs.update({"tau": tau})
    print(f"Upper threshrold E_bright = {E_bright.detach().cpu().numpy()}")
    
    print("Defining darkest pixel level")
# Camera calibration: defining BLACK level
    image_dmd.set_all_ones()
    image_slm.set_custom_input(value=-1.0)
    E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=DEVICE, dtype=torch.float32)
    for _ in range(calibration_rep):
        E_capt += propagate_and_read()[0]

    E_dim = (E_capt / calibration_rep).min()

    print(f"Lower threshrold E_dim = {E_dim.detach().cpu().numpy()}")

    cam_low, cam_high = define_limits_camera_ADC(dim=E_dim, bright=E_bright)
    
    print("Camera DV-to-electrons limits:")
    print(f'Cam high: {cam_high}, Cam low: {cam_low}')
    DV_offset = (2**cam_adc_bits - 1) * cam_low / (cam_high - cam_low + 1e-9)
    print('Digital Value offset: ',DV_offset.detach().cpu().numpy())
    
# System calibration: defining reference intensity (E0)
    print("Defining reference intensity E0:")
    set_input_type("frame_only", image_dmd, image_slm)
    E_capt = torch.zeros((lenslets, lenslets, pix_block_size, pix_block_size), device=DEVICE, dtype=torch.float32)
    for _ in range(calibration_rep):
        E_capt += propagate_and_read(quantised_reading=True)

    E0 = (E_capt / calibration_rep).sum(dim=(-2,-1))
    
    print(f"E0 = \n{E0.detach().cpu().numpy()}")
    kwargs.update({"E0": E0})

    print("==== CALIBRATION COMPLETE ====")

# Finding linear parameters (least squares)
    if find_linear_fit:
        correct    = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        contestant = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        for run in range(runs):
            print(f'[Run {run+1}/{runs}]') 
            set_input_type(input_type, image_dmd, image_slm, dataset=dataset)
            opti_result, digi_result = propagate_and_read(quantised_reading=True, norm_to_E0=True)
            contestant[run] = opti_result
            correct[run] = digi_result
            print(f'Optical result = \n{opti_result.detach().cpu().numpy()} \nDigital result = \n{digi_result.detach().cpu().numpy()}')
            
        scale, bias = find_scale_and_bias(correct,contestant)
        kwargs.update({"scale": scale, "bias": bias})
        print(f'Linear fitting done:\n Scale = \n{scale.detach().cpu().numpy()}\n Bias = \n{bias.detach().cpu().numpy()}')
        fitted_optical = scale * contestant + bias
        error_metrics = calculate_errors(correct, fitted_optical, lens_wise=True)
        print("\n".join(f"{k}:\n {v}" for k, v in error_metrics.items()))      
        
# If scale and biases pre-defined, then run:
    else:
        print(f"Assuming scale = {scale} and bias = {bias}")
        correct = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        fitted_optical = torch.empty((runs, lenslets, lenslets), device="cpu", dtype=torch.float32)
        for run in range(runs):
            set_input_type(input_type, image_dmd, image_slm, dataset)
            opti_result, digi_result = propagate_and_read(quantised_reading=True, norm_to_E0=True)
            opti_result = scale*opti_result + bias
            fitted_optical[run] = opti_result
            correct[run] = digi_result
            print(f'Optical result: \n{opti_result.detach().cpu().numpy()}')
            print(f'Digital result: \n{digi_result.detach().cpu().numpy()}')        
        error_metrics = calculate_errors(correct, fitted_optical, lens_wise=True)
        print("\n".join(f"{k}: {v}" for k, v in error_metrics.items()))      

    Path("./outcomes").mkdir(parents=True, exist_ok=True)

    if title is not None:
        create_outcome_dir(title)
    else:
        go_to_unlabeled_tests()
    log_to_file(kwargs, correct, fitted_optical, error_metrics)
    
    return error_metrics

        
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    parser = argparse.ArgumentParser(
        description="Perform a dot product using free-space optics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--title", type=str, default=None,
                        help="Experiment name (used for output folder naming). If omitted, results go to a default folder.")
    parser.add_argument("--runs", type=int, default=100, help="Number of runs to perform.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, uses random seed.")

    parser.add_argument(
        "--input_type",
        type=str,
        default="random_uncorr",
        choices=["random_uncorr", "figure", "random_uncorr_sorted", "figure_sorted", "custom", "fully_lit", "frame_only"],
        help="Input type / generation mode.",
    )
    parser.add_argument("--dataset", type=str, default="cifar", choices=["cifar", "celeb"],
                        help="Dataset used when input_type='figure'.")

    parser.add_argument("--lenslets", type=int, default=1,
                        help="Number of lenslets per axis (total dot products = lenslets^2).")
    parser.add_argument("--input_size", type=int, default=45,
                        help="Vector side length (N): total elements = N^2.")

    parser.add_argument("--wlen", type=float, default=650e-9, help="Wavelength [m].")
    parser.add_argument("--focal_length", type=float, default=425e-3,
                        help="Focal length of SLM lens phase profile [m].")
    parser.add_argument("--pupil", type=float, default=None,
                        help="Fourier-plane pupil radius [m]. If omitted, no pupil is applied.")
    parser.add_argument("--waist", type=float, default=None,
                        help="Beam waist after expansion [m]. If omitted, perfect plane wave is assumed")

    parser.add_argument("--dmd_pixels", type=int, default=1024, help="DMD resolution per axis (pixels).")
    parser.add_argument("--slm_pixels", type=int, default=1024, help="SLM resolution per axis (pixels).")
    parser.add_argument("--active_area_ratio", type=float, default=0.4943,
                        help="Active/total pixel area (defines frame width).")
    parser.add_argument("--dmd_pitch", type=float, default=8e-6, help="DMD pixel pitch [m].")
    parser.add_argument("--slm_pitch", type=float, default=8e-6, help="SLM pixel pitch [m].")

    parser.add_argument("--dx", type=float, default=1.0e-6, help="Simulation sampling interval dx [m].")
    parser.add_argument("--Nx", type=int, default=16384, help="Simulation grid size (Nx x Nx).")

    parser.add_argument("--cam_pitch", type=float, default=3.0e-6, help="Camera pixel pitch [m].")
    parser.add_argument("--cam_adc_bits", type=int, default=20, help="Camera ADC resolution [bits].")
    parser.add_argument("--pix_block_size", type=int, default=1, help="Camera capture block size [pixels].")
    parser.add_argument("--fwc", type=int, default=None,
                        help="Full well capacity [electrons]. If omitted, estimated from pixel area.")

    parser.add_argument("--cluster_size", type=int, default=16,
                        help="DMD cluster size used for encoding (defines bit depth).")

    parser.add_argument("--scale", type=float, default=1.0, help="Pre-determined scale (if not fitting).")
    parser.add_argument("--bias", type=float, default=0.0, help="Pre-determined bias (if not fitting).")
    parser.add_argument("--find_linear_fit", action="store_true",
                        help="Estimate linear scale/bias mapping optical values to digital values after running.")

    parser.add_argument("--xy_misalign", type=float, nargs=2, default=None, metavar=("X0", "Y0"),
                        help="Transverse misalignment [m]. Usage: --xy_misalign x0 y0")
    parser.add_argument("--zoom_misalign", type=float, nargs=4, default=None, metavar=("D1", "D2", "D3", "D4"),
                        help="4f relay distance deviations [m]. Usage: --zoom_misalign d1 d2 d3 d4")

    parser.add_argument("--crosstalk", type=float, default=0.0,
                        help="SLM crosstalk sigma (in units of SLM pixel pitch).")
    parser.add_argument("--shot_noise", action="store_true", help="Enable shot-noise sampling.")

    p = parser.parse_args()
    kwargs = vars(p)
    opti_product(kwargs)


    
