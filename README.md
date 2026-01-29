# Half-Signed Optical Dot-Product Multiplier

This repository accompanies *“Free-Space Coherent Optical Dot-Product Multiplier with Lensless Fan-In”* (Duque *et al.*, 2026).

It provides simulation code for a coherent free-space optical dot-product system using DMD/SLM-style modulation and camera-like readout.

The code computes the dot product between two vectors (one signed, one non-negative) by modulating the amplitude of a CW monochromatic laser beam using a DMD, followed by phase modulation with an SLM. The output is recovered via interferometric detection on a camera.

Field propagation is based on Rafael Fuente’s [Diffractsim](https://github.com/rafael-fuente/diffractsim/tree/main) library.


## Features
- **Configurable device parameters** (DMD/SLM resolution, pixel pitch)
- **Configurable propagation parameters** (wavelength, focal length, Fourier-plane pupil)
- **Configurable input sizes and distributions** (random uncorrelated; images from CIFAR-10; images from CelebA)
- **Configurable parallelism** (number of simultaneous dot products via lenslet grid)
- **Noise and nonideality modelling** (shot/read noise, LC-SLM crosstalk, device misplacement)
- **Per-run outputs**: saved logs including linear-fit plots and error metrics

## Installation
1. Clone this repository
```bat
git clone https://github.com/aliceduque/half_signed_opti_multiplier.git
cd half_signed_opti_multiplier
```
2. Create a conda environment using config file provided
```bat
conda env create -f environment.cpu.yml
conda activate half_signed_opti_multiplier
```
If running this code on CPU, this is the end of installation.

3. If using GPU (recommended), ensure torch packages are installed appropiately:
```bat
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121 torchvision==0.19.1+cu121
```

## Data
In addition to random vectors, the code supports vectors derived from CIFAR-10 and CelebA.
- **CIFAR-10** is downloaded automatically via ```torchvision``` the first time it is used.
- **CelebA** must be downloaded manually (e.g. from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)) and the raw .jpg images placed under ```./data/celeba/all/```.
(The dataset is not redistributed in this repository.)

## Usage
```opti_product.py``` is the main file. The simulation is configured via a set of command-line (CLI) arguments, which also makes it easy to build wrapper scripts to run parameter sweeps and batch experiments.

## Examples
### 1. Single dot product, 8-bit, 2025-sized vectors
```bat
python opti_product.py --input_type figure --dataset cifar --find_linear_fit --seed 42 --pix_block_size 3 
```
### 2. Single dot product, 4-bit, 32400-sized vectors
```bat
python opti_product.py --input_type figure --dataset celeb --find_linear_fit --seed 42 --pix_block_size 3 --input_size 180 --cluster_size 4
```
### 3. 9 parallel dot products, 8-bit, 225-sized vectors
```bat
python opti_product.py --input_type figure --dataset cifar --find_linear_fit --seed 42 --pix_block_size 3 --lenslets 3 --input_size 15 --active_area_ratio 0.4982 --dmd_pixels 1020 --slm_pixels 1020 --focal_length 160e-3
```
### 4. Single dot product, 8-bit, 2025-sized vectors, shot noise considered
```bat
python opti_product.py --input_type figure --dataset celeb --find_linear_fit --seed 42 --pix_block_size 3 --shot_noise
```

When running on CPU, runtime for default resolution is severely affected. In that case, it's advised to reduce resolution by 2, by including the arguments ```--Nx 8192 --dx 2e-6```.




