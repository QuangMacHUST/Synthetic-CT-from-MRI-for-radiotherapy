#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
MRI preprocessing module
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from app.utils.io_utils import load_medical_image, SyntheticCT


def normalize_intensity(image, min_percentile=1, max_percentile=99):
    """
    Normalize image intensity to [0, 1] range based on percentiles.
    
    Args:
        image: SimpleITK image
        min_percentile: Minimum percentile for normalization
        max_percentile: Maximum percentile for normalization
        
    Returns:
        Normalized SimpleITK image
    """
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Calculate percentiles
    min_val = np.percentile(array, min_percentile)
    max_val = np.percentile(array, max_percentile)
    
    # Clip and normalize
    array = np.clip(array, min_val, max_val)
    array = (array - min_val) / (max_val - min_val)
    
    # Convert back to SimpleITK image
    normalized_image = sitk.GetImageFromArray(array)
    normalized_image.CopyInformation(image)
    
    return normalized_image


def bias_field_correction(image, shrink_factor=4, num_iterations=200, num_fitting_levels=4):
    """
    Apply N4 bias field correction to MRI image.
    
    Args:
        image: SimpleITK image
        shrink_factor: Shrink factor for computation efficiency
        num_iterations: Number of iterations at each fitting level
        num_fitting_levels: Number of fitting levels
        
    Returns:
        Bias-corrected SimpleITK image
    """
    # Create mask if not provided (assume all non-zero voxels are foreground)
    array = sitk.GetArrayFromImage(image)
    mask = sitk.GetImageFromArray((array > 0).astype(np.uint8))
    mask.CopyInformation(image)
    
    # Shrink image and mask for efficiency
    if shrink_factor > 1:
        image_shrinker = sitk.ShrinkImageFilter()
        image_shrinker.SetShrinkFactors([shrink_factor] * image.GetDimension())
        image = image_shrinker.Execute(image)
        mask = image_shrinker.Execute(mask)
    
    # Apply N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * num_fitting_levels)
    corrected_image = corrector.Execute(image, mask)
    
    return corrected_image


def denoise_image(image, strength=0.1):
    """
    Apply denoising to MRI image.
    
    Args:
        image: SimpleITK image
        strength: Denoising strength
        
    Returns:
        Denoised SimpleITK image
    """
    # Apply anisotropic diffusion filter
    denoiser = sitk.CurvatureAnisotropicDiffusionImageFilter()
    denoiser.SetTimeStep(0.0625)
    denoiser.SetNumberOfIterations(5)
    denoiser.SetConductanceParameter(strength)
    
    denoised_image = denoiser.Execute(image)
    
    return denoised_image


def resample_image(image, new_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    """
    Resample image to new spacing.
    
    Args:
        image: SimpleITK image
        new_spacing: New spacing in mm
        interpolator: SimpleITK interpolator
        
    Returns:
        Resampled SimpleITK image
    """
    # Get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / new_spacing[1])),
        int(round(original_size[2] * original_spacing[2] / new_spacing[2]))
    ]
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    
    # Resample image
    resampled_image = resampler.Execute(image)
    
    return resampled_image


def register_images(fixed_image, moving_image, transform_type='rigid'):
    """
    Register moving image to fixed image.
    
    Args:
        fixed_image: Fixed SimpleITK image
        moving_image: Moving SimpleITK image
        transform_type: Type of transform ('rigid', 'affine', or 'bspline')
        
    Returns:
        Registered SimpleITK image
    """
    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set similarity metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Set optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Set initial transform
    if transform_type == 'rigid':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'affine':
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(3), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    elif transform_type == 'bspline':
        mesh_size = [10] * 3
        initial_transform = sitk.BSplineTransformInitializer(fixed_image, mesh_size)
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")
    
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    
    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    
    registered_image = resampler.Execute(moving_image)
    
    return registered_image


def preprocess_mri(input_path, output_path=None, apply_bias_correction=True, 
                  apply_denoising=True, normalize=True, resample=True, 
                  new_spacing=(1.0, 1.0, 1.0)):
    """
    Preprocess MRI image.
    
    Args:
        input_path: Path to input MRI image
        output_path: Path to save preprocessed image (optional)
        apply_bias_correction: Whether to apply bias field correction
        apply_denoising: Whether to apply denoising
        normalize: Whether to normalize intensity
        resample: Whether to resample to isotropic spacing
        new_spacing: New spacing if resampling
        
    Returns:
        Preprocessed MRI image as SyntheticCT object
    """
    logging.info(f"Preprocessing MRI image: {input_path}")
    
    # Load image
    image = load_medical_image(input_path)
    
    # Apply preprocessing steps
    if apply_bias_correction:
        logging.info("Applying bias field correction")
        image = bias_field_correction(image)
    
    if apply_denoising:
        logging.info("Applying denoising")
        image = denoise_image(image)
    
    if normalize:
        logging.info("Normalizing intensity")
        image = normalize_intensity(image)
    
    if resample:
        logging.info(f"Resampling to spacing: {new_spacing}")
        image = resample_image(image, new_spacing)
    
    # Create metadata
    metadata = {
        'original_path': str(input_path),
        'preprocessing': {
            'bias_correction': apply_bias_correction,
            'denoising': apply_denoising,
            'normalization': normalize,
            'resampling': resample,
            'spacing': new_spacing if resample else image.GetSpacing()
        }
    }
    
    # Create SyntheticCT object
    preprocessed_mri = SyntheticCT(image, metadata)
    
    # Save if output path is provided
    if output_path:
        logging.info(f"Saving preprocessed MRI to: {output_path}")
        preprocessed_mri.save(output_path)
    
    return preprocessed_mri 