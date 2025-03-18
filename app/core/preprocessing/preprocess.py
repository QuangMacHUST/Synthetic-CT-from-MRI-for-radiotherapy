#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Preprocessing module for MRI to CT conversion.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Optional, List, Union, Tuple

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import load_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


def apply_bias_field_correction(image: sitk.Image, 
                               params: Optional[Dict[str, Any]] = None) -> sitk.Image:
    """
    Apply N4 bias field correction to MRI image.
    
    Args:
        image: Input SimpleITK image
        params: Dictionary of parameters for bias field correction
        
    Returns:
        Corrected SimpleITK image
    """
    logger.info("Applying bias field correction")
    
    if params is None:
        # Get default parameters from config
        bias_config = config.get_preprocessing_params().get('bias_field_correction', {})
        params = {
            'shrink_factor': bias_config.get('shrink_factor', 4),
            'iterations': bias_config.get('number_of_iterations', [50, 50, 30, 20]),
            'convergence_threshold': bias_config.get('convergence_threshold', 0.001)
        }
    
    try:
        # Create mask by thresholding
        mask = sitk.OtsuThreshold(image, 0, 1)
        
        # Apply N4 bias field correction
        shrink_factor = params.get('shrink_factor', 4)
        iterations = params.get('iterations', [50, 50, 30, 20])
        convergence_threshold = params.get('convergence_threshold', 0.001)
        
        # Shrink the image to speed up processing
        input_image = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
        input_mask = sitk.Shrink(mask, [shrink_factor] * mask.GetDimension())
        
        # Set up N4 corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)
        
        # Apply correction
        corrected_image = corrector.Execute(input_image, input_mask)
        
        # Get log bias field
        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        
        # Correct original input image using the bias field
        corrected_original = sitk.Exp(sitk.Subtract(sitk.Log(sitk.Cast(image, sitk.sitkFloat32)), log_bias_field))
        
        logger.info("Bias field correction completed")
        return corrected_original
    
    except Exception as e:
        logger.error(f"Error in bias field correction: {str(e)}")
        logger.warning("Returning original image without bias field correction")
        return image


def apply_denoising(image: sitk.Image, 
                  params: Optional[Dict[str, Any]] = None) -> sitk.Image:
    """
    Apply denoising to MRI image.
    
    Args:
        image: Input SimpleITK image
        params: Dictionary of parameters for denoising
        
    Returns:
        Denoised SimpleITK image
    """
    logger.info("Applying denoising")
    
    if params is None:
        # Get default parameters from config
        denoising_config = config.get_preprocessing_params().get('denoising', {})
        method = denoising_config.get('method', 'gaussian')
        params = {
            'method': method,
            'params': denoising_config.get('params', {}).get(method, {})
        }
    
    method = params.get('method', 'gaussian')
    method_params = params.get('params', {})
    
    try:
        if method == 'gaussian':
            sigma = method_params.get('sigma', 0.5)
            denoised_image = sitk.DiscreteGaussian(image, sigma)
        
        elif method == 'bilateral':
            domain_sigma = method_params.get('domain_sigma', 3.0)
            range_sigma = method_params.get('range_sigma', 50.0)
            denoised_image = sitk.Bilateral(image, domain_sigma, range_sigma)
        
        elif method == 'nlm':  # Non-local means
            patch_radius = method_params.get('patch_radius', 1)
            search_radius = method_params.get('search_radius', 3)
            h = method_params.get('h', 0.05)
            
            # Convert parameters for SimpleITK
            patch_size = [patch_radius * 2 + 1] * image.GetDimension()
            search_size = [search_radius * 2 + 1] * image.GetDimension()
            
            denoised_image = sitk.PatchBasedDenoisingImageFilter.New()
            denoised_image.SetInput(image)
            denoised_image.SetPatchRadius(patch_size)
            denoised_image.SetNoiseModel(sitk.PatchBasedDenoisingImageFilter.GAUSSIAN)
            denoised_image.SetNoiseModelFidelityWeight(h)
            denoised_image = denoised_image.Execute()
        
        else:
            logger.warning(f"Unknown denoising method: {method}. Using Gaussian filter.")
            denoised_image = sitk.DiscreteGaussian(image, 0.5)
        
        logger.info(f"Denoising completed using {method} method")
        return denoised_image
    
    except Exception as e:
        logger.error(f"Error in denoising: {str(e)}")
        logger.warning("Returning original image without denoising")
        return image


def apply_normalization(image: sitk.Image, 
                       params: Optional[Dict[str, Any]] = None) -> sitk.Image:
    """
    Apply intensity normalization to MRI image.
    
    Args:
        image: Input SimpleITK image
        params: Dictionary of parameters for normalization
        
    Returns:
        Normalized SimpleITK image
    """
    logger.info("Applying intensity normalization")
    
    if params is None:
        # Get default parameters from config
        normalization_config = config.get_preprocessing_params().get('normalization', {})
        method = normalization_config.get('method', 'minmax')
        params = {
            'method': method,
            'params': normalization_config.get('params', {}).get(method, {})
        }
    
    method = params.get('method', 'minmax')
    method_params = params.get('params', {})
    
    try:
        # Get image as array for manipulation
        image_array = sitk.GetArrayFromImage(image)
        
        if method == 'minmax':
            # Min-max normalization
            min_val = method_params.get('min', 0.0)
            max_val = method_params.get('max', 1.0)
            
            # Create mask for non-zero voxels
            mask = image_array > 0
            
            if np.any(mask):
                current_min = np.min(image_array[mask])
                current_max = np.max(image_array[mask])
                
                # Avoid division by zero
                if current_max > current_min:
                    normalized_array = min_val + (image_array - current_min) * (max_val - min_val) / (current_max - current_min)
                    normalized_array[~mask] = 0  # Keep background as 0
                else:
                    normalized_array = np.zeros_like(image_array)
                    normalized_array[mask] = min_val
            else:
                normalized_array = np.zeros_like(image_array)
        
        elif method == 'z-score':
            # Z-score normalization
            mask_background = method_params.get('mask_background', True)
            
            if mask_background:
                # Create mask for non-zero voxels
                mask = image_array > 0
            else:
                mask = np.ones_like(image_array, dtype=bool)
            
            if np.any(mask):
                mean_val = np.mean(image_array[mask])
                std_val = np.std(image_array[mask])
                
                # Avoid division by zero
                if std_val > 0:
                    normalized_array = (image_array - mean_val) / std_val
                    if mask_background:
                        normalized_array[~mask] = 0  # Keep background as 0
                else:
                    normalized_array = np.zeros_like(image_array)
                    if mask_background:
                        normalized_array[mask] = 0  # All voxels set to 0 if std is 0
            else:
                normalized_array = np.zeros_like(image_array)
        
        elif method == 'histogram':
            # Histogram matching/equalization
            num_landmarks = method_params.get('num_landmarks', 100)
            match_reference = method_params.get('match_reference', False)
            
            # Create histogram equalizer
            if match_reference:
                # TODO: Implement reference histogram matching
                # For now, perform histogram equalization
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(num_landmarks)
                matcher.SetNumberOfMatchPoints(num_landmarks - 1)
                matcher.ThresholdAtMeanIntensityOn()
                
                normalized_image = matcher.Execute(image, image)  # Match to itself as placeholder
                return normalized_image
            else:
                # Perform histogram equalization
                normalizer = sitk.AdaptiveHistogramEqualizationImageFilter()
                normalizer.SetAlpha(0.3)
                normalizer.SetBeta(0.3)
                
                normalized_image = normalizer.Execute(image)
                return normalized_image
        
        else:
            logger.warning(f"Unknown normalization method: {method}. Using min-max normalization.")
            
            # Default to min-max normalization
            mask = image_array > 0
            
            if np.any(mask):
                current_min = np.min(image_array[mask])
                current_max = np.max(image_array[mask])
                
                # Avoid division by zero
                if current_max > current_min:
                    normalized_array = (image_array - current_min) / (current_max - current_min)
                    normalized_array[~mask] = 0  # Keep background as 0
                else:
                    normalized_array = np.zeros_like(image_array)
                    normalized_array[mask] = 0.5
            else:
                normalized_array = np.zeros_like(image_array)
        
        # Convert back to SimpleITK image
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)
        
        logger.info(f"Intensity normalization completed using {method} method")
        return normalized_image
    
    except Exception as e:
        logger.error(f"Error in intensity normalization: {str(e)}")
        logger.warning("Returning original image without normalization")
        return image


def apply_cropping(image: sitk.Image, 
                 params: Optional[Dict[str, Any]] = None) -> sitk.Image:
    """
    Apply cropping to MRI image.
    
    Args:
        image: Input SimpleITK image
        params: Dictionary of parameters for cropping
        
    Returns:
        Cropped SimpleITK image
    """
    logger.info("Applying cropping")
    
    if params is None:
        # Get default parameters from config
        cropping_config = config.get_preprocessing_params().get('cropping', {})
        method = cropping_config.get('method', 'boundingbox')
        params = {
            'method': method,
            'params': cropping_config.get('params', {}).get(method, {})
        }
    
    method = params.get('method', 'boundingbox')
    method_params = params.get('params', {})
    
    try:
        if method == 'boundingbox':
            # Get margin around bounding box
            margin = method_params.get('margin', [10, 10, 10])
            
            # Create mask by thresholding
            mask = sitk.OtsuThreshold(image, 0, 1)
            
            # Get bounding box of non-zero region
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(mask)
            
            # Get bounding box
            bounding_box = label_shape_filter.GetBoundingBox(1)
            
            # Parse bounding box (x, y, z, width, height, depth)
            x, y, z, width, height, depth = bounding_box
            
            # Apply margin (handle boundaries)
            size = image.GetSize()
            
            x = max(0, x - margin[0])
            y = max(0, y - margin[1])
            z = max(0, z - margin[2])
            
            width = min(size[0] - x, width + 2 * margin[0])
            height = min(size[1] - y, height + 2 * margin[1])
            depth = min(size[2] - z, depth + 2 * margin[2])
            
            # Create cropped image
            cropped_image = sitk.RegionOfInterest(image, [width, height, depth], [x, y, z])
        
        elif method == 'foreground':
            # Get threshold for foreground detection
            threshold = method_params.get('threshold', 0.01)
            
            # Get image array
            image_array = sitk.GetArrayFromImage(image)
            
            # Get min and max values
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            
            # Calculate threshold value
            threshold_value = min_val + threshold * (max_val - min_val)
            
            # Create mask
            mask = sitk.GetImageFromArray(image_array > threshold_value)
            mask.CopyInformation(image)
            
            # Get bounding box of non-zero region
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(mask)
            
            if label_shape_filter.GetNumberOfLabels() > 0:
                # Get bounding box
                bounding_box = label_shape_filter.GetBoundingBox(1)
                
                # Parse bounding box (x, y, z, width, height, depth)
                x, y, z, width, height, depth = bounding_box
                
                # Create cropped image
                cropped_image = sitk.RegionOfInterest(image, [width, height, depth], [x, y, z])
            else:
                logger.warning("No foreground detected for cropping. Returning original image.")
                return image
        
        else:
            logger.warning(f"Unknown cropping method: {method}. Returning original image.")
            return image
        
        logger.info(f"Cropping completed using {method} method")
        return cropped_image
    
    except Exception as e:
        logger.error(f"Error in cropping: {str(e)}")
        logger.warning("Returning original image without cropping")
        return image


def apply_resampling(image: sitk.Image, 
                    params: Optional[Dict[str, Any]] = None) -> sitk.Image:
    """
    Apply resampling to MRI image.
    
    Args:
        image: Input SimpleITK image
        params: Dictionary of parameters for resampling
        
    Returns:
        Resampled SimpleITK image
    """
    logger.info("Applying resampling")
    
    if params is None:
        # Get default parameters from config
        resampling_config = config.get_preprocessing_params().get('resampling', {})
        params = {
            'output_spacing': resampling_config.get('output_spacing', [1.0, 1.0, 1.0]),
            'output_size': resampling_config.get('output_size', None),
            'interpolator': resampling_config.get('interpolator', 'linear')
        }
    
    output_spacing = params.get('output_spacing', [1.0, 1.0, 1.0])
    output_size = params.get('output_size', None)
    interpolator_name = params.get('interpolator', 'linear')
    
    # Set interpolator
    if interpolator_name == 'linear':
        interpolator = sitk.sitkLinear
    elif interpolator_name == 'bspline':
        interpolator = sitk.sitkBSpline
    elif interpolator_name == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    else:
        logger.warning(f"Unknown interpolator: {interpolator_name}. Using linear interpolation.")
        interpolator = sitk.sitkLinear
    
    try:
        # Get original parameters
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        original_direction = image.GetDirection()
        original_origin = image.GetOrigin()
        
        # Calculate output size if not provided
        if output_size is None:
            output_size = [
                int(round(original_size[0] * original_spacing[0] / output_spacing[0])),
                int(round(original_size[1] * original_spacing[1] / output_spacing[1])),
                int(round(original_size[2] * original_spacing[2] / output_spacing[2]))
            ]
        
        # Create resampler
        resampled_image = sitk.Resample(
            image,
            output_size,
            sitk.Transform(),
            interpolator,
            original_origin,
            output_spacing,
            original_direction,
            0.0,  # Default pixel value
            image.GetPixelID()
        )
        
        logger.info(f"Resampling completed to spacing: {output_spacing}")
        return resampled_image
    
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        logger.warning("Returning original image without resampling")
        return image


def preprocess_mri(mri_image: Union[sitk.Image, SyntheticCT, str], 
                  preprocessing_steps: Optional[List[str]] = None,
                  params: Optional[Dict[str, Any]] = None) -> SyntheticCT:
    """
    Preprocess MRI image with specified steps.
    
    Args:
        mri_image: Input MRI image as SimpleITK image, SyntheticCT object, or path to file
        preprocessing_steps: List of preprocessing steps to apply
        params: Dictionary of parameters for preprocessing steps
        
    Returns:
        SyntheticCT object containing preprocessed MRI
    """
    logger.info("Starting MRI preprocessing")
    
    # Convert input to SyntheticCT if needed
    if isinstance(mri_image, str):
        # Load image from file
        from app.utils.io_utils import load_medical_image
        mri_image = load_medical_image(mri_image)
    
    if isinstance(mri_image, SyntheticCT):
        input_image = mri_image.image
        metadata = mri_image.metadata.copy()
    else:
        input_image = mri_image
        metadata = {}
    
    # Get preprocessing configuration
    preprocessing_config = config.get_preprocessing_params()
    
    # Determine preprocessing steps to apply
    if preprocessing_steps is None:
        preprocessing_steps = []
        
        # Add steps that are enabled in config
        if preprocessing_config.get('bias_field_correction', {}).get('enable', False):
            preprocessing_steps.append('bias_field_correction')
        
        if preprocessing_config.get('denoising', {}).get('enable', False):
            preprocessing_steps.append('denoising')
        
        if preprocessing_config.get('normalization', {}).get('enable', False):
            preprocessing_steps.append('normalization')
        
        if preprocessing_config.get('cropping', {}).get('enable', False):
            preprocessing_steps.append('cropping')
        
        if preprocessing_config.get('resampling', {}).get('enable', False):
            preprocessing_steps.append('resampling')
    
    # Initialize parameters if not provided
    if params is None:
        params = {}
    
    # Initialize preprocessing metadata
    preprocessing_metadata = {
        'steps_applied': []
    }
    
    # Apply each preprocessing step
    processed_image = input_image
    
    for step in preprocessing_steps:
        step_params = params.get(step, None)
        
        if step == 'bias_field_correction':
            processed_image = apply_bias_field_correction(processed_image, step_params)
            preprocessing_metadata['steps_applied'].append('bias_field_correction')
        
        elif step == 'denoising':
            processed_image = apply_denoising(processed_image, step_params)
            preprocessing_metadata['steps_applied'].append('denoising')
        
        elif step == 'normalization':
            processed_image = apply_normalization(processed_image, step_params)
            preprocessing_metadata['steps_applied'].append('normalization')
        
        elif step == 'cropping':
            processed_image = apply_cropping(processed_image, step_params)
            preprocessing_metadata['steps_applied'].append('cropping')
        
        elif step == 'resampling':
            processed_image = apply_resampling(processed_image, step_params)
            preprocessing_metadata['steps_applied'].append('resampling')
        
        else:
            logger.warning(f"Unknown preprocessing step: {step}. Skipping.")
    
    # Update metadata
    metadata['preprocessing'] = preprocessing_metadata
    
    # Create SyntheticCT object
    result = SyntheticCT(processed_image, metadata)
    
    logger.info(f"MRI preprocessing completed. Applied steps: {preprocessing_metadata['steps_applied']}")
    
    return result 