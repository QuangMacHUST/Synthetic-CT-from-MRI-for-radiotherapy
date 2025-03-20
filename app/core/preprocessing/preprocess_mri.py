#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI preprocessing module.
This module provides functions to preprocess MRI images before conversion to CT.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Optional, Tuple

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import get_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

def apply_bias_field_correction(image: sitk.Image, shrink_factor: int = 4, 
                             iterations: list = [50, 50, 30, 20], 
                             convergence_threshold: float = 0.001) -> sitk.Image:
    """
    Apply N4 bias field correction to an MRI image.
    
    Args:
        image: Input MRI image
        shrink_factor: Shrink factor for bias field correction
        iterations: Number of iterations at each level
        convergence_threshold: Convergence threshold for the algorithm
        
    Returns:
        Bias-corrected MRI image
    """
    logger.info("Applying N4 bias field correction")
    
    try:
        # Create mask from input image
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        
        # Shrink image and mask for faster processing
        if shrink_factor > 1:
            image_shrinked = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
            mask_shrinked = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())
        else:
            image_shrinked = image
            mask_shrinked = mask
        
        # Apply N4 bias field correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)
        
        # Use try-except in case N4 fails
        try:
            output = corrector.Execute(image_shrinked, mask_shrinked)
            
            # If we shrank the image, we need to apply the bias field to the original image
            if shrink_factor > 1:
                # Get the bias field and resample it to the original image size
                bias_field = corrector.GetLogBiasFieldAsImage(image)
                
                # Apply bias field correction to original image
                output = sitk.Exp(sitk.SubtractImageFilter()(sitk.Log(sitk.Cast(image, sitk.sitkFloat32)), bias_field))
        except RuntimeError as e:
            logger.warning(f"N4 bias field correction failed: {str(e)}. Using original image.")
            output = image
        
        return output
    
    except Exception as e:
        logger.error(f"Error in bias field correction: {str(e)}")
        return image

def apply_denoising(image: sitk.Image, method: str = 'gaussian', params: Dict = None) -> sitk.Image:
    """
    Apply denoising to an MRI image.
    
    Args:
        image: Input MRI image
        method: Denoising method ('gaussian', 'bilateral', 'nlm')
        params: Parameters for the denoising method
        
    Returns:
        Denoised MRI image
    """
    logger.info(f"Applying {method} denoising")
    
    if params is None:
        params = {}
    
    try:
        if method == 'gaussian':
            # Get sigma parameter (default: 0.5)
            sigma = params.get('sigma', 0.5)
            
            # Apply Gaussian smoothing
            return sitk.DiscreteGaussian(image, sigma)
            
        elif method == 'bilateral':
            # Get parameters
            domain_sigma = params.get('domain_sigma', 0.5)
            range_sigma = params.get('range_sigma', 50.0)
            
            # Apply bilateral filter
            return sitk.Bilateral(image, domain_sigma, range_sigma)
            
        elif method == 'nlm':
            # Get parameters
            patch_radius = params.get('patch_radius', 1)
            search_radius = params.get('search_radius', 3)
            h = params.get('h', 0.05)
            
            # Apply non-local means filter
            return sitk.NonLocalMeans(image, h, patch_radius, search_radius)
            
        else:
            logger.warning(f"Unknown denoising method: {method}. Using original image.")
            return image
            
    except Exception as e:
        logger.error(f"Error in denoising: {str(e)}")
        return image

def apply_intensity_normalization(image: sitk.Image, method: str = 'minmax', params: Dict = None) -> sitk.Image:
    """
    Apply intensity normalization to an MRI image.
    
    Args:
        image: Input MRI image
        method: Normalization method ('minmax', 'z-score', 'histogram')
        params: Parameters for the normalization method
        
    Returns:
        Normalized MRI image
    """
    logger.info(f"Applying {method} intensity normalization")
    
    if params is None:
        params = {}
    
    try:
        # Convert to numpy array
        array = sitk.GetArrayFromImage(image)
        
        # Create a mask for non-zero voxels
        mask = array > 0
        
        if method == 'minmax':
            # Get parameters
            min_val = params.get('min', 0.0)
            max_val = params.get('max', 1.0)
            
            # Compute min and max values from non-zero voxels
            data_min = np.min(array[mask]) if np.any(mask) else 0
            data_max = np.max(array[mask]) if np.any(mask) else 1
            
            # Apply min-max normalization
            if data_max > data_min:
                normalized = (array - data_min) / (data_max - data_min)
                normalized = normalized * (max_val - min_val) + min_val
            else:
                # If all values are the same, set all non-zero voxels to max_val
                normalized = np.zeros_like(array)
                normalized[mask] = max_val
                
            # Keep background as zero
            normalized[~mask] = 0
                
        elif method == 'z-score':
            # Get parameters
            mask_background = params.get('mask_background', True)
            
            # Compute mean and standard deviation from non-zero voxels
            mean = np.mean(array[mask]) if np.any(mask) else 0
            std = np.std(array[mask]) if np.any(mask) else 1
            
            # Apply z-score normalization
            if std > 0:
                normalized = (array - mean) / std
            else:
                normalized = np.zeros_like(array)
            
            # Keep background as zero if requested
            if mask_background:
                normalized[~mask] = 0
                
        elif method == 'histogram':
            # Get parameters
            num_landmarks = params.get('num_landmarks', 10)
            match_reference = params.get('match_reference', True)
            
            # Apply histogram matching
            if match_reference and 'reference' in params:
                # Match to reference histogram
                reference = params['reference']
                matcher = sitk.HistogramMatchingImageFilter()
                matcher.SetNumberOfHistogramLevels(1024)
                matcher.SetNumberOfMatchPoints(num_landmarks)
                matcher.ThresholdAtMeanIntensityOn()
                
                # Return as SimpleITK image
                return matcher.Execute(image, reference)
            else:
                # Without reference, just use intensity windowing
                normalized = array.copy()
                p2, p98 = np.percentile(array[mask], (2, 98)) if np.any(mask) else (0, 1)
                normalized = np.clip(normalized, p2, p98)
                normalized = (normalized - p2) / (p98 - p2)
                normalized[~mask] = 0
                
        else:
            logger.warning(f"Unknown normalization method: {method}. Using original image.")
            return image
        
        # Convert back to SimpleITK image
        output = sitk.GetImageFromArray(normalized)
        output.CopyInformation(image)
        
        return output
            
    except Exception as e:
        logger.error(f"Error in intensity normalization: {str(e)}")
        return image

def apply_cropping(image: sitk.Image, method: str = 'boundingbox', params: Dict = None) -> sitk.Image:
    """
    Apply cropping to an MRI image.
    
    Args:
        image: Input MRI image
        method: Cropping method ('boundingbox', 'foreground')
        params: Parameters for the cropping method
        
    Returns:
        Cropped MRI image
    """
    logger.info(f"Applying {method} cropping")
    
    if params is None:
        params = {}
    
    try:
        if method == 'boundingbox':
            # Get margin parameter
            margin = params.get('margin', [10, 10, 10])
            
            # Create binary mask for the image
            mask = sitk.OtsuThreshold(image, 0, 1)
            
            # Label connected components
            components = sitk.ConnectedComponent(mask)
            
            # Get largest component
            labelStats = sitk.LabelShapeStatisticsImageFilter()
            labelStats.Execute(components)
            largest_label = max(labelStats.GetLabels(), key=lambda x: labelStats.GetNumberOfPixels(x))
            
            # Get bounding box of largest component
            bbox = labelStats.GetBoundingBox(largest_label)
            
            # Extract region with margin
            index = list(bbox[0:3])
            size = list(bbox[3:6])
            
            # Apply margin
            for i in range(3):
                index[i] = max(0, index[i] - margin[i])
                # Ensure we don't go beyond image dimensions
                size[i] = min(image.GetSize()[i] - index[i], size[i] + 2 * margin[i])
            
            # Extract region
            return sitk.RegionOfInterest(image, size, index)
            
        elif method == 'foreground':
            # Get threshold parameter
            threshold = params.get('threshold', 0.01)
            
            # Create binary mask for the image
            array = sitk.GetArrayFromImage(image)
            mask = array > (np.max(array) * threshold)
            
            # Find bounding box of the foreground
            if np.any(mask):
                indices = np.where(mask)
                min_corner = [max(0, np.min(indices[i])) for i in range(3)]
                max_corner = [np.max(indices[i]) + 1 for i in range(3)]
                
                # Convert to SimpleITK index and size
                index = [min_corner[2], min_corner[1], min_corner[0]]
                size = [max_corner[2] - min_corner[2], 
                       max_corner[1] - min_corner[1], 
                       max_corner[0] - min_corner[0]]
                
                # Extract region
                return sitk.RegionOfInterest(image, size, index)
            else:
                logger.warning("No foreground detected. Using original image.")
                return image
                
        else:
            logger.warning(f"Unknown cropping method: {method}. Using original image.")
            return image
            
    except Exception as e:
        logger.error(f"Error in cropping: {str(e)}")
        return image

def apply_resampling(image: sitk.Image, output_spacing: list = [1.0, 1.0, 1.0], 
                    output_size: list = None, interpolator: str = 'linear') -> sitk.Image:
    """
    Apply resampling to an MRI image.
    
    Args:
        image: Input MRI image
        output_spacing: Output voxel spacing
        output_size: Output image size (if None, calculated from spacing)
        interpolator: Interpolation method ('linear', 'bspline', 'nearest')
        
    Returns:
        Resampled MRI image
    """
    logger.info("Applying resampling")
    
    try:
        # Get input spacing and size
        input_spacing = image.GetSpacing()
        input_size = image.GetSize()
        
        # Calculate output size if not provided
        if output_size is None:
            output_size = [
                int(round(input_size[0] * input_spacing[0] / output_spacing[0])),
                int(round(input_size[1] * input_spacing[1] / output_spacing[1])),
                int(round(input_size[2] * input_spacing[2] / output_spacing[2]))
            ]
        
        # Get interpolator
        if interpolator == 'linear':
            sitk_interpolator = sitk.sitkLinear
        elif interpolator == 'bspline':
            sitk_interpolator = sitk.sitkBSpline
        elif interpolator == 'nearest':
            sitk_interpolator = sitk.sitkNearestNeighbor
        else:
            logger.warning(f"Unknown interpolator: {interpolator}. Using linear interpolation.")
            sitk_interpolator = sitk.sitkLinear
        
        # Create resampled image
        resampled = sitk.Resample(
            image, 
            output_size, 
            sitk.Transform(), 
            sitk_interpolator,
            image.GetOrigin(), 
            output_spacing, 
            image.GetDirection(), 
            0.0,
            image.GetPixelID()
        )
        
        return resampled
        
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        return image

def preprocess_mri(mri_image, bias_correction=True, denoise=True, normalize=True, 
                 crop=True, resample=True, region='head', params=None):
    """
    Preprocess MRI image for synthetic CT generation.
    
    Args:
        mri_image: Input MRI image (SimpleITK image or SyntheticCT)
        bias_correction: Whether to apply bias field correction
        denoise: Whether to apply denoising
        normalize: Whether to apply intensity normalization
        crop: Whether to apply cropping
        resample: Whether to apply resampling
        region: Anatomical region ('head', 'pelvis', 'thorax')
        params: Additional parameters for preprocessing
        
    Returns:
        Preprocessed MRI image
    """
    logger.info(f"Starting MRI preprocessing for {region} region")
    
    # Get default parameters from config
    preprocessing_config = config.get_preprocessing_params(region)
    
    # Override with provided parameters
    if params is None:
        params = {}
    
    # Combine default and provided parameters
    for key, value in params.items():
        if key in preprocessing_config:
            if isinstance(preprocessing_config[key], dict) and isinstance(value, dict):
                preprocessing_config[key].update(value)
            else:
                preprocessing_config[key] = value
    
    # Convert input to SimpleITK image if needed
    if isinstance(mri_image, SyntheticCT):
        input_image = mri_image.image
        metadata = mri_image.metadata.copy()
    else:
        input_image = mri_image
        metadata = {}
    
    # Create preprocessing metadata
    preprocessing_metadata = {
        'bias_correction': bias_correction,
        'denoise': denoise,
        'normalize': normalize,
        'crop': crop,
        'resample': resample,
        'region': region
    }
    
    # Apply preprocessing steps
    processed_image = input_image
    
    # 1. Bias field correction
    if bias_correction and preprocessing_config.get('bias_field_correction', {}).get('enable', True):
        bias_params = preprocessing_config.get('bias_field_correction', {})
        processed_image = apply_bias_field_correction(
            processed_image,
            shrink_factor=bias_params.get('shrink_factor', 4),
            iterations=bias_params.get('number_of_iterations', [50, 50, 30, 20]),
            convergence_threshold=bias_params.get('convergence_threshold', 0.001)
        )
    
    # 2. Denoising
    if denoise and preprocessing_config.get('denoising', {}).get('enable', True):
        denoise_params = preprocessing_config.get('denoising', {})
        processed_image = apply_denoising(
            processed_image,
            method=denoise_params.get('method', 'gaussian'),
            params=denoise_params.get('params', {}).get(denoise_params.get('method', 'gaussian'), {})
        )
    
    # 3. Cropping
    if crop and preprocessing_config.get('cropping', {}).get('enable', True):
        crop_params = preprocessing_config.get('cropping', {})
        processed_image = apply_cropping(
            processed_image,
            method=crop_params.get('method', 'boundingbox'),
            params=crop_params.get('params', {}).get(crop_params.get('method', 'boundingbox'), {})
        )
    
    # 4. Resampling
    if resample and preprocessing_config.get('resampling', {}).get('enable', True):
        resample_params = preprocessing_config.get('resampling', {})
        processed_image = apply_resampling(
            processed_image,
            output_spacing=resample_params.get('output_spacing', [1.0, 1.0, 1.0]),
            output_size=resample_params.get('output_size', None),
            interpolator=resample_params.get('interpolator', 'linear')
        )
    
    # 5. Intensity normalization (after resampling to avoid artifacts)
    if normalize and preprocessing_config.get('normalization', {}).get('enable', True):
        norm_params = preprocessing_config.get('normalization', {})
        processed_image = apply_intensity_normalization(
            processed_image,
            method=norm_params.get('method', 'minmax'),
            params=norm_params.get('params', {}).get(norm_params.get('method', 'minmax'), {})
        )
    
    # Update metadata
    if 'preprocessing' not in metadata:
        metadata['preprocessing'] = preprocessing_metadata
    else:
        metadata['preprocessing'].update(preprocessing_metadata)
    
    # Create SyntheticCT object with preprocessed image and metadata
    result = SyntheticCT(processed_image, metadata)
    
    logger.info("MRI preprocessing completed")
    return result 