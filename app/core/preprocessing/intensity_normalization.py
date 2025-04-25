#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for MRI intensity normalization techniques.

This module provides various methods for normalizing MRI signal intensity,
which is a critical step in the preprocessing pipeline for synthetic CT generation.
Different normalization techniques are implemented to handle the inherent variability
in MRI signal intensities across different scanners, protocols, and patients.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Union, Optional, Tuple
from skimage import exposure

# Set up logger
logger = logging.getLogger(__name__)

def normalize_intensity(image: Union[sitk.Image, np.ndarray], 
                        method: str = 'minmax',
                        mask: Optional[Union[sitk.Image, np.ndarray]] = None, 
                        params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Normalize the intensity of an MRI image using the specified method.
    
    Args:
        image: Input image as SimpleITK image or numpy array
        method: Normalization method ('minmax', 'zscore', 'percentile', 'histogram_matching')
        mask: Optional mask to restrict normalization to specific regions
        params: Additional parameters for specific normalization methods
    
    Returns:
        Normalized image in the same format as input
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Convert SimpleITK to numpy if needed (keeping reference for conversion back)
    is_sitk = isinstance(image, sitk.Image)
    if is_sitk:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        img_array = sitk.GetArrayFromImage(image)
    else:
        img_array = image
    
    # Convert mask to numpy if needed
    if mask is not None and isinstance(mask, sitk.Image):
        mask_array = sitk.GetArrayFromImage(mask)
    else:
        mask_array = mask
    
    # Apply appropriate normalization method
    if method.lower() == 'minmax':
        normalized_array = minmax_normalization(img_array, mask_array, params)
    elif method.lower() == 'zscore':
        normalized_array = zscore_normalization(img_array, mask_array, params)
    elif method.lower() == 'percentile':
        normalized_array = percentile_normalization(img_array, mask_array, params)
    elif method.lower() == 'histogram_matching':
        normalized_array = histogram_matching(img_array, mask_array, params)
    else:
        logger.warning(f"Unknown normalization method '{method}'. Using minmax normalization.")
        normalized_array = minmax_normalization(img_array, mask_array, params)
    
    # Convert back to SimpleITK if the input was SimpleITK
    if is_sitk:
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.SetSpacing(spacing)
        normalized_image.SetOrigin(origin)
        normalized_image.SetDirection(direction)
        return normalized_image
    else:
        return normalized_array

def minmax_normalization(image: np.ndarray, 
                        mask: Optional[np.ndarray] = None, 
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalize image intensities to a range [min_val, max_val].
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict normalization to specific regions
        params: Additional parameters (min_val, max_val)
    
    Returns:
        Normalized image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    min_val = params.get('min_val', 0)
    max_val = params.get('max_val', 1)
    
    # Apply mask if provided
    if mask is not None:
        # Get min and max from masked region
        min_intensity = np.min(image[mask > 0])
        max_intensity = np.max(image[mask > 0])
    else:
        # Get global min and max
        min_intensity = np.min(image)
        max_intensity = np.max(image)
    
    # Avoid division by zero
    if min_intensity == max_intensity:
        logger.warning("Image has constant intensity. Returning original image.")
        return image
    
    # Apply normalization
    normalized = (image - min_intensity) / (max_intensity - min_intensity)
    
    # Scale to desired range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def zscore_normalization(image: np.ndarray, 
                        mask: Optional[np.ndarray] = None, 
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalize image intensities using Z-score normalization (zero mean, unit variance).
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict normalization to specific regions
        params: Additional parameters
    
    Returns:
        Normalized image as numpy array
    """
    # Apply mask if provided
    if mask is not None:
        # Get mean and std from masked region
        mean_intensity = np.mean(image[mask > 0])
        std_intensity = np.std(image[mask > 0])
    else:
        # Get global mean and std
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
    
    # Avoid division by zero
    if std_intensity == 0:
        logger.warning("Image has zero standard deviation. Returning original image.")
        return image
    
    # Apply normalization
    normalized = (image - mean_intensity) / std_intensity
    
    return normalized

def percentile_normalization(image: np.ndarray, 
                            mask: Optional[np.ndarray] = None, 
                            params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalize image intensities using percentile-based normalization.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict normalization to specific regions
        params: Additional parameters (lower_percentile, upper_percentile, min_val, max_val)
    
    Returns:
        Normalized image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    lower_percentile = params.get('lower_percentile', 1)
    upper_percentile = params.get('upper_percentile', 99)
    min_val = params.get('min_val', 0)
    max_val = params.get('max_val', 1)
    
    # Apply mask if provided
    if mask is not None:
        masked_image = image[mask > 0]
        lower_bound = np.percentile(masked_image, lower_percentile)
        upper_bound = np.percentile(masked_image, upper_percentile)
    else:
        lower_bound = np.percentile(image, lower_percentile)
        upper_bound = np.percentile(image, upper_percentile)
    
    # Clip image intensities to percentile range
    clipped = np.clip(image, lower_bound, upper_bound)
    
    # Apply min-max normalization to the clipped image
    if lower_bound == upper_bound:
        logger.warning("Percentile bounds are identical. Returning original image.")
        return image
    
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    
    # Scale to desired range
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def histogram_matching(image: np.ndarray, 
                      mask: Optional[np.ndarray] = None, 
                      params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Normalize image intensities using histogram matching to a reference histogram.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict normalization to specific regions
        params: Additional parameters (reference_image, n_bins)
    
    Returns:
        Normalized image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    reference_image = params.get('reference_image', None)
    n_bins = params.get('n_bins', 256)
    
    # Check if reference image exists
    if reference_image is None:
        logger.warning("Reference image not provided for histogram matching. Using standard distribution.")
        # Create standard normal distribution as reference
        reference = np.random.normal(0, 1, size=1000000)
        reference = np.clip(reference, -3, 3)  # Clip to [-3, 3]
        reference = (reference + 3) / 6  # Normalize to [0, 1]
    elif isinstance(reference_image, sitk.Image):
        reference = sitk.GetArrayFromImage(reference_image)
    else:
        reference = reference_image
    
    # Apply mask if provided
    if mask is not None:
        # Apply histogram matching to masked region only
        result = image.copy()
        
        # Match histograms for the masked region
        matched = exposure.match_histograms(
            image[mask > 0], 
            reference.ravel() if reference.ndim > 1 else reference,
            channel_axis=None
        )
        
        # Replace masked region with matched intensities
        result[mask > 0] = matched
    else:
        # Apply histogram matching to entire image
        result = exposure.match_histograms(
            image, 
            reference.ravel() if reference.ndim > 1 else reference,
            channel_axis=None
        )
    
    return result

def bias_field_correction(image: Union[sitk.Image, np.ndarray],
                          mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                          params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Apply N4 bias field correction to an MRI image.
    
    Args:
        image: Input image as SimpleITK image or numpy array
        mask: Optional mask to restrict correction to specific regions
        params: Additional parameters for N4 bias field correction
    
    Returns:
        Corrected image in the same format as input
    """
    # Default parameters
    if params is None:
        params = {}
    n_iterations = params.get('n_iterations', [50, 50, 30, 20])
    n_fitting_levels = params.get('n_fitting_levels', 4)
    shrink_factor = params.get('shrink_factor', 3)
    convergence_threshold = params.get('convergence_threshold', 0.001)
    
    # Convert numpy to SimpleITK if needed
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        orig_image = sitk.GetImageFromArray(image)
    else:
        orig_image = image
    
    # Convert mask to SimpleITK if needed
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            mask_sitk = mask
    else:
        # Create a mask from the image if none is provided
        mask_sitk = sitk.OtsuThreshold(orig_image, 0, 1, 200)
    
    # Shrink the image for faster processing
    if shrink_factor > 1:
        input_image = sitk.Shrink(orig_image, [shrink_factor] * orig_image.GetDimension())
        mask_image = sitk.Shrink(mask_sitk, [shrink_factor] * mask_sitk.GetDimension())
    else:
        input_image = orig_image
        mask_image = mask_sitk
    
    # Apply N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iterations)
    corrector.SetNumberOfFittingLevels(n_fitting_levels)
    corrector.SetConvergenceThreshold(convergence_threshold)
    
    try:
        corrected_image = corrector.Execute(input_image, mask_image)
        
        # If the image was shrunk, apply the bias field to the original resolution image
        if shrink_factor > 1:
            # Get the log bias field and resample to original resolution
            log_bias_field = corrector.GetLogBiasFieldAsImage(orig_image)
            
            # Apply bias field correction to original image
            corrected_image = orig_image / sitk.Exp(log_bias_field)
    except Exception as e:
        logger.error(f"N4 bias field correction failed: {str(e)}")
        logger.warning("Returning original image without bias field correction.")
        corrected_image = orig_image
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        corrected_image = sitk.GetArrayFromImage(corrected_image)
    
    return corrected_image

def correct_inhomogeneity(image: Union[sitk.Image, np.ndarray], 
                         mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                         method: str = 'n4',
                         params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Correct intensity inhomogeneity in an MRI image.
    
    Args:
        image: Input image as SimpleITK image or numpy array
        mask: Optional mask to restrict correction to specific regions
        method: Correction method ('n4' for N4 bias field correction)
        params: Additional parameters for the specific correction method
    
    Returns:
        Corrected image in the same format as input
    """
    if method.lower() == 'n4':
        return bias_field_correction(image, mask, params)
    else:
        logger.warning(f"Unknown inhomogeneity correction method '{method}'. Using N4 bias field correction.")
        return bias_field_correction(image, mask, params) 