#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for MRI denoising techniques.

This module provides various methods for denoising MRI images,
which is a critical step in the preprocessing pipeline for synthetic CT generation.
Different denoising techniques are implemented to handle various types of noise
commonly found in MRI data.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Union, Optional, Tuple, List
from skimage.restoration import denoise_nl_means, denoise_wavelet, denoise_tv_chambolle
from scipy.ndimage import gaussian_filter, median_filter

# Set up logger
logger = logging.getLogger(__name__)

def denoise_image(image: Union[sitk.Image, np.ndarray],
                method: str = 'gaussian',
                mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Apply denoising to an MRI image using the specified method.
    
    Args:
        image: Input image as SimpleITK image or numpy array
        method: Denoising method ('gaussian', 'median', 'nlm', 'wavelet', 'tv')
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters for specific denoising methods
    
    Returns:
        Denoised image in the same format as input
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
    
    # Apply appropriate denoising method
    if method.lower() == 'gaussian':
        denoised_array = apply_gaussian_filter(img_array, mask_array, params)
    elif method.lower() == 'median':
        denoised_array = apply_median_filter(img_array, mask_array, params)
    elif method.lower() == 'nlm':
        denoised_array = apply_nlm_filter(img_array, mask_array, params)
    elif method.lower() == 'wavelet':
        denoised_array = apply_wavelet_filter(img_array, mask_array, params)
    elif method.lower() == 'tv':
        denoised_array = apply_tv_filter(img_array, mask_array, params)
    elif method.lower() == 'anisotropic':
        denoised_array = apply_anisotropic_diffusion(img_array, mask_array, params)
    else:
        logger.warning(f"Unknown denoising method '{method}'. Using Gaussian filter.")
        denoised_array = apply_gaussian_filter(img_array, mask_array, params)
    
    # Convert back to SimpleITK if the input was SimpleITK
    if is_sitk:
        denoised_image = sitk.GetImageFromArray(denoised_array)
        denoised_image.SetSpacing(spacing)
        denoised_image.SetOrigin(origin)
        denoised_image.SetDirection(direction)
        return denoised_image
    else:
        return denoised_array

def apply_gaussian_filter(image: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply Gaussian filter to denoise an image.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (sigma)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    sigma = params.get('sigma', 1.0)
    
    # Apply Gaussian filter
    if mask is not None:
        # Apply filter only to masked region
        result = image.copy()
        masked_region = gaussian_filter(image, sigma)
        result[mask > 0] = masked_region[mask > 0]
    else:
        # Apply filter to entire image
        result = gaussian_filter(image, sigma)
    
    return result

def apply_median_filter(image: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply median filter to denoise an image.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (size)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    size = params.get('size', 3)
    
    # Apply median filter
    if mask is not None:
        # Apply filter only to masked region
        result = image.copy()
        masked_region = median_filter(image, size=size)
        result[mask > 0] = masked_region[mask > 0]
    else:
        # Apply filter to entire image
        result = median_filter(image, size=size)
    
    return result

def apply_nlm_filter(image: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply Non-local Means filter to denoise an image.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (patch_size, patch_distance, h)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    patch_size = params.get('patch_size', 5)
    patch_distance = params.get('patch_distance', 6)
    h = params.get('h', 0.1)  # Filter parameter controlling the decay of weights
    
    # Apply non-local means filter to each slice
    if image.ndim == 3:
        # Process 3D volume slice by slice
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            if mask is not None:
                # Apply filter only to masked region in this slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for NLM
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                denoised_slice = denoise_nl_means(temp, 
                                              patch_size=patch_size, 
                                              patch_distance=patch_distance, 
                                              h=h,
                                              multichannel=False)
                
                # Rescale back to original range
                denoised_slice = denoised_slice * (max_val - min_val) + min_val
                
                mask_slice = mask[i] if mask.ndim == 3 else mask
                result[i] = image[i].copy()
                result[i][mask_slice > 0] = denoised_slice[mask_slice > 0]
            else:
                # Apply filter to entire slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for NLM
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                denoised_slice = denoise_nl_means(temp, 
                                              patch_size=patch_size, 
                                              patch_distance=patch_distance, 
                                              h=h,
                                              multichannel=False)
                
                # Rescale back to original range
                result[i] = denoised_slice * (max_val - min_val) + min_val
    else:
        # Process 2D image
        if mask is not None:
            # Apply filter only to masked region
            result = image.copy()
            # Normalize image to [0,1] for NLM
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_nl_means(temp, 
                                        patch_size=patch_size, 
                                        patch_distance=patch_distance, 
                                        h=h,
                                        multichannel=False)
                
                # Rescale back to original range
                denoised = denoised * (max_val - min_val) + min_val
                
                result[mask > 0] = denoised[mask > 0]
            
        else:
            # Apply filter to entire image
            # Normalize image to [0,1] for NLM
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_nl_means(temp, 
                                        patch_size=patch_size, 
                                        patch_distance=patch_distance, 
                                        h=h,
                                        multichannel=False)
                
                # Rescale back to original range
                result = denoised * (max_val - min_val) + min_val
            else:
                result = image.copy()
    
    return result

def apply_wavelet_filter(image: np.ndarray,
                        mask: Optional[np.ndarray] = None,
                        params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply wavelet-based denoising to an image.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (wavelet, mode, method)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    wavelet = params.get('wavelet', 'db1')
    mode = params.get('mode', 'soft')
    method = params.get('method', 'BayesShrink')
    
    # Apply wavelet denoising
    if image.ndim == 3:
        # Process 3D volume slice by slice
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            if mask is not None:
                # Apply filter only to masked region in this slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for wavelet denoising
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                    denoised_slice = denoise_wavelet(temp, 
                                                 wavelet=wavelet, 
                                                 mode=mode, 
                                                 method=method,
                                                 multichannel=False)
                    
                    # Rescale back to original range
                    denoised_slice = denoised_slice * (max_val - min_val) + min_val
                    
                    mask_slice = mask[i] if mask.ndim == 3 else mask
                    result[i] = image[i].copy()
                    result[i][mask_slice > 0] = denoised_slice[mask_slice > 0]
                else:
                    result[i] = image[i].copy()
            else:
                # Apply filter to entire slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for wavelet denoising
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                    denoised_slice = denoise_wavelet(temp, 
                                                 wavelet=wavelet, 
                                                 mode=mode, 
                                                 method=method,
                                                 multichannel=False)
                    
                    # Rescale back to original range
                    result[i] = denoised_slice * (max_val - min_val) + min_val
                else:
                    result[i] = image[i].copy()
    else:
        # Process 2D image
        if mask is not None:
            # Apply filter only to masked region
            result = image.copy()
            # Normalize image to [0,1] for wavelet denoising
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_wavelet(temp, 
                                       wavelet=wavelet, 
                                       mode=mode, 
                                       method=method,
                                       multichannel=False)
                
                # Rescale back to original range
                denoised = denoised * (max_val - min_val) + min_val
                
                result[mask > 0] = denoised[mask > 0]
            
        else:
            # Apply filter to entire image
            # Normalize image to [0,1] for wavelet denoising
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_wavelet(temp, 
                                       wavelet=wavelet, 
                                       mode=mode, 
                                       method=method,
                                       multichannel=False)
                
                # Rescale back to original range
                result = denoised * (max_val - min_val) + min_val
            else:
                result = image.copy()
    
    return result

def apply_tv_filter(image: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply Total Variation denoising to an image.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (weight, eps, max_num_iter)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    weight = params.get('weight', 0.1)
    eps = params.get('eps', 2e-4)
    max_num_iter = params.get('max_num_iter', 200)
    
    # Apply Total Variation denoising
    if image.ndim == 3:
        # Process 3D volume slice by slice
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            if mask is not None:
                # Apply filter only to masked region in this slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for TV denoising
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                    denoised_slice = denoise_tv_chambolle(temp, 
                                                      weight=weight, 
                                                      eps=eps, 
                                                      max_num_iter=max_num_iter,
                                                      multichannel=False)
                    
                    # Rescale back to original range
                    denoised_slice = denoised_slice * (max_val - min_val) + min_val
                    
                    mask_slice = mask[i] if mask.ndim == 3 else mask
                    result[i] = image[i].copy()
                    result[i][mask_slice > 0] = denoised_slice[mask_slice > 0]
                else:
                    result[i] = image[i].copy()
            else:
                # Apply filter to entire slice
                temp = image[i].copy()
                # Normalize slice to [0,1] for TV denoising
                min_val = np.min(temp)
                max_val = np.max(temp)
                if max_val > min_val:
                    temp = (temp - min_val) / (max_val - min_val)
                    
                    denoised_slice = denoise_tv_chambolle(temp, 
                                                      weight=weight, 
                                                      eps=eps, 
                                                      max_num_iter=max_num_iter,
                                                      multichannel=False)
                    
                    # Rescale back to original range
                    result[i] = denoised_slice * (max_val - min_val) + min_val
                else:
                    result[i] = image[i].copy()
    else:
        # Process 2D image
        if mask is not None:
            # Apply filter only to masked region
            result = image.copy()
            # Normalize image to [0,1] for TV denoising
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_tv_chambolle(temp, 
                                            weight=weight, 
                                            eps=eps, 
                                            max_num_iter=max_num_iter,
                                            multichannel=False)
                
                # Rescale back to original range
                denoised = denoised * (max_val - min_val) + min_val
                
                result[mask > 0] = denoised[mask > 0]
            
        else:
            # Apply filter to entire image
            # Normalize image to [0,1] for TV denoising
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                temp = (image - min_val) / (max_val - min_val)
                
                denoised = denoise_tv_chambolle(temp, 
                                            weight=weight, 
                                            eps=eps, 
                                            max_num_iter=max_num_iter,
                                            multichannel=False)
                
                # Rescale back to original range
                result = denoised * (max_val - min_val) + min_val
            else:
                result = image.copy()
    
    return result

def apply_anisotropic_diffusion(image: np.ndarray,
                               mask: Optional[np.ndarray] = None,
                               params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Apply anisotropic diffusion filtering to an image.
    This is implemented using SimpleITK's CurvatureAnisotropicDiffusionImageFilter.
    
    Args:
        image: Input image as numpy array
        mask: Optional mask to restrict denoising to specific regions
        params: Additional parameters (conductance, time_step, iterations)
    
    Returns:
        Denoised image as numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    conductance = params.get('conductance', 1.0)
    time_step = params.get('time_step', 0.0625)
    iterations = params.get('iterations', 5)
    
    # Convert to SimpleITK image for processing
    sitk_image = sitk.GetImageFromArray(image)
    
    # Create anisotropic diffusion filter
    diffusion_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
    diffusion_filter.SetConductanceParameter(conductance)
    diffusion_filter.SetTimeStep(time_step)
    diffusion_filter.SetNumberOfIterations(iterations)
    
    # Apply filter
    diffused_image = diffusion_filter.Execute(sitk_image)
    
    # Convert back to numpy array
    result_array = sitk.GetArrayFromImage(diffused_image)
    
    # Apply mask if provided
    if mask is not None:
        # Apply filtered result only to masked region
        result = image.copy()
        result[mask > 0] = result_array[mask > 0]
        return result
    else:
        return result_array 