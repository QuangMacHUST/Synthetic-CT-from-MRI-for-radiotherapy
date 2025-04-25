#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for bone segmentation from MRI images.

This module provides various methods for automatic bone segmentation from MRI,
which is a critical step in the preprocessing pipeline for synthetic CT generation.
Different approaches are implemented to handle the challenges of bone segmentation in MRI,
where bone appears with low signal intensity and can be difficult to distinguish from air.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Union, Optional, Tuple, List
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation

# Set up logger
logger = logging.getLogger(__name__)

def segment_bones(image: Union[sitk.Image, np.ndarray],
                 method: str = 'thresholding',
                 mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                 params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Segment bone structures from an MRI image.
    
    Args:
        image: Input MRI image as SimpleITK image or numpy array
        method: Segmentation method ('thresholding', 'multithresholding', 'morphological', 'model')
        mask: Optional mask to restrict segmentation to specific regions
        params: Additional parameters for the specific segmentation method
    
    Returns:
        Bone segmentation mask in the same format as input
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
    
    # Apply appropriate segmentation method
    if method.lower() == 'thresholding':
        segmentation_array = threshold_based_bone_segmentation(img_array, mask_array, params)
    elif method.lower() == 'multithresholding':
        segmentation_array = multi_threshold_bone_segmentation(img_array, mask_array, params)
    elif method.lower() == 'morphological':
        segmentation_array = morphological_bone_segmentation(img_array, mask_array, params)
    elif method.lower() == 'model':
        segmentation_array = model_based_bone_segmentation(img_array, mask_array, params)
    else:
        logger.warning(f"Unknown bone segmentation method '{method}'. Using thresholding.")
        segmentation_array = threshold_based_bone_segmentation(img_array, mask_array, params)
    
    # Convert back to SimpleITK if the input was SimpleITK
    if is_sitk:
        segmentation_image = sitk.GetImageFromArray(segmentation_array.astype(np.uint8))
        segmentation_image.SetSpacing(spacing)
        segmentation_image.SetOrigin(origin)
        segmentation_image.SetDirection(direction)
        return segmentation_image
    else:
        return segmentation_array.astype(np.uint8)

def threshold_based_bone_segmentation(image: np.ndarray,
                                    mask: Optional[np.ndarray] = None,
                                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment bone structures using intensity thresholding.
    
    Args:
        image: Input MRI image as numpy array
        mask: Optional mask to restrict segmentation to specific regions
        params: Additional parameters (lower_threshold, upper_threshold)
    
    Returns:
        Bone segmentation mask as binary numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    
    # In T1-weighted MRI, bones typically have low signal intensity
    # The default thresholds assume normalized image [0, 1]
    lower_threshold = params.get('lower_threshold', None)
    upper_threshold = params.get('upper_threshold', 0.2)  # Bones have low intensity
    
    # Apply mask if provided
    if mask is not None:
        masked_image = np.copy(image)
        masked_image[mask == 0] = np.nan  # Set regions outside mask to NaN
    else:
        masked_image = image
    
    # Determine threshold automatically if not provided
    if lower_threshold is None:
        # Compute histogram
        hist, bin_edges = np.histogram(masked_image[~np.isnan(masked_image)].flatten(), bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find the first minimum after the first peak in the histogram
        # This often separates background/bone from soft tissues
        smoothed_hist = ndimage.gaussian_filter1d(hist, sigma=2)
        peaks = np.where((smoothed_hist[1:-1] > smoothed_hist[:-2]) & 
                       (smoothed_hist[1:-1] > smoothed_hist[2:]))[0] + 1
        
        if len(peaks) > 0:
            first_peak = peaks[0]
            # Find the first minimum after the first peak
            for i in range(first_peak + 1, len(smoothed_hist) - 1):
                if smoothed_hist[i] < smoothed_hist[i-1] and smoothed_hist[i] < smoothed_hist[i+1]:
                    lower_threshold = bin_centers[i]
                    break
        
        # If no minimum found, use Otsu's method
        if lower_threshold is None:
            lower_threshold = 0
    
    # Apply thresholding
    bone_mask = np.zeros_like(image, dtype=bool)
    if upper_threshold is not None:
        bone_mask[(image >= lower_threshold) & (image <= upper_threshold)] = True
    else:
        bone_mask[image >= lower_threshold] = True
    
    # Apply mask if provided
    if mask is not None:
        bone_mask[mask == 0] = False
    
    # Post-processing to remove small isolated regions
    min_size = params.get('min_size', 100)
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=min_size)
    
    # Fill holes in the segmentation
    if params.get('fill_holes', True):
        bone_mask = ndimage.binary_fill_holes(bone_mask)
    
    # Morphological operations to refine the segmentation
    if params.get('apply_closing', True):
        closing_radius = params.get('closing_radius', 2)
        bone_mask = morphology.binary_closing(bone_mask, morphology.ball(closing_radius))
    
    return bone_mask.astype(np.uint8)

def multi_threshold_bone_segmentation(image: np.ndarray,
                                    mask: Optional[np.ndarray] = None,
                                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment bone structures using multi-level thresholding.
    
    Args:
        image: Input MRI image as numpy array
        mask: Optional mask to restrict segmentation to specific regions
        params: Additional parameters (thresholds, connectivity)
    
    Returns:
        Bone segmentation mask as binary numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    
    # In MRI, bones typically have low signal intensity with clear boundaries
    n_classes = params.get('n_classes', 4)  # Number of intensity classes
    connectivity = params.get('connectivity', 2)  # Connectivity for removing small objects
    bone_class = params.get('bone_class', 0)  # Typically bones are the darkest class (0)
    
    # Apply mask if provided
    if mask is not None:
        working_image = np.copy(image)
        working_image[mask == 0] = np.nan
    else:
        working_image = image
    
    # Apply multi-Otsu thresholding
    try:
        thresholds = filters.threshold_multiotsu(
            working_image[~np.isnan(working_image)], classes=n_classes
        )
        
        # Create labeled image based on thresholds
        regions = np.digitize(image, bins=thresholds)
        
        # Extract bone regions (typically the darkest class)
        bone_mask = regions == bone_class
    except Exception as e:
        logger.error(f"Multi-Otsu thresholding failed: {str(e)}")
        logger.warning("Falling back to regular Otsu thresholding.")
        
        # Fall back to regular Otsu thresholding
        threshold = filters.threshold_otsu(working_image[~np.isnan(working_image)])
        bone_mask = image <= threshold
    
    # Apply mask if provided
    if mask is not None:
        bone_mask[mask == 0] = False
    
    # Post-processing to remove small isolated regions
    min_size = params.get('min_size', 100)
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=min_size)
    
    # Fill holes in the segmentation
    if params.get('fill_holes', True):
        bone_mask = ndimage.binary_fill_holes(bone_mask)
    
    # Morphological operations to refine the segmentation
    if params.get('apply_closing', True):
        closing_radius = params.get('closing_radius', 2)
        bone_mask = morphology.binary_closing(bone_mask, morphology.ball(closing_radius))
    
    return bone_mask.astype(np.uint8)

def morphological_bone_segmentation(image: np.ndarray,
                                   mask: Optional[np.ndarray] = None,
                                   params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment bone structures using morphological operations.
    
    Args:
        image: Input MRI image as numpy array
        mask: Optional mask to restrict segmentation to specific regions
        params: Additional parameters (edge_threshold, marker_threshold)
    
    Returns:
        Bone segmentation mask as binary numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Parameters for edge detection and watershed segmentation
    edge_threshold = params.get('edge_threshold', None)
    marker_threshold = params.get('marker_threshold', None)
    sigma = params.get('sigma', 1.0)  # Gaussian smoothing sigma
    
    # Apply mask if provided
    if mask is not None:
        working_image = np.copy(image)
        working_image[mask == 0] = np.nan
    else:
        working_image = image
    
    # Step 1: Smooth the image to reduce noise
    smoothed = ndimage.gaussian_filter(image, sigma=sigma)
    
    # Step 2: Calculate gradient magnitude (edges)
    gradient = filters.sobel(smoothed)
    
    # Determine edge threshold automatically if not provided
    if edge_threshold is None:
        edge_threshold = filters.threshold_otsu(gradient)
    
    # Step 3: Mark the foreground objects
    if marker_threshold is None:
        marker_threshold = filters.threshold_otsu(working_image[~np.isnan(working_image)])
    
    # Create markers for watershed
    markers = np.zeros_like(image, dtype=int)
    markers[image < marker_threshold] = 1  # Potential bone regions
    markers[image > 0.8 * np.nanmax(working_image)] = 2  # Definite non-bone regions
    
    # Step 4: Apply watershed segmentation
    watershed_result = segmentation.watershed(gradient, markers)
    
    # Extract bone regions (label 1)
    bone_mask = watershed_result == 1
    
    # Apply mask if provided
    if mask is not None:
        bone_mask[mask == 0] = False
    
    # Post-processing to remove small isolated regions
    min_size = params.get('min_size', 100)
    bone_mask = morphology.remove_small_objects(bone_mask, min_size=min_size)
    
    # Fill holes in the segmentation
    if params.get('fill_holes', True):
        bone_mask = ndimage.binary_fill_holes(bone_mask)
    
    # Morphological operations to refine the segmentation
    if params.get('apply_closing', True):
        closing_radius = params.get('closing_radius', 2)
        bone_mask = morphology.binary_closing(bone_mask, morphology.ball(closing_radius))
    
    return bone_mask.astype(np.uint8)

def model_based_bone_segmentation(image: np.ndarray,
                                mask: Optional[np.ndarray] = None,
                                params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment bone structures using a pre-trained model.
    
    Args:
        image: Input MRI image as numpy array
        mask: Optional mask to restrict segmentation to specific regions
        params: Additional parameters (model_path, region)
    
    Returns:
        Bone segmentation mask as binary numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    
    model_path = params.get('model_path', None)
    region = params.get('region', 'head')  # Anatomical region (head, pelvis, thorax)
    
    if model_path is None:
        logger.error("No model path provided for model-based bone segmentation.")
        logger.warning("Falling back to threshold-based segmentation.")
        return threshold_based_bone_segmentation(image, mask, params)
    
    try:
        # Import deep learning libraries only when needed
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Load the pre-trained model
        model = load_model(model_path)
        
        # Prepare the image for the model
        # Typically, we need to normalize, reshape, and add batch dimension
        normalized_image = (image - np.nanmean(image)) / (np.nanstd(image) + 1e-8)
        
        # Model may expect specific input shape
        orig_shape = normalized_image.shape
        
        # Handle different model input requirements based on the region
        if region.lower() == 'head':
            # Example preprocessing for head
            input_shape = params.get('input_shape', (256, 256, 1))
            # Reshape and add batch dimension
            reshaped = np.zeros(input_shape)
            
            # For 3D images, process middle slice or multiple slices
            if len(orig_shape) == 3:
                middle_slice = orig_shape[0] // 2
                from skimage.transform import resize
                
                # For simplicity, just use the middle slice
                reshaped[..., 0] = resize(
                    normalized_image[middle_slice], 
                    input_shape[:2], 
                    mode='constant'
                )
            else:
                from skimage.transform import resize
                reshaped[..., 0] = resize(
                    normalized_image, 
                    input_shape[:2], 
                    mode='constant'
                )
            
            # Add batch dimension
            model_input = np.expand_dims(reshaped, axis=0)
            
            # Run the model
            prediction = model.predict(model_input)
            
            # Post-process the prediction
            # The output is typically a probability map that needs thresholding
            threshold = params.get('threshold', 0.5)
            bone_mask_resized = (prediction[0, ..., 0] > threshold).astype(np.uint8)
            
            # Resize back to original shape
            from skimage.transform import resize
            if len(orig_shape) == 3:
                # Create 3D mask with zeros
                bone_mask = np.zeros(orig_shape, dtype=np.uint8)
                
                # Fill the middle slice with the prediction
                resized_slice = resize(
                    bone_mask_resized, 
                    (orig_shape[1], orig_shape[2]), 
                    order=0,  # Nearest neighbor to preserve binary values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
                
                bone_mask[middle_slice] = resized_slice
            else:
                bone_mask = resize(
                    bone_mask_resized, 
                    orig_shape, 
                    order=0,  # Nearest neighbor to preserve binary values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
        else:
            # For other regions or full 3D models
            # Implement specific logic based on model requirements
            logger.warning(f"Model-based segmentation for region '{region}' not fully implemented.")
            logger.warning("Falling back to threshold-based segmentation.")
            return threshold_based_bone_segmentation(image, mask, params)
        
        # Apply mask if provided
        if mask is not None:
            bone_mask[mask == 0] = 0
        
        # Post-processing to refine segmentation
        if params.get('apply_postprocessing', True):
            # Remove small objects
            min_size = params.get('min_size', 100)
            bone_mask = morphology.remove_small_objects(bone_mask.astype(bool), min_size=min_size)
            
            # Fill holes
            if params.get('fill_holes', True):
                bone_mask = ndimage.binary_fill_holes(bone_mask)
            
            # Apply closing
            if params.get('apply_closing', True):
                closing_radius = params.get('closing_radius', 2)
                if len(orig_shape) == 3:
                    bone_mask = morphology.binary_closing(
                        bone_mask, 
                        morphology.ball(closing_radius)
                    )
                else:
                    bone_mask = morphology.binary_closing(
                        bone_mask, 
                        morphology.disk(closing_radius)
                    )
        
        return bone_mask.astype(np.uint8)
    
    except Exception as e:
        logger.error(f"Model-based bone segmentation failed: {str(e)}")
        logger.warning("Falling back to threshold-based segmentation.")
        return threshold_based_bone_segmentation(image, mask, params)

def refine_bone_segmentation(image: np.ndarray,
                           bone_mask: np.ndarray,
                           params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Refine a bone segmentation mask using image features.
    
    Args:
        image: Original MRI image as numpy array
        bone_mask: Initial bone segmentation mask as binary numpy array
        params: Additional parameters for refinement
    
    Returns:
        Refined bone segmentation mask as binary numpy array
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Edge-aware refinement
    if params.get('edge_aware_refinement', True):
        # Calculate gradient magnitude (edges)
        gradient = filters.sobel(image)
        gradient_threshold = params.get('gradient_threshold', 0.1)
        edge_mask = gradient > gradient_threshold
        
        # Dilate the bone mask
        dilate_radius = params.get('dilate_radius', 2)
        dilated_mask = morphology.binary_dilation(
            bone_mask,
            morphology.ball(dilate_radius) if bone_mask.ndim == 3 else morphology.disk(dilate_radius)
        )
        
        # Combine the dilated mask with the edge information
        # Keep bone regions and add edge regions that are in the dilated mask
        refined_mask = bone_mask | (edge_mask & dilated_mask)
    else:
        refined_mask = bone_mask
    
    # Connected component analysis to remove isolated regions
    if params.get('remove_isolated_regions', True):
        min_size = params.get('min_size', 100)
        refined_mask = morphology.remove_small_objects(refined_mask.astype(bool), min_size=min_size)
    
    # Fill holes in the segmentation
    if params.get('fill_holes', True):
        refined_mask = ndimage.binary_fill_holes(refined_mask)
    
    # Smoothing
    if params.get('apply_smoothing', True):
        # Apply morphological closing to smooth boundaries
        closing_radius = params.get('closing_radius', 2)
        refined_mask = morphology.binary_closing(
            refined_mask,
            morphology.ball(closing_radius) if refined_mask.ndim == 3 else morphology.disk(closing_radius)
        )
    
    return refined_mask.astype(np.uint8) 