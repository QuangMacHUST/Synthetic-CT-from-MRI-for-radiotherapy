#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for air cavity segmentation from MRI images.

This module provides various methods for automatic segmentation of air cavities from MRI images,
which is a critical step in the preprocessing pipeline for synthetic CT generation.
Air cavities typically appear as very low intensity regions in MRI and correspond to 
significantly negative HU values in CT (approximately -1000 HU).
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

def segment_air_cavities(image: Union[sitk.Image, np.ndarray],
                        method: str = 'threshold',
                        bone_mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                        params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Segment air cavities from an MRI image.
    
    Args:
        image: Input MRI image as SimpleITK image or numpy array
        method: Segmentation method ('threshold', 'region_growing', 'model')
        bone_mask: Optional bone mask to exclude bone regions
        params: Additional parameters for the specific segmentation method
    
    Returns:
        Air cavity segmentation mask
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
    
    # Convert bone mask to numpy if needed
    if bone_mask is not None and isinstance(bone_mask, sitk.Image):
        bone_mask_array = sitk.GetArrayFromImage(bone_mask)
    else:
        bone_mask_array = bone_mask
    
    # Apply appropriate segmentation method
    if method.lower() == 'threshold':
        segmentation_array = threshold_air_segmentation(img_array, bone_mask_array, params)
    elif method.lower() == 'region_growing':
        segmentation_array = region_growing_air_segmentation(img_array, bone_mask_array, params)
    elif method.lower() == 'model':
        segmentation_array = model_based_air_segmentation(img_array, bone_mask_array, params)
    else:
        logger.warning(f"Unknown air cavity segmentation method '{method}'. Using threshold.")
        segmentation_array = threshold_air_segmentation(img_array, bone_mask_array, params)
    
    # Convert back to SimpleITK if the input was SimpleITK
    if is_sitk:
        segmentation_image = sitk.GetImageFromArray(segmentation_array)
        segmentation_image.SetSpacing(spacing)
        segmentation_image.SetOrigin(origin)
        segmentation_image.SetDirection(direction)
        return segmentation_image
    else:
        return segmentation_array

def threshold_air_segmentation(image: np.ndarray,
                             bone_mask: Optional[np.ndarray] = None,
                             params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment air cavities using intensity thresholding and morphological operations.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        params: Additional parameters (percentile, dilation_radius)
    
    Returns:
        Air cavity segmentation mask
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Air appears as very low intensity in MRI
    # Use a percentile-based threshold or other method to find the lowest intensities
    percentile = params.get('percentile', 5)
    threshold = np.percentile(image, percentile)
    
    # Create initial air mask
    air_mask = image <= threshold
    
    # Exclude bone regions if provided
    if bone_mask is not None:
        air_mask[bone_mask > 0] = False
    
    # Morphological operations to clean up the segmentation
    # Exclude small isolated air pockets
    min_size = params.get('min_size', 50)
    air_mask = morphology.remove_small_objects(air_mask, min_size=min_size)
    
    # Fill small holes in the air regions
    max_hole_size = params.get('max_hole_size', 10)
    air_mask = ndimage.binary_fill_holes(air_mask)
    
    # Additional morphological operations
    # Dilate the air regions to ensure they include the borders
    dilation_radius = params.get('dilation_radius', 1)
    if dilation_radius > 0:
        air_mask = morphology.binary_dilation(
            air_mask, 
            morphology.ball(dilation_radius) if image.ndim == 3 else morphology.disk(dilation_radius)
        )
    
    # Connect nearby air regions (useful for sinuses, nasal cavities, etc.)
    connectivity = params.get('connectivity', 1)
    if connectivity > 1:
        air_mask = morphology.binary_closing(
            air_mask, 
            morphology.ball(connectivity) if image.ndim == 3 else morphology.disk(connectivity)
        )
    
    # Label the connected components to identify distinct air cavities
    if params.get('label_components', False):
        # Background (non-air) is 0, air regions are labeled 1, 2, ...
        air_mask = measure.label(air_mask, connectivity=1)
    else:
        # Convert to binary mask (1 for air, 0 for everything else)
        air_mask = air_mask.astype(np.uint8)
    
    return air_mask

def region_growing_air_segmentation(image: np.ndarray,
                                  bone_mask: Optional[np.ndarray] = None,
                                  params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment air cavities using region growing methods.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        params: Additional parameters (tolerance, n_seeds)
    
    Returns:
        Air cavity segmentation mask
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Get the lowest intensity percentile for seed selection
    percentile = params.get('percentile', 1)
    low_threshold = np.percentile(image, percentile)
    
    # Find potential air cavity seed points (very low intensity voxels)
    seed_mask = image <= low_threshold
    
    # Exclude bone regions if provided
    if bone_mask is not None:
        seed_mask[bone_mask > 0] = False
    
    # Remove small isolated regions from seed mask
    min_seed_size = params.get('min_seed_size', 10)
    seed_mask = morphology.remove_small_objects(seed_mask, min_size=min_seed_size)
    
    # Get seed points from the seed mask
    n_seeds = params.get('n_seeds', 10)
    seed_points = []
    
    # Label connected components in the seed mask
    labeled_seeds = measure.label(seed_mask)
    regions = np.unique(labeled_seeds)
    regions = regions[regions > 0]  # Exclude background
    
    # Select the largest regions as seeds
    region_sizes = [(region, np.sum(labeled_seeds == region)) for region in regions]
    region_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Take the n largest regions or all if there are fewer
    selected_regions = region_sizes[:min(n_seeds, len(region_sizes))]
    
    for region_label, _ in selected_regions:
        region_mask = labeled_seeds == region_label
        # Find the centroid of the region
        centroid = ndimage.center_of_mass(region_mask)
        seed_points.append(tuple(map(int, centroid)))
    
    # If no seeds were found, fall back to threshold-based segmentation
    if not seed_points:
        logger.warning("No suitable seed points found for region growing. Using threshold-based segmentation.")
        return threshold_air_segmentation(image, bone_mask, params)
    
    # Initialize air cavity mask
    air_mask = np.zeros_like(image, dtype=bool)
    
    # Region growing tolerance (how much intensity can vary from the seed)
    tolerance = params.get('tolerance', 0.1)
    
    # Perform region growing from each seed point
    for seed_point in seed_points:
        # Skip if the seed is already in an air cavity
        if air_mask[seed_point]:
            continue
        
        # Find the seed's intensity
        seed_intensity = image[seed_point]
        
        # Create a mask for acceptable intensity range
        # For air, we want to include only very low intensities
        acceptable_mask = image <= (seed_intensity + tolerance)
        
        # Exclude bone regions if provided
        if bone_mask is not None:
            acceptable_mask[bone_mask > 0] = False
        
        # Exclude already segmented air regions
        acceptable_mask[air_mask] = False
        
        # Perform region growing
        region_mask = _region_growing(image, seed_point, tolerance, acceptable_mask)
        
        # Add the region to the air mask
        air_mask = air_mask | region_mask
    
    # Post-processing to clean up the segmentation
    min_size = params.get('min_size', 50)
    air_mask = morphology.remove_small_objects(air_mask, min_size=min_size)
    
    # Fill small holes
    max_hole_size = params.get('max_hole_size', 10)
    air_mask = ndimage.binary_fill_holes(air_mask)
    
    # Connect nearby air regions
    if params.get('connect_regions', True):
        connectivity = params.get('connectivity', 1)
        air_mask = morphology.binary_closing(
            air_mask, 
            morphology.ball(connectivity) if image.ndim == 3 else morphology.disk(connectivity)
        )
    
    # Label connected components if requested
    if params.get('label_components', False):
        # Background is 0, air regions are labeled 1, 2, ...
        air_mask = measure.label(air_mask, connectivity=1)
    else:
        # Convert to binary mask (1 for air, 0 for everything else)
        air_mask = air_mask.astype(np.uint8)
    
    return air_mask

def _region_growing(image: np.ndarray, seed_point: Tuple, tolerance: float, mask: np.ndarray) -> np.ndarray:
    """
    Helper function for region growing.
    
    Args:
        image: Input image
        seed_point: Starting point for region growing
        tolerance: Maximum intensity difference allowed
        mask: Mask of regions to consider
    
    Returns:
        Binary mask of the grown region
    """
    # Initialize region with the seed point
    region = np.zeros_like(image, dtype=bool)
    seed_value = image[seed_point]
    
    # Create a queue for the points to process
    queue = [seed_point]
    region[seed_point] = True
    
    # Define neighborhood
    if image.ndim == 3:
        neighborhood = [
            (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)
        ]
    else:
        neighborhood = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Process points in the queue
    while queue:
        current_point = queue.pop(0)
        
        # Check neighbors
        for offset in neighborhood:
            neighbor = tuple(p + o for p, o in zip(current_point, offset))
            
            # Check if neighbor is within image bounds
            if all(0 <= p < s for p, s in zip(neighbor, image.shape)):
                # Check if neighbor is within tolerance and not already in region
                if (not region[neighbor] and mask[neighbor] and 
                    image[neighbor] <= (seed_value + tolerance)):
                    # Add to region and queue
                    region[neighbor] = True
                    queue.append(neighbor)
    
    return region

def model_based_air_segmentation(image: np.ndarray,
                               bone_mask: Optional[np.ndarray] = None,
                               params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment air cavities using a pre-trained deep learning model.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        params: Additional parameters (model_path, region)
    
    Returns:
        Air cavity segmentation mask
    """
    # Default parameters
    if params is None:
        params = {}
    
    model_path = params.get('model_path', None)
    region = params.get('region', 'head')  # Anatomical region (head, pelvis, thorax)
    
    if model_path is None:
        logger.error("No model path provided for model-based air segmentation.")
        logger.warning("Falling back to threshold-based segmentation.")
        return threshold_air_segmentation(image, bone_mask, params)
    
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
            # Assuming the output is a binary mask (0 = background, 1 = air)
            air_prob = prediction[0, ..., 0]
            
            # Apply threshold to get binary mask
            threshold = params.get('prob_threshold', 0.5)
            binary_mask = air_prob > threshold
            
            # Resize back to original shape
            from skimage.transform import resize
            if len(orig_shape) == 3:
                # Create 3D mask with zeros
                air_mask = np.zeros(orig_shape, dtype=np.uint8)
                
                # Fill the middle slice with the prediction
                resized_slice = resize(
                    binary_mask, 
                    (orig_shape[1], orig_shape[2]), 
                    order=0,  # Nearest neighbor to preserve binary values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
                
                air_mask[middle_slice] = resized_slice
            else:
                air_mask = resize(
                    binary_mask, 
                    orig_shape, 
                    order=0,  # Nearest neighbor to preserve binary values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
        else:
            # For other regions or full 3D models
            # Implement specific logic based on model requirements
            logger.warning(f"Model-based air segmentation for region '{region}' not fully implemented.")
            logger.warning("Falling back to threshold-based segmentation.")
            return threshold_air_segmentation(image, bone_mask, params)
        
        # Exclude bone regions if provided
        if bone_mask is not None:
            air_mask[bone_mask > 0] = 0
        
        # Post-processing to clean up the segmentation
        min_size = params.get('min_size', 50)
        air_mask = morphology.remove_small_objects(air_mask.astype(bool), min_size=min_size).astype(np.uint8)
        
        # Fill small holes
        max_hole_size = params.get('max_hole_size', 10)
        air_mask = ndimage.binary_fill_holes(air_mask).astype(np.uint8)
        
        return air_mask
    
    except Exception as e:
        logger.error(f"Model-based air segmentation failed: {str(e)}")
        logger.warning("Falling back to threshold-based segmentation.")
        return threshold_air_segmentation(image, bone_mask, params) 