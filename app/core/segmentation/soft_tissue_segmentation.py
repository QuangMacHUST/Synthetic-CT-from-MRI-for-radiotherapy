#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for soft tissue segmentation from MRI images.

This module provides various methods for automatic soft tissue segmentation from MRI,
which is a critical step in the preprocessing pipeline for synthetic CT generation.
Different approaches are implemented to handle the challenges of soft tissue segmentation in MRI,
to distinguish between various soft tissue types with different density properties.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any, Union, Optional, Tuple, List
from scipy import ndimage
from skimage import filters, measure, morphology, segmentation, feature

# Set up logger
logger = logging.getLogger(__name__)

def segment_soft_tissues(image: Union[sitk.Image, np.ndarray],
                        method: str = 'multithreshold',
                        bone_mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                        air_mask: Optional[Union[sitk.Image, np.ndarray]] = None,
                        params: Optional[Dict[str, Any]] = None) -> Union[sitk.Image, np.ndarray]:
    """
    Segment soft tissue structures from an MRI image.
    
    Args:
        image: Input MRI image as SimpleITK image or numpy array
        method: Segmentation method ('multithreshold', 'fuzzy_cmeans', 'region_growing', 'model', 'brain')
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters for the specific segmentation method
    
    Returns:
        Soft tissue segmentation mask with labels for different tissue types
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
    
    # Convert masks to numpy if needed
    if bone_mask is not None and isinstance(bone_mask, sitk.Image):
        bone_mask_array = sitk.GetArrayFromImage(bone_mask)
    else:
        bone_mask_array = bone_mask
    
    if air_mask is not None and isinstance(air_mask, sitk.Image):
        air_mask_array = sitk.GetArrayFromImage(air_mask)
    else:
        air_mask_array = air_mask
    
    # Apply appropriate segmentation method
    if method.lower() == 'multithreshold':
        segmentation_array = multithreshold_tissue_segmentation(img_array, bone_mask_array, air_mask_array, params)
    elif method.lower() == 'fuzzy_cmeans':
        segmentation_array = fuzzy_cmeans_tissue_segmentation(img_array, bone_mask_array, air_mask_array, params)
    elif method.lower() == 'region_growing':
        segmentation_array = region_growing_tissue_segmentation(img_array, bone_mask_array, air_mask_array, params)
    elif method.lower() == 'model':
        segmentation_array = model_based_tissue_segmentation(img_array, bone_mask_array, air_mask_array, params)
    elif method.lower() == 'brain':
        segmentation_array = segment_brain_tissues(img_array, bone_mask_array, air_mask_array, params)
    else:
        logger.warning(f"Unknown soft tissue segmentation method '{method}'. Using multithreshold.")
        segmentation_array = multithreshold_tissue_segmentation(img_array, bone_mask_array, air_mask_array, params)
    
    # Convert back to SimpleITK if the input was SimpleITK
    if is_sitk:
        segmentation_image = sitk.GetImageFromArray(segmentation_array)
        segmentation_image.SetSpacing(spacing)
        segmentation_image.SetOrigin(origin)
        segmentation_image.SetDirection(direction)
        return segmentation_image
    else:
        return segmentation_array

def multithreshold_tissue_segmentation(image: np.ndarray,
                                     bone_mask: Optional[np.ndarray] = None,
                                     air_mask: Optional[np.ndarray] = None,
                                     params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment soft tissues using multi-level thresholding.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters (n_tissue_classes)
    
    Returns:
        Soft tissue segmentation mask with labels for different tissue types
    """
    # Default parameters
    if params is None:
        params = {}
    
    n_tissue_classes = params.get('n_tissue_classes', 4)  # Number of soft tissue classes
    
    # Create a mask of regions to be segmented (exclude bone and air)
    mask = np.ones_like(image, dtype=bool)
    if bone_mask is not None:
        mask[bone_mask > 0] = False
    if air_mask is not None:
        mask[air_mask > 0] = False
    
    # Apply multi-Otsu thresholding to the masked region
    try:
        # Extract intensity values within the mask
        masked_values = image[mask]
        
        # Perform multi-Otsu thresholding
        thresholds = filters.threshold_multiotsu(masked_values, classes=n_tissue_classes + 1)
        
        # Create a labeled array for the full image
        segmentation = np.zeros_like(image, dtype=np.uint8)
        
        # Assign labels to the image based on thresholds
        for i in range(len(thresholds) + 1):
            if i == 0:
                # Regions below the first threshold
                idx = (image <= thresholds[i]) & mask
            elif i == len(thresholds):
                # Regions above the last threshold
                idx = (image > thresholds[i-1]) & mask
            else:
                # Regions between thresholds
                idx = (image > thresholds[i-1]) & (image <= thresholds[i]) & mask
            
            segmentation[idx] = i + 1  # Start from label 1
        
        # Include bone and air masks in the segmentation
        if bone_mask is not None:
            segmentation[bone_mask > 0] = n_tissue_classes + 2  # Bone label
        if air_mask is not None:
            segmentation[air_mask > 0] = n_tissue_classes + 3  # Air label
        
        # Remove small isolated regions
        min_size = params.get('min_size', 100)
        for label in range(1, n_tissue_classes + 2):
            binary_mask = segmentation == label
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
            # Update regions that were removed
            removed = binary_mask & ~cleaned_mask
            # Assign removed regions to the nearest label
            if np.any(removed):
                distance_maps = []
                labels_to_check = list(range(1, n_tissue_classes + 4))
                labels_to_check.remove(label)
                for l in labels_to_check:
                    # Compute distance map to each label
                    label_mask = segmentation == l
                    if np.any(label_mask):
                        dist_map = ndimage.distance_transform_edt(~label_mask)
                        distance_maps.append((l, dist_map))
                
                if distance_maps:
                    # For each removed voxel, find the nearest label
                    removed_indices = np.where(removed)
                    for idx in zip(*removed_indices):
                        min_dist = float('inf')
                        nearest_label = 0
                        for l, dist_map in distance_maps:
                            if dist_map[idx] < min_dist:
                                min_dist = dist_map[idx]
                                nearest_label = l
                        segmentation[idx] = nearest_label
    
    except Exception as e:
        logger.error(f"Multi-threshold segmentation failed: {str(e)}")
        logger.warning("Falling back to simple intensity-based segmentation.")
        
        # Simple fallback: divide intensity range into equal bins
        segmentation = np.zeros_like(image, dtype=np.uint8)
        min_val = np.min(image[mask])
        max_val = np.max(image[mask])
        
        if max_val > min_val:
            bin_width = (max_val - min_val) / n_tissue_classes
            for i in range(n_tissue_classes):
                lower = min_val + i * bin_width
                upper = min_val + (i + 1) * bin_width
                
                if i == n_tissue_classes - 1:
                    # Include the upper bound for the last bin
                    idx = (image >= lower) & (image <= upper) & mask
                else:
                    idx = (image >= lower) & (image < upper) & mask
                
                segmentation[idx] = i + 1
            
            # Include bone and air masks
            if bone_mask is not None:
                segmentation[bone_mask > 0] = n_tissue_classes + 1
            if air_mask is not None:
                segmentation[air_mask > 0] = n_tissue_classes + 2
    
    return segmentation

def fuzzy_cmeans_tissue_segmentation(image: np.ndarray,
                                   bone_mask: Optional[np.ndarray] = None,
                                   air_mask: Optional[np.ndarray] = None,
                                   params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment soft tissues using Fuzzy C-means clustering.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters (n_clusters, m)
    
    Returns:
        Soft tissue segmentation mask with labels for different tissue types
    """
    # Default parameters
    if params is None:
        params = {}
    
    n_clusters = params.get('n_clusters', 4)  # Number of tissue clusters
    m = params.get('m', 2)  # Fuzziness parameter
    error = params.get('error', 0.005)  # Convergence threshold
    max_iter = params.get('max_iter', 100)  # Maximum number of iterations
    
    # Create a mask of regions to be segmented (exclude bone and air)
    mask = np.ones_like(image, dtype=bool)
    if bone_mask is not None:
        mask[bone_mask > 0] = False
    if air_mask is not None:
        mask[air_mask > 0] = False
    
    try:
        # Import skfuzzy if available
        try:
            import skfuzzy as fuzz
        except ImportError:
            logger.error("skfuzzy not installed. Please install it with 'pip install scikit-fuzzy'")
            logger.warning("Falling back to multithreshold segmentation.")
            return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
        
        # Extract the data points for clustering
        points = image[mask].flatten()
        points = points.reshape(-1, 1)  # Reshape for clustering
        
        # Normalize the data
        min_val = np.min(points)
        max_val = np.max(points)
        if max_val > min_val:
            points = (points - min_val) / (max_val - min_val)
        
        # Apply Fuzzy C-means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            points.T, n_clusters, m, error=error, maxiter=max_iter, init=None
        )
        
        # Create segmentation array
        segmentation = np.zeros_like(image, dtype=np.uint8)
        
        # Get the cluster label for each point (highest membership value)
        cluster_labels = np.argmax(u, axis=0) + 1  # Start from 1
        
        # Map back to the original image space
        segmentation_flat = np.zeros(image.size, dtype=np.uint8)
        mask_flat = mask.flatten()
        segmentation_flat[mask_flat] = cluster_labels
        segmentation = segmentation_flat.reshape(image.shape)
        
        # Include bone and air masks
        if bone_mask is not None:
            segmentation[bone_mask > 0] = n_clusters + 1
        if air_mask is not None:
            segmentation[air_mask > 0] = n_clusters + 2
        
        # Post-processing to clean up the segmentation
        min_size = params.get('min_size', 100)
        for label in range(1, n_clusters + 1):
            binary_mask = segmentation == label
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
            # Update regions that were removed
            removed = binary_mask & ~cleaned_mask
            # Assign removed regions to the nearest cluster
            if np.any(removed):
                # Find nearest centroids for each removed point
                removed_points = points[mask_flat & removed.flatten()]
                
                # Compute distances to centroids
                nearest_clusters = np.argmin(np.abs(removed_points - cntr.T), axis=1) + 1
                
                # Map back to the segmentation
                removed_indices = np.where(removed)
                for i, idx in enumerate(zip(*removed_indices)):
                    if i < len(nearest_clusters):
                        segmentation[idx] = nearest_clusters[i]
    
    except Exception as e:
        logger.error(f"Fuzzy C-means segmentation failed: {str(e)}")
        logger.warning("Falling back to multithreshold segmentation.")
        return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
    
    return segmentation

def region_growing_tissue_segmentation(image: np.ndarray,
                                     bone_mask: Optional[np.ndarray] = None,
                                     air_mask: Optional[np.ndarray] = None,
                                     params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment soft tissues using region growing methods.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters (n_seeds, tolerance)
    
    Returns:
        Soft tissue segmentation mask with labels for different tissue types
    """
    # Default parameters
    if params is None:
        params = {}
    
    n_seeds = params.get('n_seeds', 10)  # Number of seed points
    tolerance = params.get('tolerance', 0.1)  # Region growing tolerance
    
    # Create a mask of regions to be segmented (exclude bone and air)
    mask = np.ones_like(image, dtype=bool)
    if bone_mask is not None:
        mask[bone_mask > 0] = False
    if air_mask is not None:
        mask[air_mask > 0] = False
    
    try:
        # Initialize segmentation
        segmentation = np.zeros_like(image, dtype=np.uint8)
        
        # Get intensity range
        min_val = np.min(image[mask])
        max_val = np.max(image[mask])
        intensity_range = max_val - min_val
        
        # Generate seed points across the intensity range
        seed_intensities = np.linspace(min_val, max_val, n_seeds)
        
        # For each seed intensity, find a suitable seed point
        current_label = 1
        for seed_intensity in seed_intensities:
            # Find points close to the target intensity
            intensity_mask = np.abs(image - seed_intensity) < (tolerance * intensity_range)
            candidate_points = np.where(intensity_mask & mask & (segmentation == 0))
            
            if len(candidate_points[0]) > 0:
                # Select a random seed point
                idx = np.random.randint(len(candidate_points[0]))
                seed_point = (candidate_points[0][idx], candidate_points[1][idx])
                if image.ndim == 3:
                    seed_point = (candidate_points[0][idx], candidate_points[1][idx], candidate_points[2][idx])
                
                # Apply region growing
                region_mask = _region_growing(
                    image, seed_point, tolerance * intensity_range, mask & (segmentation == 0)
                )
                
                # Add the region to the segmentation
                if np.sum(region_mask) > params.get('min_region_size', 50):
                    segmentation[region_mask] = current_label
                    current_label += 1
        
        # If no regions were segmented, fall back to multithreshold
        if current_label == 1:
            logger.warning("Region growing failed to segment any regions.")
            return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
        
        # Handle any unsegmented regions
        unsegmented = mask & (segmentation == 0)
        if np.any(unsegmented):
            # Assign unsegmented regions to the nearest label based on intensity
            unsegmented_points = image[unsegmented]
            for point_idx, point in zip(zip(*np.where(unsegmented)), unsegmented_points):
                # Find the label with the closest mean intensity
                min_diff = float('inf')
                best_label = 0
                for label in range(1, current_label):
                    label_mask = segmentation == label
                    if np.any(label_mask):
                        label_mean = np.mean(image[label_mask])
                        diff = abs(point - label_mean)
                        if diff < min_diff:
                            min_diff = diff
                            best_label = label
                
                if best_label > 0:
                    segmentation[point_idx] = best_label
        
        # Include bone and air masks
        if bone_mask is not None:
            segmentation[bone_mask > 0] = current_label
            current_label += 1
        if air_mask is not None:
            segmentation[air_mask > 0] = current_label
        
        # Post-processing to clean up the segmentation
        min_size = params.get('min_size', 100)
        for label in range(1, current_label):
            binary_mask = segmentation == label
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
            segmentation[binary_mask & ~cleaned_mask] = 0
        
        # Relabel to ensure consecutive labels
        segmentation = measure.label(segmentation > 0)
        
        # Merge similar regions if there are too many
        max_regions = params.get('max_regions', 10)
        if len(np.unique(segmentation)) - 1 > max_regions:
            segmentation = _merge_similar_regions(image, segmentation, max_regions)
    
    except Exception as e:
        logger.error(f"Region growing segmentation failed: {str(e)}")
        logger.warning("Falling back to multithreshold segmentation.")
        return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
    
    return segmentation

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
                    abs(image[neighbor] - seed_value) <= tolerance):
                    # Add to region and queue
                    region[neighbor] = True
                    queue.append(neighbor)
    
    return region

def _merge_similar_regions(image: np.ndarray, segmentation: np.ndarray, target_regions: int) -> np.ndarray:
    """
    Merge similar regions based on intensity.
    
    Args:
        image: Input image
        segmentation: Current segmentation with many regions
        target_regions: Target number of regions after merging
    
    Returns:
        Updated segmentation with merged regions
    """
    # Get all region labels
    labels = np.unique(segmentation)
    labels = labels[labels > 0]  # Exclude background
    
    if len(labels) <= target_regions:
        return segmentation
    
    # Calculate mean intensity for each region
    region_intensities = {}
    for label in labels:
        region_mask = segmentation == label
        region_intensities[label] = np.mean(image[region_mask])
    
    # Create a new segmentation array
    new_segmentation = np.zeros_like(segmentation)
    
    # Sort regions by intensity
    sorted_regions = sorted(region_intensities.items(), key=lambda x: x[1])
    
    # Group similar regions
    bins = np.array_split(sorted_regions, target_regions)
    
    # Assign new labels
    for i, bin_regions in enumerate(bins):
        for label, _ in bin_regions:
            new_segmentation[segmentation == label] = i + 1
    
    return new_segmentation

def model_based_tissue_segmentation(image: np.ndarray,
                                  bone_mask: Optional[np.ndarray] = None,
                                  air_mask: Optional[np.ndarray] = None,
                                  params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Segment soft tissues using a pre-trained deep learning model.
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters (model_path, region)
    
    Returns:
        Soft tissue segmentation mask with labels for different tissue types
    """
    # Default parameters
    if params is None:
        params = {}
    
    model_path = params.get('model_path', None)
    region = params.get('region', 'head')  # Anatomical region (head, pelvis, thorax)
    
    if model_path is None:
        logger.error("No model path provided for model-based tissue segmentation.")
        logger.warning("Falling back to multithreshold segmentation.")
        return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
    
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
            # Assuming the output has multiple channels, one for each tissue class
            n_classes = prediction.shape[-1]
            
            # Get the class with the highest probability for each pixel
            tissue_classes = np.argmax(prediction[0], axis=-1) + 1  # Start from 1
            
            # Resize back to original shape
            from skimage.transform import resize
            if len(orig_shape) == 3:
                # Create 3D mask with zeros
                segmentation = np.zeros(orig_shape, dtype=np.uint8)
                
                # Fill the middle slice with the prediction
                resized_slice = resize(
                    tissue_classes, 
                    (orig_shape[1], orig_shape[2]), 
                    order=0,  # Nearest neighbor to preserve class values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
                
                segmentation[middle_slice] = resized_slice
            else:
                segmentation = resize(
                    tissue_classes, 
                    orig_shape, 
                    order=0,  # Nearest neighbor to preserve class values
                    preserve_range=True,
                    mode='constant'
                ).astype(np.uint8)
        else:
            # For other regions or full 3D models
            # Implement specific logic based on model requirements
            logger.warning(f"Model-based tissue segmentation for region '{region}' not fully implemented.")
            logger.warning("Falling back to multithreshold segmentation.")
            return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
        
        # Include bone and air masks if provided
        max_label = np.max(segmentation)
        if bone_mask is not None:
            segmentation[bone_mask > 0] = max_label + 1
        if air_mask is not None:
            segmentation[air_mask > 0] = max_label + 2
        
        # Post-processing to clean up the segmentation
        min_size = params.get('min_size', 100)
        for label in range(1, max_label + 3):  # Include bone and air labels
            binary_mask = segmentation == label
            if np.any(binary_mask):
                cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
                # Remove small objects
                segmentation[binary_mask & ~cleaned_mask] = 0
        
        return segmentation
    
    except Exception as e:
        logger.error(f"Model-based tissue segmentation failed: {str(e)}")
        logger.warning("Falling back to multithreshold segmentation.")
        return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params) 

def segment_brain_tissues(image: np.ndarray,
                         bone_mask: Optional[np.ndarray] = None,
                         air_mask: Optional[np.ndarray] = None,
                         params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Specialized method for segmenting brain tissues in MRI.
    
    This function segments brain tissues into classes such as:
    - Gray matter (GM)
    - White matter (WM)
    - Cerebrospinal fluid (CSF)
    - Other soft tissues
    
    Args:
        image: Input MRI image as numpy array
        bone_mask: Optional bone mask to exclude bone regions
        air_mask: Optional air mask to exclude air regions
        params: Additional parameters
    
    Returns:
        Brain tissue segmentation mask with labels for different tissue types
    """
    # Default parameters
    if params is None:
        params = {}
    
    # Create a mask of regions to be segmented (exclude bone and air)
    mask = np.ones_like(image, dtype=bool)
    if bone_mask is not None:
        mask[bone_mask > 0] = False
    if air_mask is not None:
        mask[air_mask > 0] = False
    
    n_classes = params.get('n_classes', 4)  # Default: 4 classes (CSF, GM, WM, other)
    
    # Get intensity statistics for initial classification
    masked_intensities = image[mask]
    min_intensity = np.min(masked_intensities)
    max_intensity = np.max(masked_intensities)
    
    try:
        # Try using FSL's FAST-like approach with Gaussian Mixture Models
        try:
            from sklearn.mixture import GaussianMixture
            
            # Extract features for clustering (just intensity for now)
            X = masked_intensities.reshape(-1, 1)
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(
                n_components=n_classes,
                covariance_type='full',
                max_iter=params.get('max_iter', 100),
                random_state=params.get('random_state', 42)
            )
            gmm.fit(X)
            
            # Get tissue labels
            flat_labels = gmm.predict(X)
            
            # Create output segmentation
            segmentation = np.zeros_like(image, dtype=np.uint8)
            segmentation[mask] = flat_labels + 1  # Add 1 to avoid 0 label (reserved for background)
            
            # Sort labels by mean intensity
            means = gmm.means_.flatten()
            sorted_indices = np.argsort(means)
            
            # Remap labels based on intensity (typically: CSF < GM < WM)
            label_mapping = {old_label + 1: new_label + 1 for new_label, old_label in enumerate(sorted_indices)}
            remapped_segmentation = np.zeros_like(segmentation, dtype=np.uint8)
            
            for old_label, new_label in label_mapping.items():
                remapped_segmentation[segmentation == old_label] = new_label
            
            segmentation = remapped_segmentation
            
            # Standard labels:
            # 1: CSF (lowest intensity)
            # 2: Gray Matter
            # 3: White Matter
            # 4: Other soft tissues (if present)
            
        except ImportError:
            # Fall back to multi-threshold if sklearn is not available
            logger.warning("sklearn not available, falling back to multi-threshold segmentation")
            return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)
        
        # Post-processing to clean up the segmentation
        # 1. Remove small isolated regions
        min_size = params.get('min_size', 100)
        for label in range(1, n_classes + 1):
            binary_mask = segmentation == label
            cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
            # Replace small regions with the nearest label
            diff_mask = binary_mask & ~cleaned_mask
            if np.any(diff_mask):
                # Dilate other labels to fill in the gaps
                temp_mask = np.zeros_like(segmentation, dtype=bool)
                for other_label in range(1, n_classes + 1):
                    if other_label != label:
                        temp_mask |= segmentation == other_label
                dilated = morphology.binary_dilation(temp_mask)
                # Update the small removed regions
                segmentation[diff_mask & dilated] = 0
        
        # 2. Apply morphological operations to smooth boundaries
        smoothing_method = params.get('smoothing', 'median')
        if smoothing_method == 'median':
            # Apply median filter to each label separately
            for label in range(1, n_classes + 1):
                label_mask = segmentation == label
                if np.any(label_mask):
                    smoothed = ndimage.median_filter(label_mask.astype(np.float32), 
                                                  size=params.get('median_size', 3))
                    smoothed_binary = smoothed > 0.5
                    # Only update where no other label exists
                    update_mask = smoothed_binary & (segmentation == 0)
                    segmentation[update_mask] = label
        
        # 3. Fill holes in each tissue class
        for label in range(1, n_classes + 1):
            label_mask = segmentation == label
            if np.any(label_mask):
                filled = ndimage.binary_fill_holes(label_mask)
                # Only update where no other label exists
                update_mask = filled & (segmentation == 0)
                segmentation[update_mask] = label
        
        # 4. Add bone and air masks back
        if bone_mask is not None:
            segmentation[bone_mask > 0] = n_classes + 1
        if air_mask is not None:
            segmentation[air_mask > 0] = n_classes + 2
        
        return segmentation
        
    except Exception as e:
        logger.error(f"Brain tissue segmentation failed: {str(e)}")
        logger.warning("Falling back to multi-threshold segmentation.")
        return multithreshold_tissue_segmentation(image, bone_mask, air_mask, params)