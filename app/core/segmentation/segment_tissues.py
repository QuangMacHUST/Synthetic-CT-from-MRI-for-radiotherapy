#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Tissue segmentation module for MRI to CT conversion.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import load_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


class DeepLearningSegmentation:
    """
    Deep learning-based segmentation using neural networks.
    """
    
    def __init__(self, region: str = 'head'):
        """
        Initialize deep learning segmentation model.
        
        Args:
            region: Anatomical region ('head', 'pelvis', or 'thorax')
        """
        self.region = region
        
        # Get segmentation configuration
        seg_config = config.get_segmentation_params(region)
        dl_config = seg_config.get('deep_learning', {})
        
        # Get model path
        self.model_path = seg_config.get('model_path')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"Segmentation model not found at {self.model_path}")
            
            # Try default location
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_path = os.path.join(root_dir, "models", "segmentation", f"{region}_segmentation_model.h5")
            
            if os.path.exists(default_path):
                logger.info(f"Using default model at {default_path}")
                self.model_path = default_path
            else:
                logger.error(f"Default model not found at {default_path}")
                raise FileNotFoundError(f"Segmentation model not found for {region} region")
        
        # Get other parameters
        self.input_shape = dl_config.get('input_shape', [256, 256, 96])
        self.batch_size = dl_config.get('batch_size', 1)
        self.use_patch = dl_config.get('use_patch', True)
        self.patch_size = dl_config.get('patch_size', [128, 128, 32])
        self.overlap = dl_config.get('overlap', 0.5)
        self.threshold = dl_config.get('threshold', 0.5)
        
        # Load model
        self.model = self._load_model()
        
        # Get labels
        self.labels = seg_config.get('labels', {})
    
    def _load_model(self) -> tf.keras.Model:
        """
        Load deep learning model for segmentation.
        
        Returns:
            TensorFlow model
        """
        try:
            logger.info(f"Loading segmentation model from {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logger.info("Segmentation model loaded successfully")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {str(e)}")
            raise
    
    def _preprocess_input(self, image: sitk.Image) -> np.ndarray:
        """
        Preprocess input image for the neural network.
        
        Args:
            image: Input SimpleITK image
            
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert to numpy array
        array = sitk.GetArrayFromImage(image)
        
        # Normalize to [0, 1]
        array = array.astype(np.float32)
        min_val = np.min(array)
        max_val = np.max(array)
        
        if max_val > min_val:
            array = (array - min_val) / (max_val - min_val)
        
        # Add batch and channel dimensions if needed
        if len(array.shape) == 3:
            array = np.expand_dims(array, axis=(0, -1))  # [1, D, H, W, 1]
        
        return array
    
    def _extract_patches(self, array: np.ndarray) -> Tuple[List[np.ndarray], Tuple[int, int, int]]:
        """
        Extract patches from the input array for patch-based processing.
        
        Args:
            array: Input numpy array
            
        Returns:
            List of patches and original dimensions
        """
        # Get original dimensions (excluding batch and channel dims)
        orig_dims = array.shape[1:4]
        
        # Calculate patch step size (stride)
        stride = [int(self.patch_size[i] * (1 - self.overlap)) for i in range(3)]
        
        # Generate patch indices
        indices = []
        for dim in range(3):
            dim_indices = list(range(0, orig_dims[dim] - self.patch_size[dim] + 1, stride[dim]))
            
            # Add last patch if needed
            if dim_indices[-1] + self.patch_size[dim] < orig_dims[dim]:
                dim_indices.append(orig_dims[dim] - self.patch_size[dim])
            
            indices.append(dim_indices)
        
        # Extract patches
        patches = []
        for z in indices[0]:
            for y in indices[1]:
                for x in indices[2]:
                    patch = array[0,
                                 z:z+self.patch_size[0],
                                 y:y+self.patch_size[1],
                                 x:x+self.patch_size[2],
                                 0]
                    patches.append(np.expand_dims(patch, axis=(0, -1)))  # [1, patch_D, patch_H, patch_W, 1]
        
        return patches, orig_dims
    
    def _merge_patches(self, patches: List[np.ndarray], orig_dims: Tuple[int, int, int]) -> np.ndarray:
        """
        Merge predicted patches back into a full volume.
        
        Args:
            patches: List of predicted patches
            orig_dims: Original dimensions of the input volume
            
        Returns:
            Merged full-volume prediction
        """
        # Initialize output array and weight array
        output = np.zeros(orig_dims + (patches[0].shape[-1],), dtype=np.float32)  # [D, H, W, n_classes]
        weights = np.zeros(orig_dims, dtype=np.float32)  # [D, H, W]
        
        # Calculate stride
        stride = [int(self.patch_size[i] * (1 - self.overlap)) for i in range(3)]
        
        # Generate patch indices
        indices = []
        for dim in range(3):
            dim_indices = list(range(0, orig_dims[dim] - self.patch_size[dim] + 1, stride[dim]))
            
            # Add last patch if needed
            if dim_indices[-1] + self.patch_size[dim] < orig_dims[dim]:
                dim_indices.append(orig_dims[dim] - self.patch_size[dim])
            
            indices.append(dim_indices)
        
        # Merge patches
        patch_idx = 0
        for z in indices[0]:
            for y in indices[1]:
                for x in indices[2]:
                    # Get patch prediction
                    patch = patches[patch_idx][0, :, :, :, :]  # [patch_D, patch_H, patch_W, n_classes]
                    
                    # Add to output
                    output[z:z+self.patch_size[0],
                          y:y+self.patch_size[1],
                          x:x+self.patch_size[2], :] += patch
                    
                    # Update weights
                    weights[z:z+self.patch_size[0],
                           y:y+self.patch_size[1],
                           x:x+self.patch_size[2]] += 1
                    
                    patch_idx += 1
        
        # Average predictions by weights
        for c in range(output.shape[-1]):
            output[:, :, :, c] /= np.maximum(weights, 1)
        
        return output
    
    def _postprocess_output(self, pred_array: np.ndarray, original_image: sitk.Image) -> sitk.Image:
        """
        Postprocess model output and convert to SimpleITK image.
        
        Args:
            pred_array: Model prediction as numpy array
            original_image: Original input image for metadata
            
        Returns:
            Segmentation mask as SimpleITK image
        """
        # Get final segmentation mask from probabilities
        if pred_array.shape[-1] > 1:
            # Multi-class segmentation - get argmax
            seg_array = np.argmax(pred_array, axis=-1).astype(np.uint8)
        else:
            # Binary segmentation - apply threshold
            seg_array = (pred_array[:, :, :, 0] > self.threshold).astype(np.uint8)
        
        # Create SimpleITK image from array
        seg_mask = sitk.GetImageFromArray(seg_array)
        seg_mask.CopyInformation(original_image)
        
        return seg_mask
    
    def segment(self, image: sitk.Image) -> sitk.Image:
        """
        Perform deep learning-based segmentation.
        
        Args:
            image: Input SimpleITK image
            
        Returns:
            Segmentation mask as SimpleITK image
        """
        logger.info(f"Performing deep learning-based segmentation for {self.region} region")
        
        try:
            # Preprocess input
            preprocessed = self._preprocess_input(image)
            
            # Check if patch-based processing is needed
            if self.use_patch and any(preprocessed.shape[i+1] > self.patch_size[i] for i in range(3)):
                logger.info("Using patch-based segmentation")
                
                # Extract patches
                patches, orig_dims = self._extract_patches(preprocessed)
                
                # Predict on each patch
                predicted_patches = []
                for i, patch in enumerate(patches):
                    logger.debug(f"Processing patch {i+1}/{len(patches)}")
                    pred = self.model.predict(patch, batch_size=self.batch_size, verbose=0)
                    predicted_patches.append(pred)
                
                # Merge patches
                prediction = self._merge_patches(predicted_patches, orig_dims)
            
            else:
                # Perform full-image segmentation
                logger.info("Using full-image segmentation")
                
                # Resize if needed
                if any(preprocessed.shape[i+1] != self.input_shape[i] for i in range(3)):
                    logger.info(f"Resizing input from {preprocessed.shape[1:4]} to {self.input_shape}")
                    
                    # Create temporary SimpleITK image
                    temp_image = sitk.GetImageFromArray(preprocessed[0, :, :, :, 0])
                    temp_image.CopyInformation(image)
                    
                    # Resample to input shape
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetSize(self.input_shape)
                    resampler.SetInterpolator(sitk.sitkLinear)
                    resampled = resampler.Execute(temp_image)
                    
                    # Convert back to numpy array
                    preprocessed = np.expand_dims(sitk.GetArrayFromImage(resampled), axis=(0, -1))
                
                # Predict segmentation
                prediction = self.model.predict(preprocessed, batch_size=self.batch_size, verbose=0)
                
                # Resize back to original dimensions if needed
                if any(prediction.shape[i+1] != image.GetSize()[i] for i in range(3)):
                    logger.info("Resizing prediction back to original dimensions")
                    
                    # Create temporary SimpleITK images for each class
                    orig_size = image.GetSize()
                    resampled_predictions = []
                    
                    for c in range(prediction.shape[-1]):
                        # Create SimpleITK image for this class
                        class_image = sitk.GetImageFromArray(prediction[0, :, :, :, c])
                        class_image.CopyInformation(temp_image)
                        
                        # Resample to original size
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetSize(orig_size)
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampled = resampler.Execute(class_image)
                        
                        # Convert back to numpy array
                        resampled_predictions.append(sitk.GetArrayFromImage(resampled))
                    
                    # Stack resampled predictions
                    prediction = np.stack(resampled_predictions, axis=-1)
                    prediction = np.expand_dims(prediction, axis=0)
            
            # Postprocess and convert back to SimpleITK
            seg_mask = self._postprocess_output(prediction[0], image)
            
            logger.info("Deep learning-based segmentation completed")
            return seg_mask
        
        except Exception as e:
            logger.error(f"Error in deep learning-based segmentation: {str(e)}")
            raise


class AtlasBasedSegmentation:
    """
    Atlas-based segmentation using image registration.
    """
    
    def __init__(self, region: str = 'head'):
        """
        Initialize atlas-based segmentation.
        
        Args:
            region: Anatomical region ('head', 'pelvis', or 'thorax')
        """
        self.region = region
        
        # Get segmentation configuration
        seg_config = config.get_segmentation_params(region)
        atlas_config = seg_config.get('atlas', {})
        
        # Get atlas path based on region
        atlas_path_key = f"{region}_atlas_path"
        self.atlas_path = atlas_config.get(atlas_path_key)
        
        # Check if atlas exists
        if not self.atlas_path or not os.path.exists(self.atlas_path):
            logger.warning(f"Atlas not found at {self.atlas_path}")
            
            # Try default location
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_path = os.path.join(root_dir, "models", "atlas", f"{region}_segmentation_atlas.h5")
            
            if os.path.exists(default_path):
                logger.info(f"Using default atlas at {default_path}")
                self.atlas_path = default_path
            else:
                logger.error(f"Default atlas not found at {default_path}")
                raise FileNotFoundError(f"Segmentation atlas not found for {region} region")
        
        # Get registration parameters
        self.registration_method = atlas_config.get('registration_method', 'affine')
        self.label_fusion = atlas_config.get('label_fusion', 'majority')
        
        # Load atlas
        self.atlas_images, self.atlas_labels = self._load_atlas()
        
        # Get labels
        self.labels = seg_config.get('labels', {})
    
    def _load_atlas(self) -> Tuple[List[sitk.Image], List[sitk.Image]]:
        """
        Load atlas images and labels.
        
        Returns:
            Tuple of (atlas_images, atlas_labels)
        """
        try:
            logger.info(f"Loading segmentation atlas from {self.atlas_path}")
            
            # In a real implementation, this would load actual atlas data
            # For now, create placeholder data
            atlas_images = [sitk.Image(64, 64, 64, sitk.sitkFloat32) for _ in range(3)]
            atlas_labels = [sitk.Image(64, 64, 64, sitk.sitkUInt8) for _ in range(3)]
            
            logger.info(f"Loaded {len(atlas_images)} atlas samples")
            return atlas_images, atlas_labels
        
        except Exception as e:
            logger.error(f"Failed to load segmentation atlas: {str(e)}")
            raise
    
    def _register_atlas(self, image: sitk.Image, atlas_image: sitk.Image) -> sitk.Transform:
        """
        Register atlas image to input image.
        
        Args:
            image: Input SimpleITK image
            atlas_image: Atlas SimpleITK image
            
        Returns:
            SimpleITK transform
        """
        logger.info(f"Registering atlas using {self.registration_method} registration")
        
        # Initialize registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)
        
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
        
        # Initialize transform based on registration method
        if self.registration_method == 'rigid':
            initial_transform = sitk.CenteredTransformInitializer(
                image, 
                atlas_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.registration_method == 'affine':
            initial_transform = sitk.CenteredTransformInitializer(
                image, 
                atlas_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.registration_method == 'deformable':
            # For deformable registration, first do affine registration
            initial_transform = sitk.CenteredTransformInitializer(
                image, 
                atlas_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # Execute affine registration
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            affine_transform = registration_method.Execute(image, atlas_image)
            
            # Then do BSpline registration
            mesh_size = [10] * image.GetDimension()
            final_transform = sitk.BSplineTransformInitializer(
                image, 
                mesh_size
            )
            registration_method.SetInitialTransform(final_transform, inPlace=True)
            bspline_transform = registration_method.Execute(image, atlas_image)
            
            # Return composite transform
            composite_transform = sitk.CompositeTransform([affine_transform, bspline_transform])
            return composite_transform
        else:
            # Default to affine
            initial_transform = sitk.CenteredTransformInitializer(
                image, 
                atlas_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        
        # Execute registration for non-deformable methods
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        final_transform = registration_method.Execute(image, atlas_image)
        
        return final_transform
    
    def _apply_transform(self, transform: sitk.Transform, label: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
        """
        Apply transform to atlas label image.
        
        Args:
            transform: SimpleITK transform
            label: Atlas label image
            reference_image: Reference image for output properties
            
        Returns:
            Transformed label image
        """
        # Use nearest neighbor interpolation for label images
        transformed_label = sitk.Resample(
            label, 
            reference_image, 
            transform, 
            sitk.sitkNearestNeighbor,
            0,  # Default pixel value
            label.GetPixelID()
        )
        
        return transformed_label
    
    def _fuse_labels(self, transformed_labels: List[sitk.Image]) -> sitk.Image:
        """
        Fuse multiple transformed labels.
        
        Args:
            transformed_labels: List of transformed label images
            
        Returns:
            Fused label image
        """
        if self.label_fusion == 'majority':
            # Majority voting - most common label at each voxel
            logger.info("Fusing labels using majority voting")
            
            # Convert to numpy arrays
            arrays = [sitk.GetArrayFromImage(label) for label in transformed_labels]
            
            # Get unique labels across all atlases
            unique_labels = np.unique(np.concatenate([np.unique(array) for array in arrays]))
            
            # Count votes for each label
            shape = arrays[0].shape
            votes = np.zeros((len(unique_labels),) + shape, dtype=np.int32)
            
            for i, label in enumerate(unique_labels):
                for array in arrays:
                    votes[i] += (array == label)
            
            # Get winning label
            winner_indices = np.argmax(votes, axis=0)
            fused_array = np.zeros(shape, dtype=np.uint8)
            
            for i, label in enumerate(unique_labels):
                fused_array[winner_indices == i] = label
            
            # Convert back to SimpleITK
            fused_label = sitk.GetImageFromArray(fused_array)
            fused_label.CopyInformation(transformed_labels[0])
            
            return fused_label
        
        elif self.label_fusion == 'staple':
            # STAPLE algorithm - Simultaneous Truth and Performance Level Estimation
            logger.info("Fusing labels using STAPLE algorithm")
            
            # Use SimpleITK's STAPLE implementation
            staple_filter = sitk.STAPLEImageFilter()
            fused_label = staple_filter.Execute(transformed_labels)
            
            return fused_label
        
        else:
            # Default to using the first transformed label
            logger.warning(f"Unknown label fusion method: {self.label_fusion}. Using first label.")
            return transformed_labels[0]
    
    def segment(self, image: sitk.Image) -> sitk.Image:
        """
        Perform atlas-based segmentation.
        
        Args:
            image: Input SimpleITK image
            
        Returns:
            Segmentation mask as SimpleITK image
        """
        logger.info(f"Performing atlas-based segmentation for {self.region} region")
        
        try:
            # Register atlas to input image
            transformed_labels = []
            
            for i, (atlas_image, atlas_label) in enumerate(zip(self.atlas_images, self.atlas_labels)):
                logger.info(f"Registering atlas {i+1}/{len(self.atlas_images)}")
                
                # Register atlas to input image
                transform = self._register_atlas(image, atlas_image)
                
                # Apply transform to label
                transformed_label = self._apply_transform(transform, atlas_label, image)
                
                transformed_labels.append(transformed_label)
            
            # Fuse labels
            seg_mask = self._fuse_labels(transformed_labels)
            
            logger.info("Atlas-based segmentation completed")
            return seg_mask
        
        except Exception as e:
            logger.error(f"Error in atlas-based segmentation: {str(e)}")
            raise


def segment_tissues(mri_image: Union[sitk.Image, SyntheticCT, str],
                  method: str = 'auto',
                  region: str = 'head') -> sitk.Image:
    """
    Segment tissues in MRI image.
    
    Args:
        mri_image: Input MRI image as SimpleITK image, SyntheticCT object, or path to file
        method: Segmentation method ('auto', 'deep_learning', or 'atlas')
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        
    Returns:
        Segmentation mask as SimpleITK image
    """
    logger.info(f"Starting tissue segmentation using {method} method for {region} region")
    
    # Convert input to SyntheticCT if needed
    if isinstance(mri_image, str):
        # Load image from file
        from app.utils.io_utils import load_medical_image
        mri_image = load_medical_image(mri_image)
    
    if isinstance(mri_image, SyntheticCT):
        input_image = mri_image.image
    else:
        input_image = mri_image
    
    # Get segmentation configuration
    seg_config = config.get_segmentation_params(region)
    
    # Determine segmentation method
    if method == 'auto':
        # Automatically select method based on availability
        if os.path.exists(seg_config.get('model_path', '')):
            method = 'deep_learning'
            logger.info("Automatically selected deep learning segmentation method")
        else:
            atlas_path_key = f"{region}_atlas_path"
            atlas_path = seg_config.get('atlas', {}).get(atlas_path_key, '')
            
            if os.path.exists(atlas_path):
                method = 'atlas'
                logger.info("Automatically selected atlas-based segmentation method")
            else:
                logger.warning("No suitable segmentation method found. Defaulting to deep learning.")
                method = 'deep_learning'
    
    # Perform segmentation
    try:
        if method == 'deep_learning':
            # Deep learning-based segmentation
            segmenter = DeepLearningSegmentation(region)
            seg_mask = segmenter.segment(input_image)
        
        elif method == 'atlas':
            # Atlas-based segmentation
            segmenter = AtlasBasedSegmentation(region)
            seg_mask = segmenter.segment(input_image)
        
        else:
            logger.warning(f"Unknown segmentation method: {method}. Using deep learning.")
            segmenter = DeepLearningSegmentation(region)
            seg_mask = segmenter.segment(input_image)
        
        logger.info(f"Tissue segmentation completed using {method} method")
        return seg_mask
    
    except Exception as e:
        logger.error(f"Error in tissue segmentation: {str(e)}")
        raise 