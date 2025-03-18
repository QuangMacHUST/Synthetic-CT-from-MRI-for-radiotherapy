#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
MRI to CT conversion module.
This module provides methods to convert MRI images to synthetic CT images
using different approaches: atlas-based, CNN-based, and GAN-based.
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


class AtlasBasedConverter:
    """
    Atlas-based MRI to CT conversion using image registration
    and atlas database.
    """
    
    def __init__(self, region='head'):
        """
        Initialize atlas-based converter.
        
        Args:
            region: Anatomical region ('head', 'pelvis', or 'thorax')
        """
        self.region = region
        
        # Get conversion configuration
        conv_config = config.get_conversion_params('atlas', region)
        
        # Get atlas path
        self.atlas_path = conv_config.get('atlas_path')
        
        # Check if atlas exists
        if not os.path.exists(self.atlas_path):
            logger.warning(f"Atlas not found at {self.atlas_path}")
            
            # Try default location
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_path = os.path.join(root_dir, "models", "atlas", f"{region}_ct_atlas.h5")
            
            if os.path.exists(default_path):
                logger.info(f"Using default atlas at {default_path}")
                self.atlas_path = default_path
            else:
                logger.error(f"Default atlas not found at {default_path}")
                raise FileNotFoundError(f"CT atlas not found for {region} region")
        
        # Get registration parameters
        reg_params = conv_config.get('registration_params', {})
        self.registration_method = reg_params.get('method', 'deformable')
        self.metric = reg_params.get('metric', 'mutual_information')
        self.sampling_percentage = reg_params.get('sampling_percentage', 0.1)
        
        # Load atlas
        self.atlas_mri_images, self.atlas_ct_images = self._load_atlas()
        
        # Get HU values for different tissues
        self.hu_values = conv_config.get('hu_values', {})
    
    def _load_atlas(self):
        """
        Load atlas MRI and CT images.
        
        Returns:
            Tuple of (atlas_mri_images, atlas_ct_images)
        """
        try:
            logger.info(f"Loading CT atlas from {self.atlas_path}")
            
            # In a real implementation, this would load the atlas data
            # For now, create placeholder data
            atlas_mri_images = [sitk.Image(64, 64, 64, sitk.sitkFloat32) for _ in range(3)]
            atlas_ct_images = [sitk.Image(64, 64, 64, sitk.sitkInt16) for _ in range(3)]
            
            logger.info(f"Loaded {len(atlas_mri_images)} atlas samples")
            return atlas_mri_images, atlas_ct_images
        
        except Exception as e:
            logger.error(f"Failed to load CT atlas: {str(e)}")
            raise
    
    def _register_images(self, fixed_image, moving_image):
        """
        Register moving image to fixed image.
        
        Args:
            fixed_image: Fixed SimpleITK image (MRI)
            moving_image: Moving SimpleITK image (Atlas MRI)
            
        Returns:
            SimpleITK transform
        """
        logger.info(f"Registering atlas using {self.registration_method} registration")
        
        # Initialize registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set similarity metric based on configuration
        if self.metric == 'mutual_information':
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elif self.metric == 'mean_squares':
            registration_method.SetMetricAsMeanSquares()
        elif self.metric == 'correlation':
            registration_method.SetMetricAsCorrelation()
        else:
            logger.warning(f"Unknown metric: {self.metric}. Using mutual information.")
            registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        
        # Set sampling strategy
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(self.sampling_percentage)
        
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
        
        # Set initial transform based on registration method
        if self.registration_method == 'rigid':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.registration_method == 'affine':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif self.registration_method == 'deformable':
            # For deformable registration, first do affine registration
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            
            # Execute affine registration
            registration_method.SetInitialTransform(initial_transform, inPlace=True)
            affine_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Then do BSpline registration
            mesh_size = [10] * 3
            final_transform = sitk.BSplineTransformInitializer(
                fixed_image, 
                mesh_size
            )
            registration_method.SetInitialTransform(final_transform, inPlace=True)
            bspline_transform = registration_method.Execute(fixed_image, moving_image)
            
            # Return composite transform
            composite_transform = sitk.CompositeTransform([affine_transform, bspline_transform])
            return composite_transform
        else:
            # Default to affine
            logger.warning(f"Unknown registration method: {self.registration_method}. Using affine.")
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_image, 
                moving_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        
        # Execute registration for non-deformable methods
        registration_method.SetInitialTransform(initial_transform, inPlace=True)
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        return final_transform
    
    def convert(self, mri_image, segmentation):
        """
        Convert MRI to CT using atlas-based approach.
        
        Args:
            mri_image: Input MRI as SimpleITK image
            segmentation: Segmentation mask as SimpleITK image
            
        Returns:
            Synthetic CT as SyntheticCT object
        """
        logger.info(f"Starting atlas-based MRI to CT conversion for {self.region} region")
        
        try:
            # Convert input to SimpleITK image if needed
            if isinstance(mri_image, SyntheticCT):
                input_image = mri_image.image
                metadata = mri_image.metadata.copy()
            else:
                input_image = mri_image
                metadata = {}
            
            # Initialize synthetic CT with same dimensions as input
            synthetic_ct_array = np.zeros(sitk.GetArrayFromImage(input_image).shape, dtype=np.int16)
            
            # First approach: Use direct mapping from segmentation to HU values
            if segmentation is not None:
                logger.info("Using segmentation-based approach")
                
                segmentation_array = sitk.GetArrayFromImage(segmentation)
                
                # Map each segmentation label to corresponding HU value
                for tissue, hu_range in self.hu_values.items():
                    # Get label value for this tissue
                    seg_config = config.get_segmentation_params(self.region)
                    label = seg_config.get('labels', {}).get(tissue)
                    
                    if label is not None:
                        # Calculate mean HU value for this tissue
                        mean_hu = sum(hu_range) / len(hu_range) if isinstance(hu_range, list) else hu_range
                        
                        # Assign HU value to all voxels with this label
                        synthetic_ct_array[segmentation_array == label] = mean_hu
            
            # Second approach: Use deformable registration to transform atlas CT to input MRI
            logger.info("Using registration-based approach")
            
            transformed_ct_images = []
            
            for i, (atlas_mri, atlas_ct) in enumerate(zip(self.atlas_mri_images, self.atlas_ct_images)):
                logger.info(f"Registering atlas {i+1}/{len(self.atlas_mri_images)}")
                
                # Register atlas MRI to input MRI
                transform = self._register_images(input_image, atlas_mri)
                
                # Apply transform to atlas CT
                transformed_ct = sitk.Resample(
                    atlas_ct, 
                    input_image, 
                    transform, 
                    sitk.sitkLinear,
                    -1024,  # Default value (air)
                    atlas_ct.GetPixelID()
                )
                
                transformed_ct_images.append(transformed_ct)
            
            # Fuse transformed CTs
            if transformed_ct_images:
                logger.info("Fusing transformed CT images")
                
                # Average all transformed CTs
                fused_ct_array = np.mean([sitk.GetArrayFromImage(ct) for ct in transformed_ct_images], axis=0)
                
                # If segmentation is available, refine using segmentation
                if segmentation is not None:
                    logger.info("Refining with segmentation")
                    
                    segmentation_array = sitk.GetArrayFromImage(segmentation)
                    
                    # For each tissue type
                    for tissue, hu_range in self.hu_values.items():
                        # Get label value for this tissue
                        seg_config = config.get_segmentation_params(self.region)
                        label = seg_config.get('labels', {}).get(tissue)
                        
                        if label is not None:
                            # Get mask for this tissue
                            tissue_mask = segmentation_array == label
                            
                            if np.any(tissue_mask):
                                # Calculate min and max HU
                                min_hu, max_hu = hu_range if isinstance(hu_range, list) else (hu_range, hu_range)
                                
                                # Clip HU values for this tissue
                                fused_ct_array[tissue_mask] = np.clip(
                                    fused_ct_array[tissue_mask], 
                                    min_hu, 
                                    max_hu
                                )
                
                # Use fused CT array
                synthetic_ct_array = fused_ct_array
            
            # Convert array to SimpleITK image
            synthetic_ct_image = sitk.GetImageFromArray(synthetic_ct_array.astype(np.int16))
            synthetic_ct_image.CopyInformation(input_image)
            
            # Create metadata
            conversion_metadata = {
                'method': 'atlas',
                'region': self.region,
                'atlas_path': self.atlas_path,
                'registration_method': self.registration_method
            }
            
            if 'conversion' not in metadata:
                metadata['conversion'] = conversion_metadata
            else:
                metadata['conversion'].update(conversion_metadata)
            
            # Create SyntheticCT object
            synthetic_ct = SyntheticCT(synthetic_ct_image, metadata)
            
            logger.info("Atlas-based MRI to CT conversion completed")
            return synthetic_ct
        
        except Exception as e:
            logger.error(f"Error in atlas-based conversion: {str(e)}")
            raise


class CNNConverter:
    """
    CNN-based MRI to CT conversion using convolutional neural networks.
    """
    
    def __init__(self, model_path=None, region='head'):
        """
        Initialize CNN converter.
        
        Args:
            model_path: Path to CNN model (optional)
            region: Anatomical region ('head', 'pelvis', or 'thorax')
        """
        self.region = region
        
        # Get conversion configuration
        conv_config = config.get_conversion_params('cnn', region)
        
        # Get model path
        self.model_path = model_path or conv_config.get('model_path')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"CNN model not found at {self.model_path}")
            
            # Try default location
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_path = os.path.join(root_dir, "models", "cnn", f"{region}_cnn_model.h5")
            
            if os.path.exists(default_path):
                logger.info(f"Using default model at {default_path}")
                self.model_path = default_path
            else:
                logger.error(f"Default model not found at {default_path}")
                raise FileNotFoundError(f"CNN model not found for {region} region")
        
        # Get other parameters
        self.patch_size = conv_config.get('patch_size', [64, 64, 64])
        self.stride = conv_config.get('stride', [32, 32, 32])
        self.batch_size = conv_config.get('batch_size', 4)
        self.normalization = conv_config.get('normalization', 'z-score')
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load CNN model.
        
        Returns:
            TensorFlow model
        """
        try:
            logger.info(f"Loading CNN model from {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logger.info("CNN model loaded successfully")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load CNN model: {str(e)}")
            raise
    
    def convert(self, mri_image, segmentation=None):
        """
        Convert MRI to CT using CNN approach.
        
        Args:
            mri_image: Input MRI as SimpleITK image or SyntheticCT object
            segmentation: Segmentation mask as SimpleITK image (optional)
            
        Returns:
            Synthetic CT as SyntheticCT object
        """
        logger.info(f"Starting CNN-based MRI to CT conversion for {self.region} region")
        
        try:
            # Convert input to SimpleITK image if needed
            if isinstance(mri_image, SyntheticCT):
                input_image = mri_image.image
                metadata = mri_image.metadata.copy()
            else:
                input_image = mri_image
                metadata = {}
            
            # Get image as numpy array
            mri_array = sitk.GetArrayFromImage(input_image)
            
            # Normalize input
            if self.normalization == 'z-score':
                # Z-score normalization
                mask = mri_array > 0  # Non-zero mask
                if np.any(mask):
                    mean_val = np.mean(mri_array[mask])
                    std_val = np.std(mri_array[mask])
                    if std_val > 0:
                        mri_array = (mri_array - mean_val) / std_val
                        mri_array[~mask] = 0  # Keep background as 0
            
            elif self.normalization == 'minmax':
                # Min-max normalization
                mask = mri_array > 0  # Non-zero mask
                if np.any(mask):
                    min_val = np.min(mri_array[mask])
                    max_val = np.max(mri_array[mask])
                    if max_val > min_val:
                        mri_array = (mri_array - min_val) / (max_val - min_val)
                        mri_array[~mask] = 0  # Keep background as 0
            
            # Extract patches
            # NOTE: In a real implementation, we would extract patches and process them in batches
            # For simplicity, we'll use the entire volume here
            
            # Add batch and channel dimensions
            input_data = np.expand_dims(mri_array, axis=(0, -1)).astype(np.float32)
            
            # If segmentation is provided, include it as an additional channel
            if segmentation is not None:
                segmentation_array = sitk.GetArrayFromImage(segmentation)
                
                # One-hot encode the segmentation
                num_classes = np.max(segmentation_array) + 1
                one_hot_seg = np.eye(num_classes)[segmentation_array]
                
                # Add batch dimension and combine with MRI
                seg_data = np.expand_dims(one_hot_seg, axis=0)
                input_data = np.concatenate([input_data, seg_data], axis=-1)
            
            # Predict synthetic CT
            synthetic_ct_array = self.model.predict(input_data, batch_size=self.batch_size, verbose=0)
            
            # Remove batch dimension
            synthetic_ct_array = synthetic_ct_array[0, ..., 0]
            
            # Convert to HU units (assumes model output is in HU)
            synthetic_ct_array = synthetic_ct_array.astype(np.int16)
            
            # Create SimpleITK image
            synthetic_ct_image = sitk.GetImageFromArray(synthetic_ct_array)
            synthetic_ct_image.CopyInformation(input_image)
            
            # Create metadata
            conversion_metadata = {
                'method': 'cnn',
                'region': self.region,
                'model_path': self.model_path,
                'normalization': self.normalization
            }
            
            if 'conversion' not in metadata:
                metadata['conversion'] = conversion_metadata
            else:
                metadata['conversion'].update(conversion_metadata)
            
            # Create SyntheticCT object
            synthetic_ct = SyntheticCT(synthetic_ct_image, metadata)
            
            logger.info("CNN-based MRI to CT conversion completed")
            return synthetic_ct
        
        except Exception as e:
            logger.error(f"Error in CNN-based conversion: {str(e)}")
            raise


class GANConverter:
    """
    GAN-based MRI to CT conversion using deep generative models.
    Supports both 2D slice-by-slice and 3D volume approaches.
    """
    
    def __init__(self, region='head', use_3d=None, config=None):
        """
        Initialize GAN-based converter.
        
        Args:
            region: Anatomical region ('head', 'pelvis', or 'thorax')
            use_3d: Whether to use 3D GAN (True) or 2D slice-by-slice (False).
                    If None, use the default from config.
            config: Configuration object or None to use default
        """
        self.region = region
        
        # Get configuration
        self.config = config or load_config()
        gan_config = self.config.get('conversion', {}).get('gan', {}).get(region, {})
        
        # Determine whether to use 3D GAN or 2D slice-by-slice
        self.use_3d = use_3d if use_3d is not None else gan_config.get('use_3d', False)
        
        # Get paths for models
        self.generator_path = gan_config.get('generator_path')
        
        # Verify model exists
        if not os.path.exists(self.generator_path):
            raise ValueError(f"GAN generator model not found at: {self.generator_path}")
        
        # Get batch size
        self.batch_size = gan_config.get('batch_size', 1)
        
        # Get input shape
        if self.use_3d:
            self.input_shape = gan_config.get('patch_size', [256, 256, 32])
            self.stride = gan_config.get('stride', [128, 128, 16])
        else:
            self.input_shape = gan_config.get('input_shape', [256, 256, 1])
            
        # Load generator model
        logger.info(f"Loading GAN generator model from {self.generator_path}")
        self.generator = self._load_generator()
        
        # Whether to use multi-sequence MRI
        self.use_multi_sequence = gan_config.get('use_multi_sequence', False)
        self.sequence_names = gan_config.get('sequence_names', ['T1'])
        
    def _load_generator(self):
        """Load the pre-trained generator model."""
        try:
            import tensorflow as tf
            
            # Custom objects for loading the model if needed
            custom_objects = {
                # Define any custom layers or losses here if needed
            }
            
            # Load the model
            logger.info(f"Loading generator model from {self.generator_path}")
            generator = tf.keras.models.load_model(
                self.generator_path, 
                custom_objects=custom_objects,
                compile=False  # No need to compile for inference
            )
            
            return generator
            
        except ImportError:
            logger.error("TensorFlow not installed. Install tensorflow to use GAN models.")
            raise
        except Exception as e:
            logger.error(f"Error loading GAN generator model: {str(e)}")
            raise
    
    def convert(self, mri, segmentation=None):
        """
        Convert MRI to synthetic CT using GAN.
        
        Args:
            mri: Input MRI image (single sequence or MultiSequenceMRI)
            segmentation: Optional tissue segmentation (for conditioning)
            
        Returns:
            Synthetic CT image
        """
        if self.use_multi_sequence:
            return self._convert_multi_sequence(mri, segmentation)
        else:
            return self._convert_single_sequence(mri, segmentation)
    
    def _convert_single_sequence(self, mri, segmentation=None):
        """Convert single sequence MRI to synthetic CT."""
        # Convert to numpy array
        mri_data = sitk.GetArrayFromImage(mri).astype(np.float32)
        
        # Normalize intensity to [-1, 1] for GAN input
        mri_data = self._normalize_intensity(mri_data)
        
        # Prepare segmentation if available
        if segmentation is not None:
            seg_data = sitk.GetArrayFromImage(segmentation)
            # One-hot encode segmentation for conditioning
            seg_one_hot = self._one_hot_encode_segmentation(seg_data)
        else:
            seg_one_hot = None
        
        # Generate synthetic CT
        if self.use_3d:
            ct_data = self._generate_ct_3d(mri_data, seg_one_hot)
        else:
            ct_data = self._generate_ct_2d(mri_data, seg_one_hot)
        
        # Convert back to original intensity range (HU values)
        ct_data = self._denormalize_intensity(ct_data)
        
        # Create SimpleITK image with same metadata as input MRI
        ct_image = sitk.GetImageFromArray(ct_data)
        ct_image.SetOrigin(mri.GetOrigin())
        ct_image.SetSpacing(mri.GetSpacing())
        ct_image.SetDirection(mri.GetDirection())
        
        return ct_image
    
    def _convert_multi_sequence(self, multi_mri, segmentation=None):
        """Convert multi-sequence MRI to synthetic CT."""
        if not hasattr(multi_mri, 'get_sequence'):
            raise ValueError("Expected MultiSequenceMRI object for multi-sequence conversion")
        
        # Get available sequences from the input
        available_sequences = multi_mri.get_sequence_names()
        
        # Determine which sequences to use
        sequences_to_use = [seq for seq in self.sequence_names if seq in available_sequences]
        
        if not sequences_to_use:
            logger.warning(f"None of the required sequences {self.sequence_names} found in input. "
                          f"Available: {available_sequences}. Using first available sequence.")
            sequences_to_use = [available_sequences[0]]
        
        # Get data for each sequence
        mri_sequence_data = {}
        for seq_name in sequences_to_use:
            seq_image = multi_mri.get_sequence(seq_name)
            seq_data = sitk.GetArrayFromImage(seq_image).astype(np.float32)
            mri_sequence_data[seq_name] = self._normalize_intensity(seq_data)
        
        # Prepare segmentation if available
        if segmentation is not None:
            seg_data = sitk.GetArrayFromImage(segmentation)
            seg_one_hot = self._one_hot_encode_segmentation(seg_data)
        else:
            seg_one_hot = None
            
        # Reference image for metadata
        reference_image = multi_mri.get_reference()
        
        # Generate synthetic CT using multiple MRI sequences
        if self.use_3d:
            ct_data = self._generate_ct_3d_multi_sequence(mri_sequence_data, seg_one_hot)
        else:
            ct_data = self._generate_ct_2d_multi_sequence(mri_sequence_data, seg_one_hot)
            
        # Convert back to original intensity range (HU values)
        ct_data = self._denormalize_intensity(ct_data)
        
        # Create SimpleITK image with same metadata as input reference MRI
        ct_image = sitk.GetImageFromArray(ct_data)
        ct_image.SetOrigin(reference_image.GetOrigin())
        ct_image.SetSpacing(reference_image.GetSpacing())
        ct_image.SetDirection(reference_image.GetDirection())
        
        return ct_image
    
    def _generate_ct_2d(self, mri_data, segmentation=None):
        """Generate synthetic CT slice by slice using 2D GAN."""
        import tensorflow as tf
        
        # Get dimensions
        depth, height, width = mri_data.shape
        
        # Prepare output array
        ct_data = np.zeros_like(mri_data)
        
        # Process slices in batches
        for start_idx in range(0, depth, self.batch_size):
            end_idx = min(start_idx + self.batch_size, depth)
            batch_size = end_idx - start_idx
            
            # Extract slices for this batch
            batch_slices = mri_data[start_idx:end_idx]
            
            # Reshape for network input: (batch, height, width, 1)
            batch_input = batch_slices.reshape(batch_size, height, width, 1)
            
            # If segmentation is available, add as additional input channels
            if segmentation is not None:
                # Extract segmentation slices
                seg_slices = segmentation[start_idx:end_idx]
                
                # Concatenate MRI and segmentation along channel axis
                batch_input = tf.concat([batch_input, seg_slices], axis=-1)
            
            # Apply generator to convert MRI to CT
            logger.debug(f"Processing slices {start_idx}-{end_idx-1}")
            batch_output = self.generator.predict(batch_input)
            
            # If output has multiple channels, take the first one as CT
            if batch_output.shape[-1] > 1:
                batch_output = batch_output[..., 0]
                
            # Add to output array
            ct_data[start_idx:end_idx] = batch_output.reshape(batch_size, height, width)
        
        return ct_data
    
    def _generate_ct_2d_multi_sequence(self, mri_sequence_data, segmentation=None):
        """Generate synthetic CT from multiple MRI sequences slice by slice."""
        import tensorflow as tf
        
        # Get first sequence to determine dimensions
        first_seq = next(iter(mri_sequence_data.values()))
        depth, height, width = first_seq.shape
        
        # Prepare output array
        ct_data = np.zeros((depth, height, width), dtype=np.float32)
        
        # Process slices in batches
        for start_idx in range(0, depth, self.batch_size):
            end_idx = min(start_idx + self.batch_size, depth)
            batch_size = end_idx - start_idx
            
            # Extract slices from each sequence and concatenate along channel axis
            seq_channels = []
            for seq_data in mri_sequence_data.values():
                batch_slices = seq_data[start_idx:end_idx]
                seq_channels.append(batch_slices.reshape(batch_size, height, width, 1))
            
            # Concatenate all sequences
            batch_input = tf.concat(seq_channels, axis=-1)
            
            # If segmentation is available, add as additional input channels
            if segmentation is not None:
                # Extract segmentation slices
                seg_slices = segmentation[start_idx:end_idx]
                
                # Concatenate MRI sequences and segmentation along channel axis
                batch_input = tf.concat([batch_input, seg_slices], axis=-1)
            
            # Apply generator to convert MRI to CT
            logger.debug(f"Processing multi-sequence slices {start_idx}-{end_idx-1}")
            batch_output = self.generator.predict(batch_input)
            
            # If output has multiple channels, take the first one as CT
            if batch_output.shape[-1] > 1:
                batch_output = batch_output[..., 0]
                
            # Add to output array
            ct_data[start_idx:end_idx] = batch_output.reshape(batch_size, height, width)
        
        return ct_data
    
    def _generate_ct_3d(self, mri_data, segmentation=None):
        """Generate synthetic CT using 3D patches with overlap."""
        import tensorflow as tf
        from itertools import product
        
        # Get dimensions
        depth, height, width = mri_data.shape
        patch_height, patch_width, patch_depth = self.input_shape
        stride_height, stride_width, stride_depth = self.stride
        
        # Calculate patches
        z_steps = max(1, (depth - patch_depth + stride_depth - 1) // stride_depth + 1)
        y_steps = max(1, (height - patch_height + stride_height - 1) // stride_height + 1)
        x_steps = max(1, (width - patch_width + stride_width - 1) // stride_width + 1)
        
        # Prepare output array and weight accumulator for averaging overlapping regions
        ct_data = np.zeros_like(mri_data)
        weight_data = np.zeros_like(mri_data)
        
        # Process each patch
        for z, y, x in product(range(z_steps), range(y_steps), range(x_steps)):
            # Calculate patch coordinates
            z_start = min(z * stride_depth, depth - patch_depth)
            y_start = min(y * stride_height, height - patch_height)
            x_start = min(x * stride_width, width - patch_width)
            
            z_end = z_start + patch_depth
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            
            # Extract patch
            mri_patch = mri_data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Prepare input for network
            mri_patch = np.expand_dims(mri_patch, axis=0)  # Add batch dimension
            mri_patch = np.transpose(mri_patch, (0, 2, 3, 1))  # (B, H, W, D)
            
            # Add channel dimension if needed
            if len(mri_patch.shape) == 4:  # (B, H, W, D)
                mri_patch = np.expand_dims(mri_patch, axis=-1)  # (B, H, W, D, 1)
                
            # Prepare segmentation if available
            if segmentation is not None:
                seg_patch = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
                seg_patch = np.expand_dims(seg_patch, axis=0)  # Add batch dimension
                
                # Combine MRI and segmentation
                if len(seg_patch.shape) == 5:  # If already has channel dimension
                    input_patch = tf.concat([mri_patch, seg_patch], axis=-1)
                else:
                    # Add channel dimension and concatenate
                    seg_patch = np.transpose(seg_patch, (0, 2, 3, 1))  # (B, H, W, D)
                    input_patch = tf.concat([mri_patch, tf.expand_dims(seg_patch, axis=-1)], axis=-1)
            else:
                input_patch = mri_patch
                
            # Apply generator
            logger.debug(f"Processing 3D patch at ({z_start}-{z_end}, {y_start}-{y_end}, {x_start}-{x_end})")
            ct_patch = self.generator.predict(input_patch)
            
            # If output has multiple channels, take the first one
            if ct_patch.shape[-1] > 1:
                ct_patch = ct_patch[..., 0]
                
            # Convert back to original shape
            ct_patch = np.squeeze(ct_patch, axis=0)  # Remove batch dimension
            if len(ct_patch.shape) == 4:  # (H, W, D, 1)
                ct_patch = np.squeeze(ct_patch, axis=-1)  # Remove channel dimension
                
            # Transpose back if needed
            if ct_patch.shape != (patch_height, patch_width, patch_depth):
                ct_patch = np.transpose(ct_patch, (2, 0, 1))  # (D, H, W)
                
            # Create weight matrix (higher weights in the center, lower at the edges)
            weight = self._create_patch_weight(patch_depth, patch_height, patch_width)
            
            # Add to output arrays with weights
            ct_data[z_start:z_end, y_start:y_end, x_start:x_end] += ct_patch * weight
            weight_data[z_start:z_end, y_start:y_end, x_start:x_end] += weight
            
        # Normalize by weights
        non_zero_weights = weight_data > 0
        ct_data[non_zero_weights] /= weight_data[non_zero_weights]
        
        return ct_data
    
    def _generate_ct_3d_multi_sequence(self, mri_sequence_data, segmentation=None):
        """Generate synthetic CT from multiple MRI sequences using 3D patches."""
        import tensorflow as tf
        from itertools import product
        
        # Get first sequence to determine dimensions
        first_seq = next(iter(mri_sequence_data.values()))
        depth, height, width = first_seq.shape
        patch_height, patch_width, patch_depth = self.input_shape
        stride_height, stride_width, stride_depth = self.stride
        
        # Calculate patches
        z_steps = max(1, (depth - patch_depth + stride_depth - 1) // stride_depth + 1)
        y_steps = max(1, (height - patch_height + stride_height - 1) // stride_height + 1)
        x_steps = max(1, (width - patch_width + stride_width - 1) // stride_width + 1)
        
        # Prepare output array and weight accumulator for averaging overlapping regions
        ct_data = np.zeros((depth, height, width), dtype=np.float32)
        weight_data = np.zeros((depth, height, width), dtype=np.float32)
        
        # Process each patch
        for z, y, x in product(range(z_steps), range(y_steps), range(x_steps)):
            # Calculate patch coordinates
            z_start = min(z * stride_depth, depth - patch_depth)
            y_start = min(y * stride_height, height - patch_height)
            x_start = min(x * stride_width, width - patch_width)
            
            z_end = z_start + patch_depth
            y_end = y_start + patch_height
            x_end = x_start + patch_width
            
            # Extract patch from each sequence
            seq_patches = []
            for seq_data in mri_sequence_data.values():
                seq_patch = seq_data[z_start:z_end, y_start:y_end, x_start:x_end]
                seq_patch = np.expand_dims(seq_patch, axis=0)  # Add batch dimension
                seq_patch = np.transpose(seq_patch, (0, 2, 3, 1))  # (B, H, W, D)
                seq_patch = np.expand_dims(seq_patch, axis=-1)  # (B, H, W, D, 1)
                seq_patches.append(seq_patch)
                
            # Concatenate all sequences along channel dimension
            mri_patch = tf.concat(seq_patches, axis=-1)
                
            # Prepare segmentation if available
            if segmentation is not None:
                seg_patch = segmentation[z_start:z_end, y_start:y_end, x_start:x_end]
                seg_patch = np.expand_dims(seg_patch, axis=0)  # Add batch dimension
                
                # Combine MRI and segmentation
                if len(seg_patch.shape) == 5:  # If already has channel dimension
                    input_patch = tf.concat([mri_patch, seg_patch], axis=-1)
                else:
                    # Add channel dimension and concatenate
                    seg_patch = np.transpose(seg_patch, (0, 2, 3, 1))  # (B, H, W, D)
                    input_patch = tf.concat([mri_patch, tf.expand_dims(seg_patch, axis=-1)], axis=-1)
            else:
                input_patch = mri_patch
                
            # Apply generator
            logger.debug(f"Processing 3D multi-sequence patch at ({z_start}-{z_end}, {y_start}-{y_end}, {x_start}-{x_end})")
            ct_patch = self.generator.predict(input_patch)
            
            # If output has multiple channels, take the first one
            if ct_patch.shape[-1] > 1:
                ct_patch = ct_patch[..., 0]
                
            # Convert back to original shape
            ct_patch = np.squeeze(ct_patch, axis=0)  # Remove batch dimension
            if len(ct_patch.shape) == 4:  # (H, W, D, 1)
                ct_patch = np.squeeze(ct_patch, axis=-1)  # Remove channel dimension
                
            # Transpose back if needed
            if ct_patch.shape != (patch_height, patch_width, patch_depth):
                ct_patch = np.transpose(ct_patch, (2, 0, 1))  # (D, H, W)
                
            # Create weight matrix (higher weights in the center, lower at the edges)
            weight = self._create_patch_weight(patch_depth, patch_height, patch_width)
            
            # Add to output arrays with weights
            ct_data[z_start:z_end, y_start:y_end, x_start:x_end] += ct_patch * weight
            weight_data[z_start:z_end, y_start:y_end, x_start:x_end] += weight
            
        # Normalize by weights
        non_zero_weights = weight_data > 0
        ct_data[non_zero_weights] /= weight_data[non_zero_weights]
        
        return ct_data
        
    def _create_patch_weight(self, depth, height, width):
        """Create weight matrix for patch blending, giving higher weights to central voxels."""
        # Create 1D weight vectors with cosine falloff
        z_weight = np.cos((np.arange(depth) / (depth - 1) - 0.5) * np.pi)
        y_weight = np.cos((np.arange(height) / (height - 1) - 0.5) * np.pi)
        x_weight = np.cos((np.arange(width) / (width - 1) - 0.5) * np.pi)
        
        # Create 3D weight matrix using outer products
        weight = np.zeros((depth, height, width))
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    weight[z, y, x] = z_weight[z] * y_weight[y] * x_weight[x]
                    
        return weight
    
    def _normalize_intensity(self, data):
        """Normalize intensity values to range [-1, 1] for GAN input."""
        # Clip outliers
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        
        # Scale to [-1, 1]
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            data = 2 * (data - data_min) / (data_max - data_min) - 1
        
        return data
    
    def _denormalize_intensity(self, data, hu_min=-1000, hu_max=3000):
        """Convert normalized values back to HU range."""
        # Scale from [-1, 1] to [hu_min, hu_max]
        data = ((data + 1) / 2) * (hu_max - hu_min) + hu_min
        
        return data
    
    def _one_hot_encode_segmentation(self, segmentation, num_classes=None):
        """One-hot encode segmentation for conditioning input."""
        import tensorflow as tf
        
        # Determine number of classes if not specified
        if num_classes is None:
            num_classes = int(np.max(segmentation)) + 1
            
        # Create one-hot encoding
        if self.use_3d:
            # For 3D segmentation
            depth, height, width = segmentation.shape
            one_hot = np.zeros((depth, height, width, num_classes), dtype=np.float32)
            
            for i in range(num_classes):
                one_hot[..., i] = (segmentation == i).astype(np.float32)
                
            return one_hot
        else:
            # For 2D slices
            depth, height, width = segmentation.shape
            one_hot = np.zeros((depth, height, width, num_classes), dtype=np.float32)
            
            for i in range(num_classes):
                one_hot[..., i] = (segmentation == i).astype(np.float32)
                
            return one_hot


def convert_mri_to_ct(mri_image, segmentation=None, model_type='gan', region='head'):
    """
    Convert MRI to synthetic CT using the specified method.
    
    Args:
        mri_image: Input MRI as SimpleITK image, SyntheticCT object, or path to file
        segmentation: Segmentation mask as SimpleITK image or path to file (optional)
        model_type: Conversion method ('atlas', 'cnn', or 'gan')
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        
    Returns:
        Synthetic CT as SyntheticCT object
    """
    logger.info(f"Starting MRI to CT conversion using {model_type} method for {region} region")
    
    # Convert input to SyntheticCT if needed
    if isinstance(mri_image, str):
        # Load image from file
        from app.utils.io_utils import load_medical_image
        mri_image = load_medical_image(mri_image)
    
    if not isinstance(mri_image, SyntheticCT) and not isinstance(mri_image, sitk.Image):
        raise ValueError("mri_image must be a SimpleITK image, SyntheticCT object, or path to file")
    
    # Convert segmentation to SimpleITK image if needed
    if segmentation is not None and isinstance(segmentation, str):
        # Load segmentation from file
        from app.utils.io_utils import load_medical_image
        segmentation = load_medical_image(segmentation)
    
    # Select converter based on model type
    try:
        if model_type == 'atlas':
            # Atlas-based conversion
            converter = AtlasBasedConverter(region)
            synthetic_ct = converter.convert(mri_image, segmentation)
        
        elif model_type == 'cnn':
            # CNN-based conversion
            converter = CNNConverter(region=region)
            synthetic_ct = converter.convert(mri_image, segmentation)
        
        elif model_type == 'gan':
            # GAN-based conversion
            converter = GANConverter(region=region)
            synthetic_ct = converter.convert(mri_image, segmentation)
        
        else:
            logger.warning(f"Unknown model type: {model_type}. Using GAN.")
            converter = GANConverter(region=region)
            synthetic_ct = converter.convert(mri_image, segmentation)
        
        logger.info(f"MRI to CT conversion completed using {model_type} method")
        return synthetic_ct
    
    except Exception as e:
        logger.error(f"Error in MRI to CT conversion: {str(e)}")
        raise 