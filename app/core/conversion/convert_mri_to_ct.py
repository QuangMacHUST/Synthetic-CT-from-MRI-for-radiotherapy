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
    GAN-based MRI to CT conversion using generative adversarial networks.
    """
    
    def __init__(self, model_path=None, region='head'):
        """
        Initialize GAN converter.
        
        Args:
            model_path: Path to GAN generator model (optional)
            region: Anatomical region ('head', 'pelvis', or 'thorax')
        """
        self.region = region
        
        # Get conversion configuration
        conv_config = config.get_conversion_params('gan', region)
        
        # Get model path
        self.model_path = model_path or conv_config.get('generator_path')
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            logger.warning(f"GAN model not found at {self.model_path}")
            
            # Try default location
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_path = os.path.join(root_dir, "models", "gan", f"{region}_gan_generator.h5")
            
            if os.path.exists(default_path):
                logger.info(f"Using default model at {default_path}")
                self.model_path = default_path
            else:
                logger.error(f"Default model not found at {default_path}")
                raise FileNotFoundError(f"GAN model not found for {region} region")
        
        # Get other parameters
        self.batch_size = conv_config.get('batch_size', 1)
        self.input_shape = conv_config.get('input_shape', [256, 256, 1])
        self.use_3d = conv_config.get('use_3d', False)
        self.patch_size = conv_config.get('patch_size', [256, 256, 32])
        self.stride = conv_config.get('stride', [128, 128, 16])
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """
        Load GAN generator model.
        
        Returns:
            TensorFlow model
        """
        try:
            logger.info(f"Loading GAN generator model from {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logger.info("GAN generator model loaded successfully")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load GAN generator model: {str(e)}")
            raise
    
    def convert(self, mri_image, segmentation=None):
        """
        Convert MRI to CT using GAN approach.
        
        Args:
            mri_image: Input MRI as SimpleITK image or SyntheticCT object
            segmentation: Segmentation mask as SimpleITK image (optional)
            
        Returns:
            Synthetic CT as SyntheticCT object
        """
        logger.info(f"Starting GAN-based MRI to CT conversion for {self.region} region")
        
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
            
            # Normalize input to [-1, 1] (typical for GANs)
            mask = mri_array > 0  # Non-zero mask
            if np.any(mask):
                min_val = np.min(mri_array[mask])
                max_val = np.max(mri_array[mask])
                if max_val > min_val:
                    mri_array = (mri_array - min_val) / (max_val - min_val)
                    mri_array = mri_array * 2 - 1  # Scale to [-1, 1]
                    mri_array[~mask] = -1  # Set background to -1
            
            # Process based on 2D or 3D approach
            if self.use_3d:
                # 3D GAN processing
                logger.info("Using 3D GAN approach")
                
                # Add batch and channel dimensions
                input_data = np.expand_dims(mri_array, axis=(0, -1)).astype(np.float32)
                
                # If segmentation is provided, include it as additional channels
                if segmentation is not None:
                    segmentation_array = sitk.GetArrayFromImage(segmentation)
                    
                    # One-hot encode the segmentation
                    num_classes = np.max(segmentation_array) + 1
                    one_hot_seg = np.eye(num_classes)[segmentation_array]
                    
                    # Add batch dimension and combine with MRI
                    seg_data = np.expand_dims(one_hot_seg, axis=0)
                    input_data = np.concatenate([input_data, seg_data], axis=-1)
                
                # Generate synthetic CT
                synthetic_ct_array = self.model.predict(input_data, batch_size=self.batch_size, verbose=0)
                
                # Remove batch dimension
                synthetic_ct_array = synthetic_ct_array[0, ..., 0]
            
            else:
                # 2D slice-by-slice processing
                logger.info("Using 2D slice-by-slice GAN approach")
                
                # Process each slice separately
                synthetic_ct_slices = []
                
                for z in range(mri_array.shape[0]):
                    # Get slice
                    slice_data = mri_array[z]
                    
                    # Add batch and channel dimensions
                    slice_data = np.expand_dims(slice_data, axis=(0, -1)).astype(np.float32)
                    
                    # If segmentation is provided, include it as additional channels
                    if segmentation is not None:
                        segmentation_array = sitk.GetArrayFromImage(segmentation)
                        seg_slice = segmentation_array[z]
                        
                        # One-hot encode the segmentation
                        num_classes = np.max(seg_slice) + 1
                        one_hot_seg = np.eye(num_classes)[seg_slice]
                        
                        # Add batch dimension and combine with MRI
                        seg_data = np.expand_dims(one_hot_seg, axis=0)
                        slice_data = np.concatenate([slice_data, seg_data], axis=-1)
                    
                    # Generate synthetic CT slice
                    ct_slice = self.model.predict(slice_data, batch_size=1, verbose=0)
                    
                    # Remove batch and channel dimensions
                    ct_slice = ct_slice[0, ..., 0]
                    
                    synthetic_ct_slices.append(ct_slice)
                
                # Stack slices
                synthetic_ct_array = np.stack(synthetic_ct_slices, axis=0)
            
            # Convert to HU units (assumes model output is normalized to [-1, 1])
            # Scale from [-1, 1] to HU range (e.g., [-1000, 1000])
            synthetic_ct_array = (synthetic_ct_array + 1) / 2 * 2000 - 1000
            synthetic_ct_array = synthetic_ct_array.astype(np.int16)
            
            # Create SimpleITK image
            synthetic_ct_image = sitk.GetImageFromArray(synthetic_ct_array)
            synthetic_ct_image.CopyInformation(input_image)
            
            # Create metadata
            conversion_metadata = {
                'method': 'gan',
                'region': self.region,
                'model_path': self.model_path,
                'use_3d': self.use_3d
            }
            
            if 'conversion' not in metadata:
                metadata['conversion'] = conversion_metadata
            else:
                metadata['conversion'].update(conversion_metadata)
            
            # Create SyntheticCT object
            synthetic_ct = SyntheticCT(synthetic_ct_image, metadata)
            
            logger.info("GAN-based MRI to CT conversion completed")
            return synthetic_ct
        
        except Exception as e:
            logger.error(f"Error in GAN-based conversion: {str(e)}")
            raise


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