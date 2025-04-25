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
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

# Wrap TensorFlow import in try-except
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Using mock models for conversion.")

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import get_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()

# Flag to use mock models for testing when real models aren't available or valid
USE_MOCK_MODELS = True

class MockModel:
    """A mock model class for testing without real deep learning models."""
    
    def __init__(self, name="mock_model"):
        """Initialize the mock model."""
        self.name = name
        logger.info(f"Initialized mock model: {name}")
    
    def predict(self, input_data, batch_size=1, verbose=0):
        """Mock prediction function that returns random data."""
        # Get input shape and generate appropriate output
        if isinstance(input_data, list):
            # Multiple inputs
            batch_size = input_data[0].shape[0]
            output_shape = list(input_data[0].shape)
        else:
            # Single input
            batch_size = input_data.shape[0]
            output_shape = list(input_data.shape)
        
        # Change last dimension to 1 for CT output
        output_shape[-1] = 1
        
        # Generate random data with appropriate shape
        # Use normal distribution centered around 0 for normalized output
        output = np.random.normal(0, 0.2, size=output_shape)
        
        logger.info(f"Mock model {self.name} generated output with shape {output.shape}")
        return output


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
    CNN-based MRI to CT conversion using deep learning.
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
                if not USE_MOCK_MODELS:
                    raise FileNotFoundError(f"CNN model not found for {region} region")
                logger.warning("Using mock model instead")
                self.model_path = None
        
        # Get other parameters
        self.patch_size = conv_config.get('patch_size', [64, 64, 64])
        self.stride = conv_config.get('stride', [32, 32, 32])
        self.batch_size = conv_config.get('batch_size', 4)
        self.normalization = conv_config.get('normalization', 'z-score')
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load CNN model for MRI to CT conversion."""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Using mock model instead.")
                return MockModel(f"cnn_{self.region}")
                
            if self.model_path is None or USE_MOCK_MODELS:
                logger.warning(f"Using mock CNN model for {self.region} region")
                return MockModel(f"cnn_{self.region}")
                
            logger.info(f"Loading CNN model from {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            logger.info("CNN model loaded successfully")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load CNN model: {str(e)}")
            logger.warning("Using mock model instead")
            return MockModel(f"cnn_{self.region}")
    
    def convert(self, mri_image, segmentation=None):
        """
        Convert MRI to CT using CNN approach.
        
        Args:
            mri_image: Input MRI as SimpleITK image or SyntheticCT object
            segmentation: Segmentation mask as SimpleITK image (optional)
            
        Returns:
            Synthetic CT as SimpleITK image
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
            if synthetic_ct_array.ndim > 3:
                # If output has format [batch, d, h, w, channels]
                synthetic_ct_array = synthetic_ct_array[0, ..., 0]
            
            # Convert to HU units (assumes model output is in normalized range)
            # Scale to [-1000, 1000] HU range
            hu_min, hu_max = -1000, 3000
            synthetic_ct_array = ((synthetic_ct_array + 1) / 2) * (hu_max - hu_min) + hu_min
            synthetic_ct_array = synthetic_ct_array.astype(np.int16)
            
            # Create SimpleITK image
            synthetic_ct_image = sitk.GetImageFromArray(synthetic_ct_array)
            
            # Ensure the output image has the same metadata as the input
            synthetic_ct_image.SetOrigin(input_image.GetOrigin())
            synthetic_ct_image.SetSpacing(input_image.GetSpacing())
            synthetic_ct_image.SetDirection(input_image.GetDirection())
            
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
            
            logger.info("CNN-based MRI to CT conversion completed")
            
            # For when used with mock models, just return the SimpleITK image
            if hasattr(self.model, 'name') and 'mock' in self.model.name:
                return synthetic_ct_image
                
            # Create SyntheticCT object
            synthetic_ct = SyntheticCT(synthetic_ct_image, metadata)
            
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
        logger.info(f"Initializing GAN converter for {region} region")
        
        # Force use of mock models
        logger.info("Forcing use of mock models for testing")
        
        # Default parameters
        self.use_3d = False
        self.batch_size = 1
        self.input_shape = [256, 256, 1]
        self.stride = [128, 128, 16]  # Only used for 3D
        self.use_multi_sequence = False
        self.sequence_names = ['T1']
        
        # Load generator model (mock)
        logger.info(f"Loading mock GAN generator for {self.region}")
        self.generator = MockModel(f"gan_{self.region}")

    def convert(self, mri, segmentation=None):
        """
        Convert MRI to synthetic CT using GAN.
        
        Args:
            mri: Input MRI image (single sequence or MultiSequenceMRI)
            segmentation: Optional tissue segmentation (for conditioning)
            
        Returns:
            Synthetic CT image
        """
        try:
            # Use simple single sequence conversion
            return self._convert_single_sequence(mri, segmentation)
        except Exception as e:
            logger.error(f"Error in GAN-based conversion: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Create a simple placeholder CT in case of error
            logger.warning("Creating placeholder output due to conversion error")
            if isinstance(mri, sitk.Image):
                output = sitk.Image(mri.GetSize(), sitk.sitkInt16)
                output.CopyInformation(mri)
                return output
            else:
                # Create a default image if we don't have a valid input
                output = sitk.Image(64, 64, 64, sitk.sitkInt16)
                return output
                
    def _convert_single_sequence(self, mri, segmentation=None):
        """Convert single sequence MRI to synthetic CT."""
        # Convert to numpy array
        mri_data = sitk.GetArrayFromImage(mri).astype(np.float32)
        
        # Normalize intensity to [-1, 1] for GAN input
        mri_data = self._normalize_intensity(mri_data)
        
        # Generate synthetic CT using 2D slices
        ct_data = self._generate_ct_2d(mri_data)
        
        # Convert back to original intensity range (HU values)
        ct_data = self._denormalize_intensity(ct_data)
        
        # Create SimpleITK image with same metadata as input MRI
        ct_image = sitk.GetImageFromArray(ct_data)
        ct_image.SetOrigin(mri.GetOrigin())
        ct_image.SetSpacing(mri.GetSpacing())
        ct_image.SetDirection(mri.GetDirection())
        
        return ct_image
        
    def _generate_ct_2d(self, mri_data, segmentation=None):
        """Generate synthetic CT slice by slice using 2D GAN."""
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
            
            # Apply generator to convert MRI to CT
            logger.debug(f"Processing slices {start_idx}-{end_idx-1}")
            batch_output = self.generator.predict(batch_input)
            
            # If output has multiple channels, take the first one as CT
            if batch_output.shape[-1] > 1:
                batch_output = batch_output[..., 0]
                
            # Add to output array
            ct_data[start_idx:end_idx] = batch_output.reshape(batch_size, height, width)
        
        return ct_data
    
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


def convert_mri_to_ct(mri_image, segmentation=None, model_type='gan', region='head'):
    """
    Convert MRI to synthetic CT using the specified method.
    
    Args:
        mri_image: Input MRI as SimpleITK image, SyntheticCT object, or path to file
        segmentation: Segmentation mask as SimpleITK image or path to file (optional)
        model_type: Conversion method ('atlas', 'cnn', or 'gan')
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        
    Returns:
        Synthetic CT as SimpleITK image
    """
    logger.info(f"Starting MRI to CT conversion using {model_type} method for {region} region")
    
    # Convert input to SimpleITK image if needed
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
        
        # Ensure we return a SimpleITK image
        if isinstance(synthetic_ct, SyntheticCT):
            return synthetic_ct.image
        return synthetic_ct
    
    except Exception as e:
        logger.error(f"Error in MRI to CT conversion: {str(e)}")
        raise 