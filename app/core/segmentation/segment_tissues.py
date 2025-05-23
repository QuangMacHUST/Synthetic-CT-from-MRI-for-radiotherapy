#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tissue segmentation module for synthetic CT generation.

This module provides functionality for segmenting different tissue types
from MRI images, which is a crucial step in generating synthetic CT images.
Different segmentation methods are implemented, including threshold-based,
atlas-based, and deep learning-based approaches.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path

# Check for additional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Deep learning segmentation will not be available.")

# Import utility modules
# First, try direct import for flexibility
try:
    from app.utils.config_utils import get_region_params
    from app.utils.io_utils import MultiSequenceMRI, SyntheticCT
except ImportError:
    # Fallback to package imports
    from app.utils import get_region_params
    from app.utils import MultiSequenceMRI, SyntheticCT


class TissueSegmentation:
    """Base class for tissue segmentation."""
    
    def __init__(self, config: Optional[Dict] = None, region: str = "brain"):
        """
        Initialize the tissue segmentation.
        
        Args:
            config: Configuration dictionary (optional)
            region: Anatomical region to segment
        """
        self.logger = logging.getLogger(__name__)
        self.region = region
        
        # Get region-specific parameters
        self.params = get_region_params(region)
        
        # Override with provided config if any
        if config:
            self.params.update(config)
            
        # Get tissue classes for this region
        self.tissue_classes = self.params.get('tissue_classes', [])
        
        # Get HU ranges for each tissue class
        self.hu_ranges = self.params.get('hu_ranges', {})
        
        # Get registration parameters
        self.registration_params = self.params.get('registration_params', {})
        
        self.logger.info(f"Initialized tissue segmentation for region: {region}")
        
        # Store configuration
        self.config = config or {}
    
    def segment_tissues(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> Dict[str, sitk.Image]:
        """
        Segment tissues from MRI data.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Dictionary mapping tissue types to segmentation masks
        """
        self.logger.info(f"Segmenting tissues for region: {self.region}")
        
        # Create empty segmentation masks for each tissue class
        segmentations = {}
        reference_image = self._get_reference_image(mri_data)
        
        for tissue in self.tissue_classes:
            segmentations[tissue] = self._create_empty_mask(reference_image)
            
        # Implement in derived classes
        return segmentations
    
    def _get_reference_image(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> sitk.Image:
        """
        Get reference image from MRI data.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Reference SimpleITK image
        """
        if isinstance(mri_data, MultiSequenceMRI):
            return mri_data.get_reference()
        else:
            return mri_data
    
    def _create_empty_mask(self, reference_image: sitk.Image) -> sitk.Image:
        """
        Create an empty segmentation mask based on reference image.
        
        Args:
            reference_image: Reference SimpleITK image
            
        Returns:
            Empty segmentation mask
        """
        # Create empty mask with same geometry as reference image
        mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
        mask.SetOrigin(reference_image.GetOrigin())
        mask.SetSpacing(reference_image.GetSpacing())
        mask.SetDirection(reference_image.GetDirection())
        
        return mask
    
    def get_tissue_classes(self) -> List[str]:
        """
        Get list of tissue classes for current region.
        
        Returns:
            List of tissue class names
        """
        return self.tissue_classes
        

class ThresholdSegmentation(TissueSegmentation):
    """Threshold-based tissue segmentation."""
    
    def __init__(self, config: Optional[Dict] = None, region: str = "brain"):
        """
        Initialize threshold-based segmentation.
        
        Args:
            config: Configuration dictionary (optional)
            region: Anatomical region to segment
        """
        super().__init__(config, region)
        self.logger.info("Using threshold-based segmentation")
        
        # Get threshold ranges for each tissue class
        self.hu_ranges = self.params.get("hu_ranges", {})
        if not self.hu_ranges:
            self.logger.warning(f"No HU ranges defined for region: {region}")
    
    def segment_tissues(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> Dict[str, sitk.Image]:
        """
        Segment tissues from MRI data using thresholding.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Dictionary mapping tissue types to segmentation masks
        """
        segmentations = super().segment_tissues(mri_data)
        if not segmentations:
            return {}
            
        # Get reference image
        reference_image = self._get_reference_image(mri_data)
        
        # Apply thresholding for each tissue class
        for tissue, mask in segmentations.items():
            # Get threshold range for this tissue
            if tissue not in self.hu_ranges:
                self.logger.warning(f"No HU range defined for tissue: {tissue}")
                continue
                
            lower, upper = self.hu_ranges[tissue]
            
            # Apply threshold
            self.logger.info(f"Applying threshold for {tissue}: [{lower}, {upper}]")
            threshold = sitk.BinaryThresholdImageFilter()
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetInsideValue(1)
            threshold.SetOutsideValue(0)
            
            # Apply threshold and update segmentation
            mask = threshold.Execute(reference_image)
            segmentations[tissue] = mask
            
        return segmentations
    
    def _preprocess_image(self, image: sitk.Image) -> sitk.Image:
        """
        Preprocess image for thresholding.
        
        Args:
            image: Input SimpleITK image
            
        Returns:
            Preprocessed SimpleITK image
        """
        # Apply smoothing
        smoothed = sitk.CurvatureFlow(image, timeStep=0.125, numberOfIterations=5)
        
        return smoothed


class AtlasSegmentation(TissueSegmentation):
    """Atlas-based tissue segmentation."""
    
    def __init__(self, config: Optional[Dict] = None, region: str = "brain", atlas_dir: Optional[str] = None):
        """
        Initialize atlas-based segmentation.
        
        Args:
            config: Configuration dictionary (optional)
            region: Anatomical region to segment
            atlas_dir: Directory containing atlas images and labels
        """
        super().__init__(config, region)
        self.logger.info("Using atlas-based segmentation")
        
        # Set atlas directory
        if atlas_dir is None:
            if self.config and "data" in self.config and "atlas_dir" in self.config["data"]:
                atlas_dir = self.config["data"]["atlas_dir"]
            else:
                atlas_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "data", 
                    "atlas"
                )
        
        self.atlas_dir = atlas_dir
        self.logger.info(f"Atlas directory: {self.atlas_dir}")
        
        # Check atlas directory
        if not os.path.exists(self.atlas_dir):
            self.logger.warning(f"Atlas directory not found: {self.atlas_dir}")
    
    def segment_tissues(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> Dict[str, sitk.Image]:
        """
        Segment tissues from MRI data using atlas-based segmentation.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Dictionary mapping tissue types to segmentation masks
        """
        segmentations = super().segment_tissues(mri_data)
        if not segmentations:
            return {}
            
        # Get reference image
        reference_image = self._get_reference_image(mri_data)
        
        # Load atlas images and labels for region
        atlas_images, atlas_labels = self._load_atlas_data()
        if not atlas_images or not atlas_labels:
            self.logger.error(f"Failed to load atlas data for region: {self.region}")
            return segmentations
            
        # Register atlas to target image
        self.logger.info("Registering atlas to target image")
        registered_labels = self._register_atlas(reference_image, atlas_images, atlas_labels)
        
        # Transfer labels to segmentations
        tissue_classes = self.get_tissue_classes()
        for i, tissue in enumerate(tissue_classes):
            # Extract mask for this tissue
            threshold = sitk.BinaryThresholdImageFilter()
            threshold.SetLowerThreshold(i + 1)
            threshold.SetUpperThreshold(i + 1)
            threshold.SetInsideValue(1)
            threshold.SetOutsideValue(0)
            
            segmentations[tissue] = threshold.Execute(registered_labels)
            
        return segmentations
    
    def _load_atlas_data(self) -> Tuple[List[sitk.Image], List[sitk.Image]]:
        """
        Load atlas images and labels.
        
        Returns:
            Tuple of (atlas images, atlas labels)
        """
        # Check atlas directory
        if not os.path.exists(self.atlas_dir):
            self.logger.error(f"Atlas directory not found: {self.atlas_dir}")
            return [], []
            
        # Get region-specific atlas directory
        region_atlas_dir = os.path.join(self.atlas_dir, self.region)
        if not os.path.exists(region_atlas_dir):
            self.logger.error(f"Region-specific atlas directory not found: {region_atlas_dir}")
            return [], []
            
        # Find atlas images and labels
        atlas_images = []
        atlas_labels = []
        
        for subdir in os.listdir(region_atlas_dir):
            subdir_path = os.path.join(region_atlas_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
                
            # Find image and label files
            image_file = None
            label_file = None
            
            for file in os.listdir(subdir_path):
                if file.endswith("_image.nii.gz"):
                    image_file = os.path.join(subdir_path, file)
                elif file.endswith("_labels.nii.gz"):
                    label_file = os.path.join(subdir_path, file)
            
            if image_file and label_file:
                try:
                    image = sitk.ReadImage(image_file)
                    label = sitk.ReadImage(label_file)
                    
                    atlas_images.append(image)
                    atlas_labels.append(label)
                    
                    self.logger.info(f"Loaded atlas: {subdir}")
                except Exception as e:
                    self.logger.error(f"Error loading atlas {subdir}: {str(e)}")
        
        self.logger.info(f"Loaded {len(atlas_images)} atlases for region: {self.region}")
        return atlas_images, atlas_labels
    
    def _register_atlas(self, target: sitk.Image, atlas_images: List[sitk.Image], 
                      atlas_labels: List[sitk.Image]) -> sitk.Image:
        """
        Register atlas to target image.
        
        Args:
            target: Target image to register to
            atlas_images: List of atlas images
            atlas_labels: List of atlas labels
            
        Returns:
            Registered label image
        """
        if not atlas_images or not atlas_labels:
            return self._create_empty_mask(target)
            
        # Use first atlas for now (can be extended to multi-atlas segmentation)
        atlas_image = atlas_images[0]
        atlas_label = atlas_labels[0]
        
        # Get registration parameters
        reg_params = self.params.get("registration_params", {})
        transform_type = reg_params.get("transform_type", "rigid")
        
        # Set up registration
        registration = sitk.ImageRegistrationMethod()
        
        # Set up metric
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(reg_params.get("sampling_percentage", 0.1))
        
        # Set up optimizer
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            estimateLearningRate=registration.EachIteration
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set up interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Set up initial transform
        if transform_type == "rigid":
            initial_transform = sitk.CenteredTransformInitializer(
                target, 
                atlas_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif transform_type == "affine":
            initial_transform = sitk.CenteredTransformInitializer(
                target, 
                atlas_image, 
                sitk.AffineTransform(3), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif transform_type == "bspline":
            mesh_size = [10, 10, 10]
            initial_transform = sitk.BSplineTransformInitializer(
                target, 
                mesh_size
            )
        else:
            self.logger.warning(f"Unsupported transform type: {transform_type}, using rigid transform")
            initial_transform = sitk.CenteredTransformInitializer(
                target, 
                atlas_image, 
                sitk.Euler3DTransform(), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        
        registration.SetInitialTransform(initial_transform)
        
        # Perform registration
        self.logger.info(f"Performing {transform_type} registration")
        try:
            final_transform = registration.Execute(target, atlas_image)
            self.logger.info("Registration completed successfully")
        except Exception as e:
            self.logger.error(f"Registration failed: {str(e)}")
            self.logger.warning("Using initial transform")
            final_transform = initial_transform
        
        # Apply transform to label image
        self.logger.info("Applying transform to label image")
        reference_origin = target.GetOrigin()
        reference_spacing = target.GetSpacing()
        reference_direction = target.GetDirection()
        reference_size = target.GetSize()
        
        # Resample label image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(target)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for labels
        resampler.SetTransform(final_transform)
        
        registered_label = resampler.Execute(atlas_label)
        
        return registered_label


class DeepLearningSegmentation(TissueSegmentation):
    """Deep learning-based tissue segmentation."""
    
    def __init__(self, config: Optional[Dict] = None, region: str = "brain", model_path: Optional[str] = None):
        """
        Initialize deep learning-based segmentation.
        
        Args:
            config: Configuration dictionary (optional)
            region: Anatomical region to segment
            model_path: Path to trained model file
        """
        super().__init__(config, region)
        
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available. Deep learning segmentation cannot be used.")
            raise ImportError("PyTorch not available. Please install PyTorch to use deep learning segmentation.")
            
        self.logger.info("Using deep learning-based segmentation")
        
        # Set model path
        if model_path is None:
            if self.config and "models" in self.config and "segmentation" in self.config["models"]:
                model_path = self.config["models"]["segmentation"].get(region, None)
            
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                    "models", 
                    "segmentation",
                    f"{region}_segmentation.pt"
                )
        
        self.model_path = model_path
        self.logger.info(f"Model path: {self.model_path}")
        
        # Check model path
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model file not found: {self.model_path}")
            self.model = None
        else:
            # Load model
            self._load_model()
    
    def _load_model(self) -> None:
        """Load segmentation model."""
        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {device}")
            
            # Load model
            self.model = torch.load(self.model_path, map_location=device)
            self.model.eval()
            self.device = device
            
            self.logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def segment_tissues(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> Dict[str, sitk.Image]:
        """
        Segment tissues from MRI data using deep learning.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Dictionary mapping tissue types to segmentation masks
        """
        segmentations = super().segment_tissues(mri_data)
        if not segmentations:
            return {}
            
        if self.model is None:
            self.logger.error("No model loaded. Cannot perform deep learning segmentation.")
            return segmentations
            
        # Get reference image and input sequences
        reference_image = self._get_reference_image(mri_data)
        input_tensors = self._prepare_input(mri_data)
        
        if input_tensors is None:
            self.logger.error("Failed to prepare input tensors")
            return segmentations
            
        # Run model inference
        self.logger.info("Running model inference")
        try:
            with torch.no_grad():
                outputs = self.model(input_tensors)
                
            # Convert outputs to segmentation masks
            segmentations = self._convert_outputs_to_masks(outputs, reference_image)
        except Exception as e:
            self.logger.error(f"Error during model inference: {str(e)}")
            
        return segmentations
    
    def _prepare_input(self, mri_data: Union[MultiSequenceMRI, sitk.Image]) -> Optional[torch.Tensor]:
        """
        Prepare input tensors for the model.
        
        Args:
            mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
            
        Returns:
            Input tensor for model
        """
        try:
            # For multi-sequence MRI, use all available sequences
            if isinstance(mri_data, MultiSequenceMRI):
                sequences = []
                for name in mri_data.get_sequence_names():
                    seq_image = mri_data.get_sequence(name)
                    seq_array = sitk.GetArrayFromImage(seq_image).astype(np.float32)
                    
                    # Normalize
                    seq_array = (seq_array - seq_array.mean()) / (seq_array.std() + 1e-8)
                    
                    sequences.append(seq_array)
                
                # Stack sequences
                if sequences:
                    # Convert to tensor [B, C, D, H, W]
                    stacked = np.stack(sequences, axis=0)
                    tensor = torch.from_numpy(stacked).unsqueeze(0)
                    tensor = tensor.to(self.device)
                    return tensor
                else:
                    self.logger.error("No sequences found in MultiSequenceMRI")
                    return None
            else:
                # For single image, use as single channel
                image_array = sitk.GetArrayFromImage(mri_data).astype(np.float32)
                
                # Normalize
                image_array = (image_array - image_array.mean()) / (image_array.std() + 1e-8)
                
                # Convert to tensor [B, C, D, H, W]
                tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
                tensor = tensor.to(self.device)
                return tensor
        except Exception as e:
            self.logger.error(f"Error preparing input tensors: {str(e)}")
            return None
    
    def _convert_outputs_to_masks(self, outputs: torch.Tensor, reference_image: sitk.Image) -> Dict[str, sitk.Image]:
        """
        Convert model outputs to segmentation masks.
        
        Args:
            outputs: Model output tensor
            reference_image: Reference SimpleITK image
            
        Returns:
            Dictionary mapping tissue types to segmentation masks
        """
        tissue_classes = self.get_tissue_classes()
        segmentations = {}
        
        try:
            # Get output array (assume output is [B, C, D, H, W] with C = number of classes)
            output_array = outputs[0].cpu().numpy()  # Remove batch dimension
            
            # Create a mask for each tissue class
            for i, tissue in enumerate(tissue_classes):
                if i < output_array.shape[0]:
                    # Get probability map for this class
                    prob_map = output_array[i]
                    
                    # Threshold probability map
                    threshold_value = 0.5
                    binary_map = (prob_map > threshold_value).astype(np.uint8)
                    
                    # Convert to SimpleITK image
                    mask = sitk.GetImageFromArray(binary_map)
                    mask.SetOrigin(reference_image.GetOrigin())
                    mask.SetSpacing(reference_image.GetSpacing())
                    mask.SetDirection(reference_image.GetDirection())
                    
                    segmentations[tissue] = mask
                else:
                    # Create empty mask
                    segmentations[tissue] = self._create_empty_mask(reference_image)
        except Exception as e:
            self.logger.error(f"Error converting outputs to masks: {str(e)}")
            
            # Create empty masks
            for tissue in tissue_classes:
                segmentations[tissue] = self._create_empty_mask(reference_image)
        
        return segmentations


def segment_tissues(mri_data: Union[MultiSequenceMRI, sitk.Image], 
                  region: str = "brain", 
                  method: str = None,
                  config: Optional[Dict] = None) -> Dict[str, sitk.Image]:
    """
    Segment tissues from MRI data.
    
    Args:
        mri_data: MRI data (MultiSequenceMRI or SimpleITK image)
        region: Anatomical region to segment
        method: Segmentation method (threshold, atlas, deep_learning, or None to use region default)
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary mapping tissue types to segmentation masks
    """
    logger = logging.getLogger(__name__)
    
    # Get region parameters
    region_params = get_region_params(region)
    
    # Determine segmentation method
    if method is None:
        method = region_params.get("segmentation_method", "threshold")
    
    logger.info(f"Segmenting tissues for region {region} using {method} method")
    
    # Create appropriate segmentation object
    if method == "threshold":
        segmenter = ThresholdSegmentation(config, region)
    elif method == "atlas":
        segmenter = AtlasSegmentation(config, region)
    elif method == "deep_learning":
        try:
            segmenter = DeepLearningSegmentation(config, region)
        except ImportError:
            logger.warning("Deep learning segmentation not available. Falling back to atlas-based segmentation.")
            segmenter = AtlasSegmentation(config, region)
        else:
            logger.warning(f"Unknown segmentation method: {method}. Using threshold-based segmentation.")
            segmenter = ThresholdSegmentation(config, region)
    
    # Perform segmentation
    segmentations = segmenter.segment_tissues(mri_data)
    
    return segmentations 