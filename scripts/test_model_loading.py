#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify model loading functionality in the MRI to CT conversion system.
"""

import os
import sys
import logging
import SimpleITK as sitk
import numpy as np

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_dummy_mri():
    """Create a dummy MRI image for testing."""
    # Create a simple 3D image with a sphere
    size = [64, 64, 64]
    image = sitk.Image(size[0], size[1], size[2], sitk.sitkFloat32)
    
    # Set pixel spacing
    image.SetSpacing([1.0, 1.0, 1.0])
    
    # Fill with a simple pattern (sphere)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Calculate distance from center
                distance = np.sqrt((x - size[0]//2)**2 + 
                                   (y - size[1]//2)**2 + 
                                   (z - size[2]//2)**2)
                
                # Create a sphere
                if distance < 20:
                    image[x, y, z] = 1000  # High intensity in the sphere
                else:
                    image[x, y, z] = 100   # Low intensity outside
    
    return image

def create_dummy_segmentation():
    """Create a dummy segmentation mask for testing."""
    # Create a simple 3D mask with a sphere
    size = [64, 64, 64]
    mask = sitk.Image(size[0], size[1], size[2], sitk.sitkUInt8)
    
    # Set pixel spacing
    mask.SetSpacing([1.0, 1.0, 1.0])
    
    # Fill with a simple pattern (concentric spheres)
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Calculate distance from center
                distance = np.sqrt((x - size[0]//2)**2 + 
                                   (y - size[1]//2)**2 + 
                                   (z - size[2]//2)**2)
                
                # Create concentric regions
                if distance < 10:
                    mask[x, y, z] = 3      # Bone
                elif distance < 15:
                    mask[x, y, z] = 2      # Soft tissue
                elif distance < 20:
                    mask[x, y, z] = 4      # Fat
                else:
                    mask[x, y, z] = 1      # Air
    
    return mask

def test_atlas_loading():
    """Test loading the atlas-based converter."""
    from app.core.conversion.convert_mri_to_ct import AtlasBasedConverter
    
    logger.info("Testing atlas-based converter loading...")
    
    # Try loading for each anatomical region
    for region in ['head', 'pelvis', 'thorax']:
        try:
            converter = AtlasBasedConverter(region=region)
            logger.info(f"Successfully loaded atlas converter for {region} region")
        except Exception as e:
            logger.error(f"Failed to load atlas converter for {region} region: {str(e)}")

def test_cnn_loading():
    """Test loading the CNN-based converter."""
    from app.core.conversion.convert_mri_to_ct import CNNConverter
    
    logger.info("Testing CNN-based converter loading...")
    
    # Try loading for each anatomical region
    for region in ['head', 'pelvis', 'thorax']:
        try:
            converter = CNNConverter(region=region)
            logger.info(f"Successfully loaded CNN converter for {region} region")
        except Exception as e:
            logger.error(f"Failed to load CNN converter for {region} region: {str(e)}")

def test_gan_loading():
    """Test loading the GAN-based converter."""
    from app.core.conversion.convert_mri_to_ct import GANConverter
    
    logger.info("Testing GAN-based converter loading...")
    
    # Try loading for each anatomical region
    for region in ['head', 'pelvis', 'thorax']:
        try:
            converter = GANConverter(region=region)
            logger.info(f"Successfully loaded GAN converter for {region} region")
        except Exception as e:
            logger.error(f"Failed to load GAN converter for {region} region: {str(e)}")

def test_basic_conversion():
    """Test basic conversion functionality."""
    from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct
    
    logger.info("Testing basic conversion functionality...")
    
    # Create dummy data
    mri = create_dummy_mri()
    segmentation = create_dummy_segmentation()
    
    # Try to convert using different methods
    for method in ['atlas', 'cnn', 'gan']:
        try:
            logger.info(f"Testing {method} conversion method...")
            synthetic_ct = convert_mri_to_ct(
                mri_image=mri,
                segmentation=segmentation,
                model_type=method,
                region='head'
            )
            logger.info(f"Successfully converted using {method} method")
            
            # Basic validation of the output
            if isinstance(synthetic_ct, sitk.Image):
                # Check if the output has the same dimensions as the input
                if (synthetic_ct.GetSize() == mri.GetSize() and 
                    synthetic_ct.GetSpacing() == mri.GetSpacing() and
                    synthetic_ct.GetOrigin() == mri.GetOrigin()):
                    logger.info(f"Output validation passed for {method} method")
                else:
                    logger.error(f"Output dimensions do not match input for {method} method")
            else:
                logger.error(f"Output is not a SimpleITK image for {method} method")
                
        except Exception as e:
            logger.error(f"Failed to convert using {method} method: {str(e)}")

def save_test_results(output_dir='test_results'):
    """Save test MRI, segmentation, and converted CT for visual inspection."""
    from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct
    
    logger.info(f"Saving test results to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy data
    mri = create_dummy_mri()
    segmentation = create_dummy_segmentation()
    
    # Save input data
    sitk.WriteImage(mri, os.path.join(output_dir, 'test_mri.nii.gz'))
    sitk.WriteImage(segmentation, os.path.join(output_dir, 'test_segmentation.nii.gz'))
    
    # Convert using different methods and save results
    for method in ['atlas', 'cnn', 'gan']:
        try:
            logger.info(f"Converting using {method} method...")
            synthetic_ct = convert_mri_to_ct(
                mri_image=mri,
                segmentation=segmentation,
                model_type=method,
                region='head'
            )
            
            # Save the result
            output_path = os.path.join(output_dir, f'synthetic_ct_{method}.nii.gz')
            sitk.WriteImage(synthetic_ct, output_path)
            logger.info(f"Saved {method} result to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save {method} result: {str(e)}")
    
    logger.info("All test results saved")

def main():
    """Run all tests."""
    logger.info("Starting model loading tests...")
    
    # Test atlas loading
    test_atlas_loading()
    
    # Test CNN loading
    test_cnn_loading()
    
    # Test GAN loading
    test_gan_loading()
    
    # Test basic conversion - now enabled with mock models
    test_basic_conversion()
    
    # Save test results
    save_test_results()
    
    logger.info("Model loading tests completed")

if __name__ == "__main__":
    main() 