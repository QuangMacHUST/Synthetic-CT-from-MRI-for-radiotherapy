#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified test script to verify GAN converter functionality.
"""

import os
import sys
import logging
import SimpleITK as sitk
import numpy as np

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging with debug level
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set specific module log levels
logging.getLogger('app.core.conversion.convert_mri_to_ct').setLevel(logging.DEBUG)
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

def create_manual_config():
    """Create a manually defined config that doesn't rely on YAML loading."""
    config = {
        'conversion': {
            'gan': {
                'head': {
                    'generator_path': 'models/gan/head_gan_generator.h5',
                    'use_3d': False,
                    'batch_size': 1,
                    'input_shape': [256, 256, 1],
                    'use_multi_sequence': False,
                    'sequence_names': ['T1']
                },
                'pelvis': {
                    'generator_path': 'models/gan/pelvis_gan_generator.h5',
                    'use_3d': False,
                    'batch_size': 1,
                    'input_shape': [256, 256, 1],
                    'use_multi_sequence': False,
                    'sequence_names': ['T1']
                },
                'thorax': {
                    'generator_path': 'models/gan/thorax_gan_generator.h5',
                    'use_3d': False,
                    'batch_size': 1,
                    'input_shape': [256, 256, 1],
                    'use_multi_sequence': False,
                    'sequence_names': ['T1']
                }
            }
        }
    }
    return config

def test_gan_initialization():
    """Test just the GAN converter initialization."""
    from app.core.conversion.convert_mri_to_ct import GANConverter
    
    # Get manual config
    config = create_manual_config()
    logger.debug(f"Using manual config: {config}")
    
    logger.info("Testing GAN converter initialization...")
    
    # Try loading with explicit config
    for region in ['head', 'pelvis', 'thorax']:
        try:
            converter = GANConverter(region=region, config=config)
            logger.info(f"Successfully loaded GAN converter for {region} region with explicit config")
        except Exception as e:
            logger.error(f"Failed to load GAN converter for {region} region with explicit config: {str(e)}")
    
    # Try default loading
    for region in ['head', 'pelvis', 'thorax']:
        try:
            converter = GANConverter(region=region)
            logger.info(f"Successfully loaded GAN converter for {region} region with default config")
        except Exception as e:
            logger.error(f"Failed to load GAN converter for {region} region with default config: {str(e)}")

def test_gan_conversion():
    """Test basic GAN conversion functionality."""
    from app.core.conversion.convert_mri_to_ct import GANConverter, convert_mri_to_ct
    
    logger.info("Testing GAN conversion...")
    
    # Create dummy MRI
    mri = create_dummy_mri()
    
    # Get manual config
    config = create_manual_config()
    
    # First try direct conversion
    try:
        logger.info("Testing direct GAN conversion...")
        synthetic_ct = convert_mri_to_ct(
            mri_image=mri,
            model_type='gan',
            region='head'
        )
        
        logger.info("Checking output dimensions...")
        if (synthetic_ct.GetSize() == mri.GetSize() and 
            synthetic_ct.GetSpacing() == mri.GetSpacing() and
            synthetic_ct.GetOrigin() == mri.GetOrigin()):
            logger.info("GAN conversion output dimensions match input")
        else:
            logger.error("GAN conversion output dimensions do not match input")
            
        logger.info("Successfully converted using GAN")
    except Exception as e:
        logger.error(f"Failed to convert using GAN directly: {str(e)}")
    
    # Now try using the converter directly
    try:
        logger.info("Testing converter instance directly...")
        converter = GANConverter(region='head', config=config)
        synthetic_ct = converter.convert(mri)
        
        logger.info("Checking output dimensions...")
        if (synthetic_ct.GetSize() == mri.GetSize() and 
            synthetic_ct.GetSpacing() == mri.GetSpacing() and
            synthetic_ct.GetOrigin() == mri.GetOrigin()):
            logger.info("Direct GAN converter output dimensions match input")
        else:
            logger.error("Direct GAN converter output dimensions do not match input")
            
        logger.info("Successfully converted using direct GAN converter")
    except Exception as e:
        logger.error(f"Failed to convert using GAN converter directly: {str(e)}")

def main():
    """Run tests."""
    logger.info("Starting GAN tests...")
    
    # Test GAN converter initialization
    test_gan_initialization()
    
    # Test GAN conversion
    test_gan_conversion()
    
    logger.info("GAN tests completed")

if __name__ == "__main__":
    main() 