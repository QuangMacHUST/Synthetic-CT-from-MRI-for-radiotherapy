#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for io_utils module
"""

import unittest
import os
import tempfile
import numpy as np
import SimpleITK as sitk

from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT


class TestIOUtils(unittest.TestCase):
    """Test cases for io_utils module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test image
        size = [20, 30, 40]
        spacing = [1.0, 1.0, 1.0]
        origin = [0.0, 0.0, 0.0]
        
        # Create a simple 3D image with a pattern
        img_array = np.zeros(size[::-1], dtype=np.float32)
        
        # Add a sphere
        center = np.array(size) / 2
        for i in range(size[2]):
            for j in range(size[1]):
                for k in range(size[0]):
                    p = np.array([k, j, i])
                    d = np.linalg.norm(p - center)
                    if d < 10:
                        img_array[i, j, k] = 100.0
        
        # Convert to SimpleITK image
        self.test_image = sitk.GetImageFromArray(img_array)
        self.test_image.SetSpacing(spacing)
        self.test_image.SetOrigin(origin)
        
        # Save the test image
        self.test_nii_path = os.path.join(self.test_dir, "test_image.nii.gz")
        sitk.WriteImage(self.test_image, self.test_nii_path)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        if os.path.exists(self.test_nii_path):
            os.remove(self.test_nii_path)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_load_medical_image(self):
        """Test loading a medical image."""
        # Load the test image
        loaded_image = load_medical_image(self.test_nii_path)
        
        # Check if the loaded image is a SimpleITK image
        self.assertIsInstance(loaded_image, sitk.Image)
        
        # Check image properties
        self.assertEqual(loaded_image.GetSize(), self.test_image.GetSize())
        self.assertEqual(loaded_image.GetSpacing(), self.test_image.GetSpacing())
        self.assertEqual(loaded_image.GetOrigin(), self.test_image.GetOrigin())
        
        # Check pixel values (convert to numpy arrays)
        loaded_array = sitk.GetArrayFromImage(loaded_image)
        test_array = sitk.GetArrayFromImage(self.test_image)
        np.testing.assert_allclose(loaded_array, test_array)
    
    def test_save_medical_image(self):
        """Test saving a medical image."""
        # Path for the saved image
        output_path = os.path.join(self.test_dir, "output_image.nii.gz")
        
        # Save the image
        save_medical_image(self.test_image, output_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load the saved image
        saved_image = load_medical_image(output_path)
        
        # Check if the saved image matches the original
        saved_array = sitk.GetArrayFromImage(saved_image)
        test_array = sitk.GetArrayFromImage(self.test_image)
        np.testing.assert_allclose(saved_array, test_array)
        
        # Clean up
        os.remove(output_path)
    
    def test_synthetic_ct_class(self):
        """Test SyntheticCT class."""
        # Create a SyntheticCT object
        synthetic_ct = SyntheticCT(self.test_image)
        
        # Check if the image is stored
        self.assertIsInstance(synthetic_ct.image, sitk.Image)
        
        # Check metadata
        self.assertIsInstance(synthetic_ct.metadata, dict)
        self.assertIn("creation_time", synthetic_ct.metadata)
        
        # Test setting and getting metadata
        synthetic_ct.metadata["test_key"] = "test_value"
        self.assertEqual(synthetic_ct.metadata["test_key"], "test_value")
        
        # Test saving and loading
        output_path = os.path.join(self.test_dir, "synthetic_ct.nii.gz")
        synthetic_ct.save(output_path)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Test loading a SyntheticCT
        loaded_ct = SyntheticCT.load(output_path)
        
        # Check if the loaded object is a SyntheticCT
        self.assertIsInstance(loaded_ct, SyntheticCT)
        
        # Check if the image matches
        loaded_array = sitk.GetArrayFromImage(loaded_ct.image)
        test_array = sitk.GetArrayFromImage(self.test_image)
        np.testing.assert_allclose(loaded_array, test_array)
        
        # Check if metadata was preserved
        self.assertIn("test_key", loaded_ct.metadata)
        self.assertEqual(loaded_ct.metadata["test_key"], "test_value")
        
        # Clean up
        os.remove(output_path)


if __name__ == "__main__":
    unittest.main() 