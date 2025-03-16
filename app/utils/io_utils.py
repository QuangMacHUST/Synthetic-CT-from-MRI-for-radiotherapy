#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for input/output operations
"""

import os
import logging
import datetime
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import FileDataset
from pydicom.uid import generate_uid


def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(
                    "logs", 
                    f"synthetic_ct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
            )
        ]
    )
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)


def load_dicom_series(directory):
    """
    Load a DICOM series from a directory.
    
    Args:
        directory: Path to directory containing DICOM files
        
    Returns:
        SimpleITK image
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def load_nifti(file_path):
    """
    Load a NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        SimpleITK image
    """
    return sitk.ReadImage(file_path)


def load_medical_image(path):
    """
    Load a medical image from a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        SimpleITK image
    """
    path = Path(path)
    
    if path.is_dir():
        # Try to load as DICOM series
        return load_dicom_series(str(path))
    else:
        # Try to load as NIfTI
        if path.suffix in ['.nii', '.gz']:
            return load_nifti(str(path))
        else:
            # Try to load as single DICOM file
            return sitk.ReadImage(str(path))


def save_medical_image(image, output_path):
    """
    Save a medical image to a file.
    
    Args:
        image: SimpleITK image to save
        output_path: Path to save the image
    """
    output_path = Path(output_path)
    
    # Create parent directories if they don't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Save based on file extension
    if output_path.suffix in ['.nii', '.gz']:
        sitk.WriteImage(image, str(output_path))
    elif output_path.suffix in ['.dcm']:
        sitk.WriteImage(image, str(output_path))
    else:
        # Default to NIfTI
        sitk.WriteImage(image, str(output_path))


def save_as_nifti(image, output_path):
    """
    Save a SimpleITK image as NIfTI.
    
    Args:
        image: SimpleITK image
        output_path: Path to save the NIfTI file
    """
    sitk.WriteImage(image, output_path)


def save_as_dicom(image, output_dir, patient_info=None):
    """
    Save a SimpleITK image as DICOM series.
    
    Args:
        image: SimpleITK image
        output_dir: Directory to save the DICOM series
        patient_info: Dictionary with patient information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert SimpleITK image to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Get image properties
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    
    # Create a base DICOM dataset
    if patient_info is None:
        patient_info = {
            'PatientName': 'ANONYMOUS',
            'PatientID': 'ANONYMOUS',
            'PatientBirthDate': '',
            'PatientSex': '',
            'StudyDescription': 'Synthetic CT',
            'SeriesDescription': 'Synthetic CT from MRI',
        }
    
    # Generate UIDs
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    
    # Save each slice as a separate DICOM file
    for i in range(array.shape[0]):
        # Create a new DICOM dataset for this slice
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        
        ds = FileDataset(
            os.path.join(output_dir, f'slice_{i:04d}.dcm'),
            {},
            file_meta=file_meta,
            preamble=b'\0' * 128
        )
        
        # Patient information
        ds.PatientName = patient_info.get('PatientName', 'ANONYMOUS')
        ds.PatientID = patient_info.get('PatientID', 'ANONYMOUS')
        ds.PatientBirthDate = patient_info.get('PatientBirthDate', '')
        ds.PatientSex = patient_info.get('PatientSex', '')
        
        # Study information
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
        ds.StudyDescription = patient_info.get('StudyDescription', 'Synthetic CT')
        
        # Series information
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = 1
        ds.SeriesDescription = patient_info.get('SeriesDescription', 'Synthetic CT from MRI')
        
        # Image information
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = 'CT'
        
        # CT-specific attributes
        ds.RescaleIntercept = -1024.0
        ds.RescaleSlope = 1.0
        ds.RescaleType = 'HU'
        
        # Image position and orientation
        ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + i * spacing[2]]
        ds.ImageOrientationPatient = [
            direction[0], direction[1], direction[2],
            direction[3], direction[4], direction[5]
        ]
        
        # Pixel spacing
        ds.PixelSpacing = [spacing[0], spacing[1]]
        ds.SliceThickness = spacing[2]
        
        # Pixel data
        ds.Rows = array.shape[1]
        ds.Columns = array.shape[2]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1  # Signed
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        
        # Convert HU values to pixel values
        pixel_array = array[i].astype(np.int16)
        ds.PixelData = pixel_array.tobytes()
        
        # Save the DICOM file
        ds.save_as(os.path.join(output_dir, f'slice_{i:04d}.dcm'))


class SyntheticCT:
    """
    Class to represent a synthetic CT image.
    """
    
    def __init__(self, image, metadata=None):
        """
        Initialize a synthetic CT image.
        
        Args:
            image: SimpleITK image
            metadata: Dictionary with metadata
        """
        self.image = image
        if metadata is None:
            metadata = {
                "creation_time": datetime.datetime.now().isoformat()
            }
        self.metadata = metadata
    
    def save(self, output_path):
        """
        Save the synthetic CT image.
        
        Args:
            output_path: Path to save the image
        """
        output_path = Path(output_path)
        
        if output_path.suffix in ['.nii', '.gz']:
            save_as_nifti(self.image, str(output_path))
        elif output_path.is_dir():
            save_as_dicom(self.image, str(output_path), self.metadata.get('patient_info'))
        else:
            # Default to NIfTI
            save_as_nifti(self.image, str(output_path))
    
    def get_array(self):
        """
        Get the image as a numpy array.
        
        Returns:
            Numpy array
        """
        return sitk.GetArrayFromImage(self.image)
    
    def get_metadata(self):
        """
        Get the metadata.
        
        Returns:
            Dictionary with metadata
        """
        return self.metadata
    
    @classmethod
    def load(cls, file_path, metadata=None):
        """
        Load a synthetic CT image from a file.
        
        Args:
            file_path: Path to the image file
            metadata: Optional metadata dictionary
            
        Returns:
            SyntheticCT object
        """
        image = load_medical_image(file_path)
        return cls(image, metadata) 