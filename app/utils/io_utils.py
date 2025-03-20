#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Input/Output utilities for MRI to CT conversion.

This module provides functions for loading, saving, and validating medical images,
as well as utilities for handling file paths and logging.
"""

import os
import sys
import logging
import shutil
import json
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any

# Try to import medical imaging libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("NiBabel not available. Limited NIfTI file support.")
    
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logging.warning("PyDICOM not available. Limited DICOM file support.")
    
try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK not available. Limited image processing support.")


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized (level: {logging.getLevelName(level)})")


def validate_input_file(file_path: str) -> bool:
    """
    Validate if a file path exists and is a valid medical image file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        True if file exists and is valid, False otherwise
    """
    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"Input file does not exist: {file_path}")
        return False
    
    # Check if file is a file (not a directory)
    if not os.path.isfile(file_path):
        logging.error(f"Input path is not a file: {file_path}")
        return False
    
    # Check file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # DICOM files might not have extension
    if not file_extension and PYDICOM_AVAILABLE:
        try:
            # Try to read as DICOM
            pydicom.dcmread(file_path)
            return True
        except Exception:
            pass
    
    # Check for valid medical image extensions
    valid_extensions = ['.nii', '.nii.gz', '.dcm', '.img', '.nrrd', '.mha', '.mhd']
    
    if file_extension not in valid_extensions and not file_path.lower().endswith('.nii.gz'):
        logging.warning(f"Input file has unknown extension: {file_extension}")
        # Don't return False here, still try to load the file
    
    # Additional validation could be done here, e.g. try to load the file
    # But for performance reasons, we don't do that by default
    
    return True


def ensure_output_dir(output_path: str) -> str:
    """
    Ensure the output directory exists. If the output path is a file path,
    ensure its parent directory exists.
    
    Args:
        output_path: Path to output file or directory
        
    Returns:
        Absolute path to the output
    """
    # Convert to absolute path
    abs_path = os.path.abspath(output_path)
    
    # Check if path has extension (likely a file path)
    if os.path.splitext(abs_path)[1]:
        # Create parent directory
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    else:
        # Create directory
        os.makedirs(abs_path, exist_ok=True)
    
    logging.debug(f"Ensured output path exists: {abs_path}")
    return abs_path


class MultiSequenceMRI:
    """Class for handling multi-sequence MRI data."""
    
    def __init__(self, sequences: Dict[str, Any] = None):
        """
        Initialize a multi-sequence MRI object.
    
        Args:
            sequences: Dictionary of sequence names to image data
        """
        self.sequences = sequences or {}
        self.metadata = {}

    def add_sequence(self, name: str, data: Any, metadata: Optional[Dict] = None) -> None:
        """
        Add a sequence to the multi-sequence MRI.
    
        Args:
            name: Sequence name (e.g., 'T1', 'T2', 'FLAIR')
            data: Image data
            metadata: Metadata dictionary
        """
        self.sequences[name] = data
        if metadata:
            if name not in self.metadata:
                self.metadata[name] = {}
            self.metadata[name].update(metadata)
    
    def get_sequence(self, name: str) -> Optional[Any]:
        """
        Get a specific sequence.
        
        Args:
            name: Sequence name
            
        Returns:
            Image data for the specified sequence, or None if not found
        """
        return self.sequences.get(name)
    
    def get_metadata(self, name: str) -> Dict:
        """
        Get metadata for a specific sequence.
        
        Args:
            name: Sequence name
            
        Returns:
            Metadata dictionary for the specified sequence
        """
        return self.metadata.get(name, {})
    
    def get_sequence_names(self) -> List[str]:
        """
        Get names of all available sequences.
        
        Returns:
            List of sequence names
        """
        return list(self.sequences.keys())
    
    def has_sequence(self, name: str) -> bool:
        """
        Check if a specific sequence exists.
        
        Args:
            name: Sequence name
            
        Returns:
            True if sequence exists, False otherwise
        """
        return name in self.sequences


class SyntheticCT:
    """Class for handling synthetic CT data."""
    
    def __init__(self, data: Any = None, metadata: Optional[Dict] = None):
        """
        Initialize a synthetic CT object.
        
        Args:
            data: Image data
            metadata: Metadata dictionary
        """
        self.data = data
        self.metadata = metadata or {}
        self.segmentation = None
        self.hu_map = None
    
    def set_segmentation(self, segmentation: Any) -> None:
        """
        Set segmentation data.
        
        Args:
            segmentation: Segmentation data
        """
        self.segmentation = segmentation
    
    def set_hu_map(self, hu_map: Dict) -> None:
        """
        Set HU value mapping.
        
        Args:
            hu_map: Dictionary mapping tissue types to HU values
        """
        self.hu_map = hu_map
    
    def get_data(self) -> Any:
        """
        Get image data.
        
        Returns:
            Image data
        """
        return self.data
    
    def get_metadata(self) -> Dict:
        """
        Get metadata.
        
        Returns:
            Metadata dictionary
        """
        return self.metadata
    

def load_medical_image(file_path: str) -> Any:
    """
    Load a medical image file.
    
    Uses SimpleITK to load common medical image formats (DICOM, NIfTI).
    Handles DICOM series in directories, individual DICOM files, and NIfTI files.
    For DICOM directories, it will search recursively to find all slices.
    
    Args:
        file_path: Path to medical image file or directory containing DICOM files
        
    Returns:
        SimpleITK.Image object or equivalent representation
        
    Raises:
        ValueError: If file doesn't exist or can't be loaded
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File or directory does not exist: {file_path}")
    
    # Normalize the path (convert backslashes to forward slashes)
    normalized_path = os.path.normpath(file_path).replace('\\', '/')
    
    logging.info(f"Loading medical image: {normalized_path}")
    
    try:
        import SimpleITK as sitk
        
        # Check file type and load accordingly
        if os.path.isfile(normalized_path):
            if normalized_path.lower().endswith(('.nii', '.nii.gz')):
                # NIfTI file
                logging.info(f"Loading NIfTI file: {normalized_path}")
                return sitk.ReadImage(normalized_path)
            else:
                # Attempt to load as DICOM or other image format
                try:
                    logging.info(f"Loading image file: {normalized_path}")
                    return sitk.ReadImage(normalized_path)
                except Exception as e:
                    logging.error(f"Failed to load file as medical image: {str(e)}")
                    raise ValueError(f"Unable to load {normalized_path}: {str(e)}")
        
        elif os.path.isdir(normalized_path):
            # Directory: could contain a DICOM series
            logging.info(f"Scanning directory for DICOM series: {normalized_path}")
            
            # First, collect all potential DICOM files
            dicom_files = []
            for root, _, files in os.walk(normalized_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Include files with .dcm extension or no extension (common for DICOM)
                    if file.lower().endswith('.dcm') or not os.path.splitext(file)[1]:
                        dicom_files.append(file_path)
            
            logging.info(f"Found {len(dicom_files)} potential DICOM files in directory")
            
            if not dicom_files:
                raise ValueError(f"No potential DICOM files found in directory: {normalized_path}")
            
            # Try using ImageSeriesReader with GDCM
            reader = sitk.ImageSeriesReader()
            try:
                series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(normalized_path)
                
                if series_IDs:
                    logging.info(f"Found {len(series_IDs)} DICOM series")
                    
                    # Try each series and load the one with the most files
                    best_series = None
                    max_file_count = 0
                    
                    for series_ID in series_IDs:
                        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(normalized_path, series_ID)
                        if len(dicom_names) > max_file_count:
                            max_file_count = len(dicom_names)
                            best_series = series_ID
                    
                    if best_series and max_file_count > 0:
                        dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(normalized_path, best_series)
                        logging.info(f"Loading DICOM series with {len(dicom_names)} files")
                        reader.SetFileNames(dicom_names)
                        try:
                            image = reader.Execute()
                            logging.info(f"Successfully loaded DICOM series: {image.GetSize()[0]}x{image.GetSize()[1]}x{image.GetSize()[2]}")
                            return image
                        except Exception as e:
                            logging.warning(f"Error executing reader for series {best_series}: {str(e)}")
            except Exception as e:
                logging.warning(f"Error using GDCM to read DICOM series: {str(e)}")
            
            # If GDCM method fails, try manual approach with sorting
            logging.info("Trying alternative approach for DICOM series loading")
            
            # Sort DICOM files based on slice position
            sorted_dicom_files = []
            
            try:
                if PYDICOM_AVAILABLE:
                    sorted_dicom_info = []
                    for file_path in dicom_files:
                        try:
                            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                            
                            # Get a position value for sorting
                            position = None
                            
                            # Try to get instance number
                            if hasattr(ds, 'InstanceNumber'):
                                position = float(ds.InstanceNumber)
                            # Try to get slice location
                            elif hasattr(ds, 'SliceLocation'):
                                position = float(ds.SliceLocation)
                            # Try to get image position patient (z coordinate)
                            elif hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
                                position = float(ds.ImagePositionPatient[2])
                            # Use unique ID as last resort
                            elif hasattr(ds, 'SOPInstanceUID'):
                                position = ds.SOPInstanceUID
                            else:
                                position = file_path  # Use file path as fallback
                            
                            # Only add files with pixel data or certain attributes
                            if 'PixelData' in ds or hasattr(ds, 'Rows'):
                                # Also check for matching width and height across slices
                                if hasattr(ds, 'Rows') and hasattr(ds, 'Columns'):
                                    # Store position, file path, and dimensions
                                    sorted_dicom_info.append((position, file_path, ds.Rows, ds.Columns))
                        except Exception as e:
                            logging.debug(f"Error reading DICOM file {file_path}: {str(e)}")
                    
                    # Group by dimensions to find the largest matching set
                    dimension_groups = {}
                    for info in sorted_dicom_info:
                        position, file_path, rows, cols = info
                        key = (rows, cols)
                        if key not in dimension_groups:
                            dimension_groups[key] = []
                        dimension_groups[key].append((position, file_path))
                    
                    # Find the largest group
                    max_group_size = 0
                    largest_group = None
                    for dim, group in dimension_groups.items():
                        if len(group) > max_group_size:
                            max_group_size = len(group)
                            largest_group = group
                    
                    if largest_group:
                        # Sort by position
                        try:
                            largest_group.sort(key=lambda x: x[0] if isinstance(x[0], (int, float)) else 0)
                        except Exception:
                            # If sorting fails, keep original order
                            pass
                        
                        # Extract file paths
                        sorted_dicom_files = [info[1] for info in largest_group]
                        logging.info(f"Found {len(sorted_dicom_files)} DICOM files with matching dimensions")
                    else:
                        # If grouping fails, use all files
                        sorted_dicom_files = dicom_files
                else:
                    # Without pydicom, just use all files
                    sorted_dicom_files = dicom_files
            except Exception as e:
                logging.warning(f"Error sorting DICOM files: {str(e)}")
                # Use unsorted files if sorting fails
                sorted_dicom_files = dicom_files
            
            if not sorted_dicom_files:
                raise ValueError(f"No valid DICOM files found in directory: {normalized_path}")
            
            logging.info(f"Attempting to load {len(sorted_dicom_files)} sorted DICOM files")
            
            # Load the sorted DICOM files
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(sorted_dicom_files)
            
            try:
                image = reader.Execute()
                size = image.GetSize()
                if size[0] > 0 and size[1] > 0 and size[2] > 0:
                    logging.info(f"Successfully loaded DICOM series: {size[0]}x{size[1]}x{size[2]}")
                    return image
                else:
                    raise ValueError("Empty image returned")
            except Exception as e:
                logging.error(f"Error loading sorted DICOM files: {str(e)}")
                
                # If all else fails, try reading just the first file
                if sorted_dicom_files:
                    logging.info("Trying to load single DICOM file")
                    try:
                        return sitk.ReadImage(sorted_dicom_files[0])
                    except Exception as e2:
                        logging.error(f"Error loading single DICOM file: {str(e2)}")
                
                raise ValueError(f"Failed to load DICOM series from {normalized_path}: {str(e)}")
        else:
            # Not a file or directory
            raise ValueError(f"Path is neither a file nor a directory: {normalized_path}")
            
    except ImportError:
        logging.critical("SimpleITK not available. Cannot load medical images.")
        raise ValueError("SimpleITK library is required to load medical images.")
    except Exception as e:
        error_msg = f"Error loading medical image {file_path}: {str(e)}"
        logging.error(error_msg)
        # Raise error instead of returning placeholder to allow proper error handling
        raise ValueError(error_msg) from e


def save_medical_image(image_data: Any, output_path: str, format: str = "nifti") -> str:
    """
    Save a medical image file.
    
    This is a placeholder. In a real implementation, this would use libraries like
    SimpleITK, nibabel, or pydicom to save the image data.
    
    Args:
        image_data: Image data to save
        output_path: Path to save the image
        format: Output format (nifti, dicom)
        
    Returns:
        Path to saved file
    """
    # Ensure output directory exists
    output_path = ensure_output_dir(output_path)
    
    # Add appropriate extension if not present
    if format.lower() == "nifti" and not output_path.lower().endswith(('.nii', '.nii.gz')):
        output_path += '.nii.gz'
    elif format.lower() == "dicom" and not output_path.lower().endswith('.dcm'):
        output_path += '.dcm'
    
    logging.info(f"Saving medical image to: {output_path}")
    
    # Placeholder for actual save logic
    # In a real implementation, this would use appropriate libraries
    # based on the output format
    
    # Simulate file creation for testing
    with open(output_path, 'w') as f:
        f.write("Placeholder for image data")
    
    return output_path


def load_dicom_series(directory_path: str) -> Any:
    """
    Load a DICOM series from a directory.
    
    This is a placeholder. In a real implementation, this would use libraries like
    pydicom or SimpleITK to load the DICOM series.
    
    Args:
        directory_path: Path to directory containing DICOM files
        
    Returns:
        Loaded image data
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    logging.info(f"Loading DICOM series from: {directory_path}")
    
    # Placeholder return
    return {"directory_path": directory_path, "data": None}


def load_nifti(file_path: str) -> Any:
    """
    Load a NIfTI file.
    
    This is a placeholder. In a real implementation, this would use libraries like
    nibabel or SimpleITK to load the NIfTI file.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Loaded image data
    """
    if not validate_input_file(file_path):
        raise ValueError(f"Invalid input file: {file_path}")
    
    logging.info(f"Loading NIfTI file: {file_path}")
    
    # Placeholder return
    return {"file_path": file_path, "data": None}


def save_as_nifti(image_data: Any, output_path: str) -> str:
    """
    Save image data as a NIfTI file.
    
    This is a placeholder. In a real implementation, this would use libraries like
    nibabel or SimpleITK to save the image data.
    
    Args:
        image_data: Image data to save
        output_path: Path to save the image
        
    Returns:
        Path to saved file
    """
    return save_medical_image(image_data, output_path, format="nifti")


def save_as_dicom(image_data: Any, output_path: str) -> str:
    """
    Save image data as a DICOM file.
    
    This is a placeholder. In a real implementation, this would use libraries like
    pydicom or SimpleITK to save the image data.
        
        Args:
        image_data: Image data to save
        output_path: Path to save the image
            
        Returns:
        Path to saved file
        """
    return save_medical_image(image_data, output_path, format="dicom") 