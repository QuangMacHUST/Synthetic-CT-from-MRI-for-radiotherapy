#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
DICOM utilities for MRI to CT conversion.
This module provides functions for handling DICOM files, including loading/saving DICOM series,
extracting metadata, and converting between DICOM and SimpleITK formats.
"""

import os
import logging
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian
import SimpleITK as sitk

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import load_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


def load_dicom_series(dicom_dir: str, series_id: Optional[str] = None) -> sitk.Image:
    """
    Load DICOM series from directory.
    
    Args:
        dicom_dir: Directory containing DICOM files
        series_id: Specific series ID to load (if multiple series in directory)
        
    Returns:
        SimpleITK image
    """
    logger.info(f"Loading DICOM series from {dicom_dir}")
    
    try:
        # Create DICOM reader
        reader = sitk.ImageSeriesReader()
        
        # Get series IDs
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
        
        # If series_id not specified and multiple series found, use first one
        if series_id is None:
            if len(series_ids) > 1:
                logger.warning(f"Multiple series found in {dicom_dir}. Using first series.")
            series_id = series_ids[0]
        elif series_id not in series_ids:
            raise ValueError(f"Series ID {series_id} not found in {dicom_dir}")
        
        # Get filenames for the series
        dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        
        # Set filenames and load series
        reader.SetFileNames(dicom_filenames)
        image = reader.Execute()
        
        logger.info(f"Loaded DICOM series with dimensions {image.GetSize()}")
        return image
    
    except Exception as e:
        logger.error(f"Error loading DICOM series: {str(e)}")
        raise


def load_dicom_file(file_path: str) -> sitk.Image:
    """
    Load a single DICOM file.
    
    Args:
        file_path: Path to DICOM file
        
    Returns:
        SimpleITK image
    """
    return sitk.ReadImage(file_path)


def get_dicom_series_list(root_dir: str) -> List[Dict[str, Any]]:
    """
    Get list of DICOM series in a directory tree.
    
    Args:
        root_dir: Root directory to search for DICOM series
        
    Returns:
        List of dictionaries with series information
    """
    logger.info(f"Searching for DICOM series in {root_dir}")
    
    series_list = []
    
    try:
        # Create DICOM reader
        reader = sitk.ImageSeriesReader()
        
        # Walk through directory tree
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Skip if no files
            if not filenames:
                continue
            
            # Check if directory contains DICOM files
            try:
                series_ids = reader.GetGDCMSeriesIDs(dirpath)
                
                # Process each series
                for series_id in series_ids:
                    try:
                        # Get metadata
                        metadata = load_dicom_metadata(dirpath, series_id)
                        
                        # Add directory path
                        metadata['directory'] = dirpath
                        
                        # Add to list
                        series_list.append(metadata)
                    except Exception as e:
                        logger.warning(f"Error getting metadata for series {series_id} in {dirpath}: {str(e)}")
            
            except Exception:
                # Not a DICOM directory or error reading DICOM
                pass
        
        logger.info(f"Found {len(series_list)} DICOM series")
        return series_list
    
    except Exception as e:
        logger.error(f"Error searching for DICOM series: {str(e)}")
        raise


def load_dicom_metadata(dicom_dir: str, series_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DICOM metadata from a series.
    
    Args:
        dicom_dir: Directory containing DICOM files
        series_id: Specific series ID to load (if multiple series in directory)
        
    Returns:
        Dictionary containing DICOM metadata
    """
    logger.info(f"Loading DICOM metadata from {dicom_dir}")
    
    try:
        # Get series IDs
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
        
        # If series_id not specified and multiple series found, use first one
        if series_id is None:
            if len(series_ids) > 1:
                logger.warning(f"Multiple series found in {dicom_dir}. Using first series.")
            series_id = series_ids[0]
        elif series_id not in series_ids:
            raise ValueError(f"Series ID {series_id} not found in {dicom_dir}")
        
        # Get filenames for the series
        dicom_filenames = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        
        # Read metadata from first DICOM file
        ds = pydicom.dcmread(dicom_filenames[0], stop_before_pixels=True)
        
        # Extract relevant metadata
        metadata = {
            'series_id': series_id,
            'patient_id': getattr(ds, 'PatientID', 'UNKNOWN'),
            'patient_name': str(getattr(ds, 'PatientName', 'UNKNOWN')),
            'patient_sex': getattr(ds, 'PatientSex', 'UNKNOWN'),
            'patient_age': getattr(ds, 'PatientAge', 'UNKNOWN'),
            'patient_birthdate': getattr(ds, 'PatientBirthDate', 'UNKNOWN'),
            'study_date': getattr(ds, 'StudyDate', 'UNKNOWN'),
            'study_time': getattr(ds, 'StudyTime', 'UNKNOWN'),
            'study_description': getattr(ds, 'StudyDescription', 'UNKNOWN'),
            'series_description': getattr(ds, 'SeriesDescription', 'UNKNOWN'),
            'modality': getattr(ds, 'Modality', 'UNKNOWN'),
            'manufacturer': getattr(ds, 'Manufacturer', 'UNKNOWN'),
            'scanner_model': getattr(ds, 'ManufacturerModelName', 'UNKNOWN'),
            'institution_name': getattr(ds, 'InstitutionName', 'UNKNOWN'),
            'slice_thickness': getattr(ds, 'SliceThickness', 0),
            'spacing_between_slices': getattr(ds, 'SpacingBetweenSlices', 0),
            'rows': getattr(ds, 'Rows', 0),
            'columns': getattr(ds, 'Columns', 0),
            'number_of_files': len(dicom_filenames)
        }
        
        # Try to get pixel spacing
        if hasattr(ds, 'PixelSpacing'):
            metadata['pixel_spacing'] = [float(ps) for ps in ds.PixelSpacing]
        
        logger.info(f"Loaded metadata for DICOM series: {metadata['series_description']}")
        return metadata
    
    except Exception as e:
        logger.error(f"Error loading DICOM metadata: {str(e)}")
        raise


def extract_dicom_tags(dicom_file: str) -> Dict[str, Any]:
    """
    Extract all tags from a DICOM file.
    
    Args:
        dicom_file: Path to DICOM file
        
    Returns:
        Dictionary of DICOM tags
    """
    try:
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        tags = {}
        
        for elem in ds:
            if elem.keyword:
                try:
                    value = elem.value
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', 'ignore')
                    tags[elem.keyword] = value
                except Exception:
                    # Skip tags that cannot be converted or decoded
                    pass
        
        return tags
    
    except Exception as e:
        logger.error(f"Error extracting DICOM tags: {str(e)}")
        raise


def save_dicom_series(image: Union[sitk.Image, SyntheticCT], 
                    output_dir: str, 
                    reference_dicom: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    series_description: Optional[str] = None) -> str:
    """
    Save SimpleITK image as DICOM series.
    
    Args:
        image: SimpleITK image or SyntheticCT object
        output_dir: Directory to save DICOM files
        reference_dicom: Directory or file containing reference DICOM to copy metadata from
        metadata: Dictionary of metadata to include in DICOM
        series_description: Description for the series
        
    Returns:
        Path to the saved DICOM series
    """
    logger.info(f"Saving DICOM series to {output_dir}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get SimpleITK image from SyntheticCT if needed
        if isinstance(image, SyntheticCT):
            sitk_image = image.image
            metadata_dict = image.metadata.copy()
            
            # Update metadata with provided metadata
            if metadata:
                metadata_dict.update(metadata)
        else:
            sitk_image = image
            metadata_dict = metadata or {}
        
        # Load reference DICOM if provided
        reference_tags = {}
        if reference_dicom:
            if os.path.isdir(reference_dicom):
                # Load metadata from directory
                reference_metadata = load_dicom_metadata(reference_dicom)
                
                # Get first file in series
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(reference_dicom)
                if series_ids:
                    dicom_filenames = reader.GetGDCMSeriesFileNames(reference_dicom, series_ids[0])
                    if dicom_filenames:
                        reference_tags = extract_dicom_tags(dicom_filenames[0])
            else:
                # Load metadata from file
                reference_tags = extract_dicom_tags(reference_dicom)
        
        # Generate new UIDs
        study_instance_uid = generate_uid()
        series_instance_uid = generate_uid()
        frame_of_reference_uid = generate_uid()
        
        # Use reference values if available
        if 'StudyInstanceUID' in reference_tags:
            study_instance_uid = reference_tags['StudyInstanceUID']
        
        # Set series description
        if series_description is None:
            modality = metadata_dict.get('modality', 'CT')
            is_synthetic = metadata_dict.get('conversion', {}).get('method', 'unknown')
            series_description = f"Synthetic {modality} ({is_synthetic})"
        
        # Get image properties
        size = sitk_image.GetSize()
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        # Extract pixel array
        pixel_array = sitk.GetArrayFromImage(sitk_image)
        
        # Ensure pixel array has correct orientation (z, y, x)
        if pixel_array.shape[0] != size[2]:
            pixel_array = np.transpose(pixel_array, (2, 0, 1))
        
        # Create DICOM writer
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        
        # Define modality
        modality = metadata_dict.get('modality', 'CT')
        
        # Create DICOM series
        slice_filenames = []
        
        for i in range(pixel_array.shape[0]):
            # Get slice pixel array
            slice_array = pixel_array[i, :, :].astype(np.int16)
            
            # Create new DICOM dataset
            ds = Dataset()
            
            # Copy reference tags if available
            for key, value in reference_tags.items():
                if key not in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                              'FrameOfReferenceUID', 'SeriesDescription']:
                    try:
                        ds.__setattr__(key, value)
                    except:
                        # Skip tags that can't be set
                        pass
            
            # Create file meta
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            
            # Add file meta to dataset
            ds.file_meta = file_meta
            
            # Set essential DICOM attributes
            ds.StudyInstanceUID = study_instance_uid
            ds.SeriesInstanceUID = series_instance_uid
            ds.SOPInstanceUID = generate_uid()
            ds.FrameOfReferenceUID = frame_of_reference_uid
            
            # Add patient information
            ds.PatientID = metadata_dict.get('patient_id', 'ANONYMOUS')
            ds.PatientName = metadata_dict.get('patient_name', 'ANONYMOUS')
            ds.PatientSex = metadata_dict.get('patient_sex', 'O')
            
            # Add study information
            current_date = datetime.datetime.now().strftime('%Y%m%d')
            current_time = datetime.datetime.now().strftime('%H%M%S')
            ds.StudyDate = metadata_dict.get('study_date', current_date)
            ds.StudyTime = metadata_dict.get('study_time', current_time)
            ds.StudyDescription = metadata_dict.get('study_description', 'Synthetic CT Study')
            
            # Add series information
            ds.SeriesDate = current_date
            ds.SeriesTime = current_time
            ds.SeriesDescription = series_description
            ds.Modality = modality
            
            # Add acquisition information
            ds.AcquisitionDate = current_date
            ds.AcquisitionTime = current_time
            
            # Add image information
            ds.Rows = size[1]
            ds.Columns = size[0]
            ds.PixelSpacing = [spacing[0], spacing[1]]
            ds.SliceThickness = spacing[2]
            ds.SpacingBetweenSlices = spacing[2]
            ds.ImagePositionPatient = [origin[0], origin[1], origin[2] + i * spacing[2]]
            ds.ImageOrientationPatient = [direction[0], direction[1], direction[2],
                                         direction[3], direction[4], direction[5]]
            
            # Set slice location
            ds.SliceLocation = origin[2] + i * spacing[2]
            
            # Set instance number
            ds.InstanceNumber = i + 1
            
            # Set window center and width for CT
            if modality == 'CT':
                ds.WindowCenter = metadata_dict.get('window_center', 40)
                ds.WindowWidth = metadata_dict.get('window_width', 400)
                ds.RescaleIntercept = 0
                ds.RescaleSlope = 1
                ds.RescaleType = 'HU'
            
            # Set pixel data
            ds.PixelData = slice_array.tobytes()
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 1  # Signed
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            
            # Save DICOM file
            filename = os.path.join(output_dir, f"slice_{i+1:04d}.dcm")
            ds.save_as(filename)
            slice_filenames.append(filename)
            
            if i % 10 == 0:
                logger.info(f"Saved {i+1}/{pixel_array.shape[0]} DICOM slices")
        
        logger.info(f"Saved DICOM series with {len(slice_filenames)} slices to {output_dir}")
        return output_dir
    
    except Exception as e:
        logger.error(f"Error saving DICOM series: {str(e)}")
        raise


def convert_dicom_to_ct(dicom_dir: str, is_reference: bool = False) -> SyntheticCT:
    """
    Convert DICOM series to CT object.
    
    Args:
        dicom_dir: Directory containing DICOM files
        is_reference: Whether this is a reference CT
        
    Returns:
        SyntheticCT object containing CT image and metadata
    """
    logger.info(f"Converting DICOM series to CT object from {dicom_dir}")
    
    try:
        # Load DICOM series
        image = load_dicom_series(dicom_dir)
        
        # Load metadata
        metadata = load_dicom_metadata(dicom_dir)
        
        # Add reference flag to metadata
        metadata['is_reference'] = is_reference
        
        # Create SyntheticCT object
        synthetic_ct = SyntheticCT(image, metadata)
        
        logger.info(f"Converted DICOM series to CT object: {metadata.get('series_description', 'UNKNOWN')}")
        return synthetic_ct
    
    except Exception as e:
        logger.error(f"Error converting DICOM to CT object: {str(e)}")
        raise


def convert_ct_to_dicom(synthetic_ct: SyntheticCT, output_dir: str, reference_dicom: Optional[str] = None) -> str:
    """
    Convert CT object to DICOM series.
    
    Args:
        synthetic_ct: SyntheticCT object
        output_dir: Directory to save DICOM files
        reference_dicom: Directory or file containing reference DICOM to copy metadata from
        
    Returns:
        Path to the saved DICOM series
    """
    logger.info(f"Converting CT object to DICOM series")
    
    try:
        # Extract metadata
        metadata = synthetic_ct.metadata.copy()
        
        # Determine series description
        series_description = metadata.get('series_description', None)
        
        if series_description is None:
            # Create series description based on conversion method
            conversion_info = metadata.get('conversion', {})
            method = conversion_info.get('method', 'unknown')
            region = conversion_info.get('region', 'unknown')
            series_description = f"Synthetic CT ({method.upper()}, {region.capitalize()})"
        
        # Set modality to CT
        metadata['modality'] = 'CT'
        
        # Save as DICOM series
        dicom_dir = save_dicom_series(
            synthetic_ct,
            output_dir,
            reference_dicom=reference_dicom,
            metadata=metadata,
            series_description=series_description
        )
        
        logger.info(f"Converted CT object to DICOM series: {series_description}")
        return dicom_dir
    
    except Exception as e:
        logger.error(f"Error converting CT object to DICOM: {str(e)}")
        raise


def create_synthetic_dicom_series(mri_dicom_dir: str, output_dir: str, region: str = 'head',
                               model_type: str = 'gan') -> Tuple[str, SyntheticCT]:
    """
    Create synthetic CT DICOM series from MRI DICOM series.
    
    Args:
        mri_dicom_dir: Directory containing MRI DICOM files
        output_dir: Directory to save synthetic CT DICOM files
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        model_type: Conversion method ('atlas', 'cnn', or 'gan')
        
    Returns:
        Tuple of (path to saved DICOM series, SyntheticCT object)
    """
    logger.info(f"Creating synthetic CT DICOM series from MRI in {mri_dicom_dir}")
    
    try:
        # Load MRI DICOM series
        mri_image = load_dicom_series(mri_dicom_dir)
        mri_metadata = load_dicom_metadata(mri_dicom_dir)
        
        # Create SyntheticCT object for MRI
        mri_synth = SyntheticCT(mri_image, mri_metadata)
        
        # Preprocess MRI
        from app.core.preprocessing.preprocess import preprocess_mri
        preprocessed_mri = preprocess_mri(mri_synth)
        
        # Segment tissues
        from app.core.segmentation.segment_tissues import segment_tissues
        segmentation = segment_tissues(preprocessed_mri, method='auto', region=region)
        
        # Convert MRI to CT
        from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct
        synthetic_ct = convert_mri_to_ct(preprocessed_mri, segmentation, model_type=model_type, region=region)
        
        # Save as DICOM series
        series_description = f"Synthetic CT ({model_type.upper()}, {region.capitalize()})"
        dicom_dir = save_dicom_series(
            synthetic_ct,
            output_dir,
            reference_dicom=mri_dicom_dir,
            series_description=series_description
        )
        
        logger.info(f"Created synthetic CT DICOM series in {dicom_dir}")
        return dicom_dir, synthetic_ct
    
    except Exception as e:
        logger.error(f"Error creating synthetic CT DICOM series: {str(e)}")
        raise


def anonymize_dicom(input_dir: str, output_dir: str, 
                  patient_id: Optional[str] = None,
                  patient_name: Optional[str] = None) -> str:
    """
    Anonymize DICOM series.
    
    Args:
        input_dir: Directory containing DICOM files
        output_dir: Directory to save anonymized DICOM files
        patient_id: New patient ID (if None, generates random ID)
        patient_name: New patient name (if None, uses 'ANONYMOUS')
        
    Returns:
        Path to the saved anonymized DICOM series
    """
    logger.info(f"Anonymizing DICOM series from {input_dir}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate random patient ID if not provided
        if patient_id is None:
            import uuid
            patient_id = str(uuid.uuid4())[:8]
        
        # Set default patient name if not provided
        if patient_name is None:
            patient_name = 'ANONYMOUS'
        
        # Get series IDs
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(input_dir)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {input_dir}")
        
        # Process each series
        for series_id in series_ids:
            # Get filenames for the series
            dicom_filenames = reader.GetGDCMSeriesFileNames(input_dir, series_id)
            
            # Generate new UIDs
            study_instance_uid = generate_uid()
            series_instance_uid = generate_uid()
            frame_of_reference_uid = generate_uid()
            
            # Process each file
            for i, filename in enumerate(dicom_filenames):
                # Read DICOM file
                ds = pydicom.dcmread(filename)
                
                # Anonymize patient information
                ds.PatientID = patient_id
                ds.PatientName = patient_name
                ds.PatientBirthDate = ''
                
                # Remove all curves and overlays
                for tag in ds.dir():
                    if tag.startswith('Curve') or tag.startswith('Overlay'):
                        delattr(ds, tag)
                
                # Remove private tags
                ds.remove_private_tags()
                
                # Update UIDs
                ds.StudyInstanceUID = study_instance_uid
                ds.SeriesInstanceUID = series_instance_uid
                ds.SOPInstanceUID = generate_uid()
                if hasattr(ds, 'FrameOfReferenceUID'):
                    ds.FrameOfReferenceUID = frame_of_reference_uid
                
                # Save anonymized file
                output_filename = os.path.join(output_dir, f"anon_{i+1:04d}.dcm")
                ds.save_as(output_filename)
                
                if i % 10 == 0:
                    logger.info(f"Anonymized {i+1}/{len(dicom_filenames)} DICOM files")
        
        logger.info(f"Anonymized DICOM series saved to {output_dir}")
        return output_dir
    
    except Exception as e:
        logger.error(f"Error anonymizing DICOM series: {str(e)}")
        raise


def dicom_to_nifti(dicom_dir: str, output_path: Optional[str] = None) -> str:
    """
    Convert a DICOM series to NIfTI format.
    
    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Path to save the NIfTI file. If None, a temporary file is created.
        
    Returns:
        Path to the saved NIfTI file
    """
    # Load DICOM series
    image = load_dicom_series(dicom_dir)
    
    # Create output path if not provided
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, f"dicom_to_nifti_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.nii.gz")
    
    # Save as NIfTI
    sitk.WriteImage(image, output_path)
    
    return output_path


def nifti_to_dicom(
    nifti_path: str, 
    output_dir: str, 
    modality: str = "CT",
    series_description: str = "Converted from NIfTI",
    derive_from_dicom: Optional[str] = None
) -> str:
    """
    Convert a NIfTI file to DICOM series.
    
    Args:
        nifti_path: Path to NIfTI file
        output_dir: Directory to save the DICOM series
        modality: DICOM modality (CT, MR, etc.)
        series_description: Description of the series
        derive_from_dicom: Path to a DICOM file or directory to derive metadata from
        
    Returns:
        Path to the saved DICOM series directory
    """
    # Load NIfTI file
    image = sitk.ReadImage(nifti_path)
    
    # Save as DICOM series
    return save_dicom_series(
        image=image,
        output_dir=output_dir,
        modality=modality,
        series_description=series_description,
        derive_from_dicom=derive_from_dicom
    )


def anonymize_dicom_directory(input_dir: str, output_dir: str) -> None:
    """
    Anonymize all DICOM files in a directory.
    
    Args:
        input_dir: Path to input directory containing DICOM files
        output_dir: Path to save anonymized DICOM files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of DICOM files
    dicom_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    # Generate consistent UIDs for the study and series
    study_uid = generate_uid()
    series_uid = generate_uid()
    
    # Anonymize each file
    for file_path in dicom_files:
        # Create output path
        rel_path = os.path.relpath(file_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # Read the DICOM file
            ds = pydicom.dcmread(file_path)
            
            # Anonymize patient information
            ds.PatientName = "ANONYMOUS"
            ds.PatientID = "ANONYMOUS"
            ds.PatientBirthDate = ""
            ds.PatientSex = ""
            
            # Set consistent UIDs
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = generate_uid()
            if 'MediaStorageSOPInstanceUID' in ds.file_meta:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            
            # Remove other identifying information
            for tag in [
                'PatientAddress', 'PatientTelephoneNumbers', 'PatientMotherBirthName',
                'PatientBirthName', 'PatientReligiousPreference', 'PatientComments',
                'OtherPatientIDs', 'OtherPatientNames', 'OtherPatientIDsSequence'
            ]:
                if tag in ds:
                    delattr(ds, tag)
            
            # Save the anonymized file
            ds.save_as(output_path)
            
        except Exception as e:
            logger.warning(f"Could not anonymize {file_path}: {str(e)}")


def find_dicom_files(directory_path: str, recursive: bool = True) -> List[str]:
    """
    Find DICOM files in a directory.
    
    Args:
        directory_path: Path to directory containing DICOM files
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of paths to DICOM files
    """
    logger.info(f"Searching for DICOM files in {directory_path}")
    
    dicom_files = []
    
    try:
        if recursive:
            # Walk through directory tree
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Try to read as DICOM
                        pydicom.dcmread(file_path, stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        # Not a DICOM file or error reading
                        continue
        else:
            # Only search in the given directory
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    try:
                        # Try to read as DICOM
                        pydicom.dcmread(file_path, stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        # Not a DICOM file or error reading
                        continue
        
        # Sort files to ensure consistent ordering
        dicom_files.sort()
        
        logger.info(f"Found {len(dicom_files)} DICOM files in {directory_path}")
        return dicom_files
    
    except Exception as e:
        logger.error(f"Error searching for DICOM files: {str(e)}")
        raise ValueError(f"Failed to search for DICOM files in {directory_path}: {str(e)}") 