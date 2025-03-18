#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Patient data management utilities.
"""

import os
import json
import shutil
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid

# Set up logger
logger = logging.getLogger(__name__)

# Constants
DATA_STORE_ENV_VAR = "SYNTHETIC_CT_DATA_STORE"
DEFAULT_DATA_STORE = os.path.join(os.path.expanduser("~"), ".synthetic_ct", "patients")


def get_data_store_path() -> str:
    """
    Get path to data store directory.
    
    Returns:
        Path to data store directory
    """
    data_store_path = os.environ.get(DATA_STORE_ENV_VAR, DEFAULT_DATA_STORE)
    os.makedirs(data_store_path, exist_ok=True)
    return data_store_path


def get_patient_dir(patient_id: str) -> str:
    """
    Get path to patient directory.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Path to patient directory
    """
    data_store_path = get_data_store_path()
    patient_dir = os.path.join(data_store_path, patient_id)
    return patient_dir


def patient_exists(patient_id: str) -> bool:
    """
    Check if patient exists in data store.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        True if patient exists, False otherwise
    """
    patient_dir = get_patient_dir(patient_id)
    return os.path.isdir(patient_dir)


def extract_patient_info(dicom_file: Union[str, Dataset]) -> Dict[str, str]:
    """
    Extract patient information from DICOM file.
    
    Args:
        dicom_file: Path to DICOM file or PyDicom Dataset
        
    Returns:
        Dictionary containing patient information
    """
    try:
        if isinstance(dicom_file, str):
            ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)
        else:
            ds = dicom_file
            
        # Extract patient information
        patient_info = {
            "patient_id": ds.PatientID if hasattr(ds, "PatientID") else "UNKNOWN",
            "name": str(ds.PatientName) if hasattr(ds, "PatientName") else "UNKNOWN",
            "birth_date": ds.PatientBirthDate if hasattr(ds, "PatientBirthDate") else "",
            "sex": ds.PatientSex if hasattr(ds, "PatientSex") else "",
            "study_date": ds.StudyDate if hasattr(ds, "StudyDate") else "",
            "study_description": ds.StudyDescription if hasattr(ds, "StudyDescription") else "",
            "modality": ds.Modality if hasattr(ds, "Modality") else "UNKNOWN",
            "accession_number": ds.AccessionNumber if hasattr(ds, "AccessionNumber") else "",
            "study_instance_uid": ds.StudyInstanceUID if hasattr(ds, "StudyInstanceUID") else generate_uid(),
            "series_instance_uid": ds.SeriesInstanceUID if hasattr(ds, "SeriesInstanceUID") else generate_uid()
        }
        
        return patient_info
        
    except Exception as e:
        logger.error(f"Error extracting patient info: {str(e)}")
        return {
            "patient_id": "UNKNOWN",
            "name": "UNKNOWN",
            "modality": "UNKNOWN"
        }


def anonymize_dicom_file(source_file: str, target_file: str) -> None:
    """
    Anonymize DICOM file.
    
    Args:
        source_file: Path to source DICOM file
        target_file: Path to target DICOM file
    """
    try:
        # Read DICOM file
        ds = pydicom.dcmread(source_file)
        
        # Anonymize patient information
        ds.PatientName = "Anonymous"
        ds.PatientID = f"ANON{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Remove birth date
        if hasattr(ds, "PatientBirthDate"):
            delattr(ds, "PatientBirthDate")
            
        # Remove other identifying information
        for tag in ["OtherPatientIDs", "OtherPatientIDsSequence", "OtherPatientNames"]:
            if hasattr(ds, tag):
                delattr(ds, tag)
                
        # Generate new UIDs
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        
        # Save anonymized file
        ds.save_as(target_file)
        
    except Exception as e:
        logger.error(f"Error anonymizing DICOM file: {str(e)}")
        raise


def import_dicom_directory(source_dir: str, target_dir: str, anonymize: bool = False) -> List[str]:
    """
    Import DICOM directory.
    
    Args:
        source_dir: Path to source directory
        target_dir: Path to target directory
        anonymize: Whether to anonymize DICOM files
        
    Returns:
        List of imported file paths
    """
    imported_files = []
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Walk source directory
        for root, _, files in os.walk(source_dir):
            for file in files:
                source_file = os.path.join(root, file)
                
                # Check if file is DICOM
                try:
                    pydicom.dcmread(source_file, stop_before_pixels=True)
                except:
                    continue
                
                # Create relative path
                rel_path = os.path.relpath(source_file, source_dir)
                target_file = os.path.join(target_dir, rel_path)
                
                # Create target directory
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                
                # Copy or anonymize file
                if anonymize:
                    anonymize_dicom_file(source_file, target_file)
                else:
                    shutil.copy2(source_file, target_file)
                    
                imported_files.append(target_file)
                
        return imported_files
        
    except Exception as e:
        logger.error(f"Error importing DICOM directory: {str(e)}")
        raise


def detect_modality(file_path: str) -> str:
    """
    Detect modality from file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Detected modality
    """
    # Check file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".dcm":
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            return ds.Modality if hasattr(ds, "Modality") else "UNKNOWN"
        except:
            return "UNKNOWN"
    elif ext in [".nii", ".nii.gz"]:
        return "NIfTI"
    elif ext in [".mha", ".mhd"]:
        return "MHA"
    else:
        return "UNKNOWN"


def import_patient_data(source_path: str, anonymize: bool = False, modalities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Import patient data.
    
    Args:
        source_path: Path to source file or directory
        anonymize: Whether to anonymize patient data
        modalities: List of modalities to import (None for all)
        
    Returns:
        Dictionary containing import results
    """
    try:
        # Check if source path exists
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
            
        # If source is a file, get parent directory
        if os.path.isfile(source_path):
            source_dir = os.path.dirname(source_path)
            source_files = [source_path]
        else:
            source_dir = source_path
            source_files = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    source_files.append(os.path.join(root, file))
        
        # Extract patient information from first DICOM file
        patient_info = None
        for file in source_files:
            try:
                if file.lower().endswith(".dcm"):
                    patient_info = extract_patient_info(file)
                    break
            except:
                continue
                
        # If no DICOM files found, use source directory name as patient ID
        if patient_info is None or patient_info["patient_id"] == "UNKNOWN":
            patient_id = os.path.basename(source_dir)
            patient_info = {
                "patient_id": patient_id,
                "name": patient_id,
                "modality": "UNKNOWN"
            }
            
        # Create patient directory
        patient_dir = get_patient_dir(patient_info["patient_id"])
        os.makedirs(patient_dir, exist_ok=True)
        
        # Save patient information
        with open(os.path.join(patient_dir, "patient_info.json"), "w") as f:
            json.dump(patient_info, f, indent=2)
            
        # Import files
        if os.path.isdir(source_path):
            # Import DICOM directory
            target_dir = os.path.join(patient_dir, "dicom")
            imported_files = import_dicom_directory(source_path, target_dir, anonymize)
        else:
            # Import single file
            modality = detect_modality(source_path)
            
            # Skip if modality not in list
            if modalities and modality not in modalities:
                return {
                    "patient_id": patient_info["patient_id"],
                    "imported_files": []
                }
                
            # Create directory for modality
            target_dir = os.path.join(patient_dir, modality.lower())
            os.makedirs(target_dir, exist_ok=True)
            
            # Generate target file path
            target_file = os.path.join(target_dir, os.path.basename(source_path))
            
            # Copy or anonymize file
            if anonymize and source_path.lower().endswith(".dcm"):
                anonymize_dicom_file(source_path, target_file)
            else:
                shutil.copy2(source_path, target_file)
                
            imported_files = [target_file]
            
        return {
            "patient_id": patient_info["patient_id"],
            "imported_files": imported_files
        }
        
    except Exception as e:
        logger.error(f"Error importing patient data: {str(e)}")
        raise


def export_patient_data(patient_id: str, output_dir: str, anonymize: bool = False, 
                       modalities: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Export patient data.
    
    Args:
        patient_id: Patient ID
        output_dir: Path to output directory
        anonymize: Whether to anonymize patient data
        modalities: List of modalities to export (None for all)
        
    Returns:
        Dictionary containing export results
    """
    try:
        # Check if patient exists
        if not patient_exists(patient_id):
            raise ValueError(f"Patient not found: {patient_id}")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get patient directory
        patient_dir = get_patient_dir(patient_id)
        
        # Export patient information
        patient_info_path = os.path.join(patient_dir, "patient_info.json")
        if os.path.isfile(patient_info_path):
            if anonymize:
                # Read patient info
                with open(patient_info_path, "r") as f:
                    patient_info = json.load(f)
                    
                # Anonymize patient info
                patient_info["patient_id"] = f"ANON{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                patient_info["name"] = "Anonymous"
                patient_info["birth_date"] = ""
                
                # Write anonymized patient info
                with open(os.path.join(output_dir, "patient_info.json"), "w") as f:
                    json.dump(patient_info, f, indent=2)
            else:
                shutil.copy2(patient_info_path, os.path.join(output_dir, "patient_info.json"))
        
        # Export files
        exported_files = []
        
        for item in os.listdir(patient_dir):
            item_path = os.path.join(patient_dir, item)
            
            # Skip patient info file (already exported)
            if item == "patient_info.json":
                continue
                
            # Skip directories not in modalities list
            if modalities and item.upper() not in modalities:
                continue
                
            # Export directory
            if os.path.isdir(item_path):
                target_dir = os.path.join(output_dir, item)
                os.makedirs(target_dir, exist_ok=True)
                
                for root, _, files in os.walk(item_path):
                    for file in files:
                        source_file = os.path.join(root, file)
                        
                        # Create relative path
                        rel_path = os.path.relpath(source_file, item_path)
                        target_file = os.path.join(target_dir, rel_path)
                        
                        # Create target directory
                        os.makedirs(os.path.dirname(target_file), exist_ok=True)
                        
                        # Copy or anonymize file
                        if anonymize and file.lower().endswith(".dcm"):
                            anonymize_dicom_file(source_file, target_file)
                        else:
                            shutil.copy2(source_file, target_file)
                            
                        exported_files.append(target_file)
        
        return {
            "patient_id": patient_id,
            "exported_files": exported_files
        }
        
    except Exception as e:
        logger.error(f"Error exporting patient data: {str(e)}")
        raise


def list_patient_data(patient_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List patient data.
    
    Args:
        patient_id: Patient ID (None to list all patients)
        
    Returns:
        List of dictionaries containing patient data
    """
    try:
        data_store_path = get_data_store_path()
        
        if patient_id:
            # List data for specific patient
            patient_dir = get_patient_dir(patient_id)
            
            if not os.path.isdir(patient_dir):
                raise ValueError(f"Patient not found: {patient_id}")
                
            result = []
            
            for item in os.listdir(patient_dir):
                item_path = os.path.join(patient_dir, item)
                
                # Skip patient info file
                if item == "patient_info.json":
                    continue
                    
                # Process directory
                if os.path.isdir(item_path):
                    # Count files
                    file_count = 0
                    for root, _, files in os.walk(item_path):
                        file_count += len(files)
                        
                    # Get modification date
                    mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))
                    mod_date_str = mod_date.strftime("%Y-%m-%d %H:%M:%S")
                    
                    result.append({
                        "modality": item.upper(),
                        "file_count": file_count,
                        "date": mod_date_str,
                        "description": f"{file_count} files",
                        "path": item_path
                    })
            
            return result
            
        else:
            # List all patients
            result = []
            
            for item in os.listdir(data_store_path):
                item_path = os.path.join(data_store_path, item)
                
                # Process patient directory
                if os.path.isdir(item_path):
                    # Load patient info
                    patient_info_path = os.path.join(item_path, "patient_info.json")
                    
                    if os.path.isfile(patient_info_path):
                        try:
                            with open(patient_info_path, "r") as f:
                                patient_info = json.load(f)
                        except:
                            patient_info = {
                                "patient_id": item,
                                "name": "UNKNOWN"
                            }
                    else:
                        patient_info = {
                            "patient_id": item,
                            "name": "UNKNOWN"
                        }
                        
                    # Count studies
                    study_count = 0
                    for study_item in os.listdir(item_path):
                        study_item_path = os.path.join(item_path, study_item)
                        if os.path.isdir(study_item_path):
                            study_count += 1
                            
                    # Get modification date
                    mod_date = datetime.datetime.fromtimestamp(os.path.getmtime(item_path))
                    mod_date_str = mod_date.strftime("%Y-%m-%d %H:%M:%S")
                    
                    result.append({
                        "patient_id": patient_info.get("patient_id", item),
                        "name": patient_info.get("name", "UNKNOWN"),
                        "study_count": study_count,
                        "date": mod_date_str,
                        "path": item_path
                    })
            
            return result
        
    except Exception as e:
        logger.error(f"Error listing patient data: {str(e)}")
        raise


def delete_patient_data(patient_id: str) -> Dict[str, Any]:
    """
    Delete patient data.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Dictionary containing deletion results
    """
    try:
        # Check if patient exists
        if not patient_exists(patient_id):
            raise ValueError(f"Patient not found: {patient_id}")
            
        # Get patient directory
        patient_dir = get_patient_dir(patient_id)
        
        # Count files
        deleted_files = 0
        for root, _, files in os.walk(patient_dir):
            deleted_files += len(files)
            
        # Delete patient directory
        shutil.rmtree(patient_dir)
        
        return {
            "patient_id": patient_id,
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        logger.error(f"Error deleting patient data: {str(e)}")
        raise


def anonymize_patient_data(patient_id: str) -> Dict[str, Any]:
    """
    Anonymize patient data.
    
    Args:
        patient_id: Patient ID
        
    Returns:
        Dictionary containing anonymization results
    """
    try:
        # Check if patient exists
        if not patient_exists(patient_id):
            raise ValueError(f"Patient not found: {patient_id}")
            
        # Get patient directory
        patient_dir = get_patient_dir(patient_id)
        
        # Create temporary directory
        temp_dir = os.path.join(os.path.dirname(patient_dir), f"temp_{patient_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Export anonymized data to temporary directory
        export_result = export_patient_data(patient_id, temp_dir, anonymize=True)
        
        # Delete original patient data
        delete_patient_data(patient_id)
        
        # Import anonymized data
        import_result = import_patient_data(temp_dir)
        
        # Delete temporary directory
        shutil.rmtree(temp_dir)
        
        return {
            "patient_id": import_result["patient_id"],
            "anonymized_files": len(import_result["imported_files"])
        }
        
    except Exception as e:
        logger.error(f"Error anonymizing patient data: {str(e)}")
        raise 