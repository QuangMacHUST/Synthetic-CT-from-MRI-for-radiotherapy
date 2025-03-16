#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for preparing training datasets for MRI to CT conversion models.
This script handles data preprocessing, splitting, and augmentation.
"""

import os
import sys
import argparse
import logging
import shutil
import numpy as np
import SimpleITK as sitk
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT
from app.core.preprocessing import preprocess_mri
from app.utils.config_utils import load_config
from app.utils.dicom_utils import (
    load_dicom_series, 
    get_dicom_series_list, 
    dicom_to_nifti
)

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def find_matching_pairs(mri_dir: str, ct_dir: str, matching_criteria: str = 'patient_id') -> List[Tuple[str, str]]:
    """
    Find matching MRI and CT pairs based on specified criteria.
    
    Args:
        mri_dir: Directory containing MRI data
        ct_dir: Directory containing CT data
        matching_criteria: Criteria for matching ('patient_id', 'name', 'date')
        
    Returns:
        List of tuples (mri_path, ct_path) for matched pairs
    """
    logger.info(f"Finding matching MRI-CT pairs in {mri_dir} and {ct_dir}")
    
    # Get list of MRI and CT files/directories
    mri_paths = list(Path(mri_dir).glob('**/*'))
    ct_paths = list(Path(ct_dir).glob('**/*'))
    
    # Filter to include only directories (for DICOM) and .nii/.nii.gz files
    mri_paths = [p for p in mri_paths if p.is_dir() or p.suffix in ['.nii', '.gz']]
    ct_paths = [p for p in ct_paths if p.is_dir() or p.suffix in ['.nii', '.gz']]
    
    logger.info(f"Found {len(mri_paths)} MRI and {len(ct_paths)} CT candidates")
    
    # Extract identifiers based on matching criteria
    mri_ids = {}
    ct_ids = {}
    
    for mri_path in mri_paths:
        # If directory, assume DICOM and extract from metadata
        if mri_path.is_dir():
            try:
                # Get DICOM series list
                series_list = get_dicom_series_list(str(mri_path))
                if series_list:
                    # Just use the first series
                    series = series_list[0]
                    if matching_criteria == 'patient_id':
                        identifier = series.get('patient_id')
                    elif matching_criteria == 'name':
                        identifier = series.get('patient_name')
                    elif matching_criteria == 'date':
                        identifier = series.get('study_date')
                    else:
                        identifier = None
                    
                    if identifier:
                        mri_ids[identifier] = str(mri_path)
            except Exception as e:
                logger.warning(f"Error processing MRI directory {mri_path}: {str(e)}")
        else:
            # For NIfTI, use filename
            identifier = mri_path.stem.split('.')[0]  # Remove extension
            mri_ids[identifier] = str(mri_path)
    
    for ct_path in ct_paths:
        # If directory, assume DICOM and extract from metadata
        if ct_path.is_dir():
            try:
                # Get DICOM series list
                series_list = get_dicom_series_list(str(ct_path))
                if series_list:
                    # Just use the first series
                    series = series_list[0]
                    if matching_criteria == 'patient_id':
                        identifier = series.get('patient_id')
                    elif matching_criteria == 'name':
                        identifier = series.get('patient_name')
                    elif matching_criteria == 'date':
                        identifier = series.get('study_date')
                    else:
                        identifier = None
                    
                    if identifier:
                        ct_ids[identifier] = str(ct_path)
            except Exception as e:
                logger.warning(f"Error processing CT directory {ct_path}: {str(e)}")
        else:
            # For NIfTI, use filename
            identifier = ct_path.stem.split('.')[0]  # Remove extension
            ct_ids[identifier] = str(ct_path)
    
    # Find matching pairs
    matched_pairs = []
    for identifier in mri_ids:
        if identifier in ct_ids:
            matched_pairs.append((mri_ids[identifier], ct_ids[identifier]))
    
    logger.info(f"Found {len(matched_pairs)} matching MRI-CT pairs")
    return matched_pairs


def preprocess_pair(mri_path: str, ct_path: str, output_dir: str, 
                    region: str = 'head') -> Tuple[str, str]:
    """
    Preprocess an MRI-CT pair for training.
    
    Args:
        mri_path: Path to MRI file or directory
        ct_path: Path to CT file or directory
        output_dir: Directory to save preprocessed files
        region: Anatomical region ('head', 'pelvis', 'thorax')
        
    Returns:
        Tuple of paths (preprocessed_mri, preprocessed_ct)
    """
    logger.info(f"Preprocessing pair: {mri_path}, {ct_path}")
    
    # Create patient directory in output_dir
    patient_id = Path(mri_path).stem.split('.')[0]
    patient_dir = Path(output_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MRI
    mri_image = load_medical_image(mri_path)
    
    # Preprocess MRI (bias field correction, denoising, normalization)
    preprocessed_mri = preprocess_mri(
        mri_image,
        apply_bias_field_correction=True,
        apply_denoising=True,
        normalize=True,
        resample=True
    )
    
    # Save preprocessed MRI
    preprocessed_mri_path = str(patient_dir / f"{patient_id}_mri_preprocessed.nii.gz")
    save_medical_image(preprocessed_mri.image, preprocessed_mri_path)
    
    # Load CT
    ct_image = load_medical_image(ct_path)
    
    # Resample CT to match MRI
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(preprocessed_mri.image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_ct = resampler.Execute(ct_image)
    
    # Save preprocessed CT
    preprocessed_ct_path = str(patient_dir / f"{patient_id}_ct_preprocessed.nii.gz")
    save_medical_image(resampled_ct, preprocessed_ct_path)
    
    logger.info(f"Saved preprocessed pair: {preprocessed_mri_path}, {preprocessed_ct_path}")
    return preprocessed_mri_path, preprocessed_ct_path


def create_patches(mri_path: str, ct_path: str, output_dir: str, 
                   patch_size: int = 64, stride: int = 32, 
                   min_tissue_percentage: float = 0.2) -> List[Tuple[str, str]]:
    """
    Create patches from MRI and CT images for training.
    
    Args:
        mri_path: Path to preprocessed MRI
        ct_path: Path to preprocessed CT
        output_dir: Directory to save patches
        patch_size: Size of cubic patches
        stride: Stride for patch extraction
        min_tissue_percentage: Minimum percentage of non-background tissue required
        
    Returns:
        List of tuples (mri_patch_path, ct_patch_path)
    """
    logger.info(f"Creating patches from {mri_path} and {ct_path}")
    
    # Load MRI and CT
    mri_image = load_medical_image(mri_path)
    ct_image = load_medical_image(ct_path)
    
    # Convert to numpy arrays
    mri_array = sitk.GetArrayFromImage(mri_image)
    ct_array = sitk.GetArrayFromImage(ct_image)
    
    # Create patches directory
    patient_id = Path(mri_path).stem.split('_')[0]
    patches_dir = Path(output_dir) / f"{patient_id}_patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract patches
    patch_pairs = []
    patch_idx = 0
    
    # Calculate dimensions
    depth, height, width = mri_array.shape
    
    for z in range(0, depth - patch_size + 1, stride):
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Extract patches
                mri_patch = mri_array[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                ct_patch = ct_array[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                
                # Check if patch has enough tissue (not just background)
                # For CT, background is typically around -1000 HU (air)
                tissue_percentage = np.mean(ct_patch > -900)
                
                if tissue_percentage >= min_tissue_percentage:
                    # Save patches
                    mri_patch_path = str(patches_dir / f"{patient_id}_mri_patch_{patch_idx}.npy")
                    ct_patch_path = str(patches_dir / f"{patient_id}_ct_patch_{patch_idx}.npy")
                    
                    np.save(mri_patch_path, mri_patch)
                    np.save(ct_patch_path, ct_patch)
                    
                    patch_pairs.append((mri_patch_path, ct_patch_path))
                    patch_idx += 1
    
    logger.info(f"Created {len(patch_pairs)} patch pairs")
    return patch_pairs


def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.7, 
                  val_ratio: float = 0.15, test_ratio: float = 0.15, 
                  random_seed: int = 42) -> Dict[str, List[Tuple[str, str]]]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        pairs: List of (mri, ct) path pairs
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing split datasets
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    # Set random seed
    random.seed(random_seed)
    
    # Shuffle pairs
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Calculate split indices
    n_samples = len(shuffled_pairs)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split dataset
    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]
    
    logger.info(f"Split dataset: {len(train_pairs)} training, {len(val_pairs)} validation, {len(test_pairs)} test")
    
    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }


def save_split_info(split_data: Dict[str, List[Tuple[str, str]]], output_dir: str) -> None:
    """
    Save dataset split information.
    
    Args:
        split_data: Dictionary containing split datasets
        output_dir: Directory to save split info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, pairs in split_data.items():
        with open(output_dir / f"{split_name}_pairs.txt", 'w') as f:
            for mri_path, ct_path in pairs:
                f.write(f"{mri_path},{ct_path}\n")
    
    logger.info(f"Saved split information to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare training dataset for MRI to CT conversion")
    
    parser.add_argument("--mri_dir", required=True, help="Directory containing MRI data")
    parser.add_argument("--ct_dir", required=True, help="Directory containing CT data")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--region", default="head", choices=["head", "pelvis", "thorax"], 
                        help="Anatomical region")
    parser.add_argument("--matching_criteria", default="patient_id", 
                       choices=["patient_id", "name", "date"], 
                       help="Criteria for matching MRI and CT data")
    parser.add_argument("--create_patches", action="store_true", 
                       help="Create patches for training")
    parser.add_argument("--patch_size", type=int, default=64, 
                       help="Size of cubic patches")
    parser.add_argument("--stride", type=int, default=32, 
                       help="Stride for patch extraction")
    parser.add_argument("--min_tissue_percentage", type=float, default=0.2, 
                       help="Minimum percentage of non-background tissue")
    parser.add_argument("--train_ratio", type=float, default=0.7, 
                       help="Ratio of training data")
    parser.add_argument("--val_ratio", type=float, default=0.15, 
                       help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.15, 
                       help="Ratio of test data")
    parser.add_argument("--random_seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find matching MRI-CT pairs
    pairs = find_matching_pairs(args.mri_dir, args.ct_dir, args.matching_criteria)
    
    if not pairs:
        logger.error("No matching MRI-CT pairs found. Exiting.")
        return 1
    
    # Preprocess pairs
    preprocessed_dir = output_dir / "preprocessed"
    preprocessed_dir.mkdir(exist_ok=True)
    
    preprocessed_pairs = []
    for mri_path, ct_path in tqdm(pairs, desc="Preprocessing"):
        try:
            preprocessed_mri, preprocessed_ct = preprocess_pair(
                mri_path, ct_path, preprocessed_dir, args.region
            )
            preprocessed_pairs.append((preprocessed_mri, preprocessed_ct))
        except Exception as e:
            logger.error(f"Error preprocessing pair {mri_path}, {ct_path}: {str(e)}")
    
    # Create patches if requested
    final_pairs = preprocessed_pairs
    if args.create_patches:
        patches_dir = output_dir / "patches"
        patches_dir.mkdir(exist_ok=True)
        
        patch_pairs = []
        for mri_path, ct_path in tqdm(preprocessed_pairs, desc="Creating patches"):
            try:
                pair_patches = create_patches(
                    mri_path, ct_path, patches_dir, 
                    args.patch_size, args.stride, args.min_tissue_percentage
                )
                patch_pairs.extend(pair_patches)
            except Exception as e:
                logger.error(f"Error creating patches for {mri_path}, {ct_path}: {str(e)}")
        
        final_pairs = patch_pairs
    
    # Split dataset
    split_data = split_dataset(
        final_pairs, 
        args.train_ratio, args.val_ratio, args.test_ratio, 
        args.random_seed
    )
    
    # Save split information
    split_info_dir = output_dir / "split_info"
    save_split_info(split_data, split_info_dir)
    
    logger.info("Dataset preparation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 