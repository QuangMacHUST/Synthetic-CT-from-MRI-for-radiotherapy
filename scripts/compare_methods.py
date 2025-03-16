#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for comparing different MRI to CT conversion methods.
This script compares multiple synthetic CT results against a reference CT.
"""

import os
import sys
import argparse
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.evaluation.evaluate_synthetic_ct import (
    calculate_mae, calculate_mse, calculate_psnr, calculate_ssim,
    calculate_tissue_specific_metrics, calculate_dvh_metrics
)
from app.utils.io_utils import load_medical_image, SyntheticCT
from app.utils.dicom_utils import load_dicom_series

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def load_ct_image(path: str) -> sitk.Image:
    """
    Load CT image from file or directory.
    
    Args:
        path: Path to CT image file or DICOM directory
        
    Returns:
        SimpleITK image
    """
    if os.path.isdir(path):
        # Load DICOM series
        return load_dicom_series(path)
    else:
        # Load NIfTI file
        return load_medical_image(path)

def resample_to_reference(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Resample image to match reference image dimensions and spacing.
    
    Args:
        image: Image to resample
        reference: Reference image
        
    Returns:
        Resampled image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def calculate_metrics(synth_ct: sitk.Image, ref_ct: sitk.Image) -> Dict[str, float]:
    """
    Calculate image quality metrics between synthetic CT and reference CT.
    
    Args:
        synth_ct: Synthetic CT image
        ref_ct: Reference CT image
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': calculate_mae(synth_ct, ref_ct),
        'MSE': calculate_mse(synth_ct, ref_ct),
        'PSNR': calculate_psnr(synth_ct, ref_ct),
        'SSIM': calculate_ssim(synth_ct, ref_ct)
    }
    
    return metrics

def generate_comparison_visualization(
    images: Dict[str, sitk.Image],
    output_path: str,
    slice_indices: Optional[Dict[str, int]] = None,
    planes: List[str] = ['axial', 'coronal', 'sagittal']
) -> None:
    """
    Generate side-by-side comparison visualization.
    
    Args:
        images: Dictionary of named images {name: image}
        output_path: Path to save visualization
        slice_indices: Optional dictionary of slice indices for each plane
        planes: List of planes to visualize
    """
    logger.info(f"Generating comparison visualization for {len(images)} images")
    
    # Number of images to compare (including reference)
    n_images = len(images)
    
    # Set up figure
    fig_width = 4 * n_images
    fig_height = 4 * len(planes)
    fig, axes = plt.subplots(len(planes), n_images, figsize=(fig_width, fig_height))
    
    # If only one plane, make axes 2D
    if len(planes) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Get reference image for dimensions
    ref_image = list(images.values())[0]
    ref_size = ref_image.GetSize()
    
    # Default slice indices (middle of each dimension)
    if slice_indices is None:
        slice_indices = {
            'axial': ref_size[2] // 2,
            'coronal': ref_size[1] // 2,
            'sagittal': ref_size[0] // 2
        }
    
    # Define plane to dimension mapping
    plane_to_dim = {
        'axial': 2,     # Z dimension
        'coronal': 1,   # Y dimension
        'sagittal': 0   # X dimension
    }
    
    # Process each plane
    for i, plane in enumerate(planes):
        if plane not in plane_to_dim:
            logger.warning(f"Unsupported plane: {plane}. Skipping.")
            continue
        
        # Get dimension and slice index
        dim = plane_to_dim[plane]
        slice_idx = slice_indices.get(plane, ref_size[dim] // 2)
        
        # Process each image
        for j, (name, image) in enumerate(images.items()):
            # Extract slice
            if dim == 0:  # Sagittal (YZ plane)
                slice_data = sitk.Extract(
                    image,
                    [1, image.GetSize()[1], image.GetSize()[2]],
                    [slice_idx, 0, 0]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
                # Transpose to get correct orientation
                array = array
            elif dim == 1:  # Coronal (XZ plane)
                slice_data = sitk.Extract(
                    image,
                    [image.GetSize()[0], 1, image.GetSize()[2]],
                    [0, slice_idx, 0]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
                # Transpose to get correct orientation
                array = array.T
            else:  # Axial (XY plane)
                slice_data = sitk.Extract(
                    image,
                    [image.GetSize()[0], image.GetSize()[1], 1],
                    [0, 0, slice_idx]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
            
            # Display slice
            ax = axes[i, j]
            im = ax.imshow(array, cmap='gray', vmin=-1000, vmax=1000)
            ax.set_title(f"{name} - {plane.capitalize()}")
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Visualization saved to {output_path}")

def generate_difference_maps(
    images: Dict[str, sitk.Image],
    ref_key: str,
    output_path: str,
    slice_indices: Optional[Dict[str, int]] = None,
    planes: List[str] = ['axial', 'coronal', 'sagittal']
) -> None:
    """
    Generate difference maps between each image and the reference.
    
    Args:
        images: Dictionary of named images {name: image}
        ref_key: Key of reference image in images dictionary
        output_path: Path to save visualization
        slice_indices: Optional dictionary of slice indices for each plane
        planes: List of planes to visualize
    """
    logger.info("Generating difference maps")
    
    # Ensure reference key exists
    if ref_key not in images:
        logger.error(f"Reference key '{ref_key}' not found in images")
        return
    
    # Reference image
    ref_image = images[ref_key]
    
    # Other images (excluding reference)
    other_images = {k: v for k, v in images.items() if k != ref_key}
    n_others = len(other_images)
    
    # Set up figure
    fig_width = 4 * n_others
    fig_height = 4 * len(planes)
    fig, axes = plt.subplots(len(planes), n_others, figsize=(fig_width, fig_height))
    
    # If only one plane and one comparison, make axes 2D
    if len(planes) == 1 and n_others == 1:
        axes = np.array([[axes]])
    elif len(planes) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_others == 1:
        axes = np.expand_dims(axes, axis=1)
    
    # Get reference dimensions
    ref_size = ref_image.GetSize()
    
    # Default slice indices (middle of each dimension)
    if slice_indices is None:
        slice_indices = {
            'axial': ref_size[2] // 2,
            'coronal': ref_size[1] // 2,
            'sagittal': ref_size[0] // 2
        }
    
    # Define plane to dimension mapping
    plane_to_dim = {
        'axial': 2,     # Z dimension
        'coronal': 1,   # Y dimension
        'sagittal': 0   # X dimension
    }
    
    # Process each plane
    for i, plane in enumerate(planes):
        if plane not in plane_to_dim:
            logger.warning(f"Unsupported plane: {plane}. Skipping.")
            continue
        
        # Get dimension and slice index
        dim = plane_to_dim[plane]
        slice_idx = slice_indices.get(plane, ref_size[dim] // 2)
        
        # Process each comparison image
        for j, (name, image) in enumerate(other_images.items()):
            # Calculate difference image
            diff_image = sitk.Subtract(image, ref_image)
            
            # Extract slice
            if dim == 0:  # Sagittal
                slice_data = sitk.Extract(
                    diff_image,
                    [1, diff_image.GetSize()[1], diff_image.GetSize()[2]],
                    [slice_idx, 0, 0]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
            elif dim == 1:  # Coronal
                slice_data = sitk.Extract(
                    diff_image,
                    [diff_image.GetSize()[0], 1, diff_image.GetSize()[2]],
                    [0, slice_idx, 0]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
                array = array.T
            else:  # Axial
                slice_data = sitk.Extract(
                    diff_image,
                    [diff_image.GetSize()[0], diff_image.GetSize()[1], 1],
                    [0, 0, slice_idx]
                )
                array = sitk.GetArrayFromImage(slice_data).squeeze()
            
            # Display slice
            ax = axes[i, j]
            
            # Symmetric colormap centered at zero
            max_abs = max(abs(np.min(array)), abs(np.max(array)))
            vmin, vmax = -max_abs, max_abs
            
            im = ax.imshow(array, cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.set_title(f"{name} - {ref_key} ({plane.capitalize()})")
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Difference maps saved to {output_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare different MRI to CT conversion methods")
    
    parser.add_argument("--ref_ct", required=True, help="Path to reference CT image")
    parser.add_argument("--output_dir", required=True, help="Output directory for comparison results")
    parser.add_argument("--methods", required=True, nargs='+', 
                       help="Paths to synthetic CT images from different methods")
    parser.add_argument("--method_names", nargs='+', 
                       help="Names of the methods (same order as methods)")
    parser.add_argument("--axial_slice", type=int, help="Axial slice index")
    parser.add_argument("--coronal_slice", type=int, help="Coronal slice index")
    parser.add_argument("--sagittal_slice", type=int, help="Sagittal slice index")
    parser.add_argument("--planes", nargs='+', default=['axial', 'coronal', 'sagittal'],
                       choices=['axial', 'coronal', 'sagittal'],
                       help="Planes to visualize")
    parser.add_argument("--roi_mask", help="Path to region of interest mask for DVH calculation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reference CT
    logger.info(f"Loading reference CT from {args.ref_ct}")
    ref_ct = load_ct_image(args.ref_ct)
    
    # Load synthetic CTs from different methods
    synth_cts = {}
    
    # Validate method_names if provided
    if args.method_names and len(args.method_names) != len(args.methods):
        logger.error("Number of method names must match number of methods")
        return 1
    
    # Use provided method names or generate default names
    method_names = args.method_names if args.method_names else [f"Method_{i+1}" for i in range(len(args.methods))]
    
    # Load and resample synthetic CTs
    for i, (method_path, method_name) in enumerate(zip(args.methods, method_names)):
        logger.info(f"Loading synthetic CT for {method_name} from {method_path}")
        synth_ct = load_ct_image(method_path)
        
        # Resample to reference CT space
        synth_ct_resampled = resample_to_reference(synth_ct, ref_ct)
        synth_cts[method_name] = synth_ct_resampled
    
    # Add reference CT to dictionary
    all_images = {"Reference": ref_ct}
    all_images.update(synth_cts)
    
    # Create slice indices dictionary
    slice_indices = {}
    if args.axial_slice is not None:
        slice_indices['axial'] = args.axial_slice
    if args.coronal_slice is not None:
        slice_indices['coronal'] = args.coronal_slice
    if args.sagittal_slice is not None:
        slice_indices['sagittal'] = args.sagittal_slice
    
    # Generate comparison visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = output_dir / f"method_comparison_{timestamp}.png"
    generate_comparison_visualization(all_images, vis_path, slice_indices, args.planes)
    
    # Generate difference maps
    diff_path = output_dir / f"difference_maps_{timestamp}.png"
    generate_difference_maps(all_images, "Reference", diff_path, slice_indices, args.planes)
    
    # Calculate metrics for each method
    metrics = {}
    for method_name, synth_ct in synth_cts.items():
        metrics[method_name] = calculate_metrics(synth_ct, ref_ct)
    
    # Convert metrics to DataFrame for easier comparison
    metrics_df = pd.DataFrame(metrics)
    
    # Save metrics to CSV
    metrics_path = output_dir / f"method_comparison_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate metrics comparison figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    
    metric_names = ['MAE', 'MSE', 'PSNR', 'SSIM']
    
    for i, metric in enumerate(metric_names):
        metrics_df.loc[metric].plot(kind='bar', ax=axs[i])
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel('Method')
        axs[i].grid(axis='y')
    
    plt.tight_layout()
    metrics_fig_path = output_dir / f"metrics_comparison_{timestamp}.png"
    plt.savefig(metrics_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Metrics visualization saved to {metrics_fig_path}")
    
    # Calculate DVH metrics if ROI mask is provided
    if args.roi_mask:
        logger.info(f"Loading ROI mask from {args.roi_mask}")
        roi_mask = load_ct_image(args.roi_mask)
        
        # Resample mask to reference CT space if needed
        roi_mask_resampled = resample_to_reference(roi_mask, ref_ct)
        
        # Calculate DVH metrics for each method
        dvh_metrics = {}
        for method_name, synth_ct in synth_cts.items():
            dvh_metrics[method_name] = calculate_dvh_metrics(synth_ct, ref_ct, roi_mask_resampled)
        
        # Save DVH metrics to JSON
        dvh_path = output_dir / f"dvh_comparison_{timestamp}.json"
        with open(dvh_path, 'w') as f:
            json.dump(dvh_metrics, f, indent=4)
        logger.info(f"DVH metrics saved to {dvh_path}")
    
    logger.info("Comparison completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 