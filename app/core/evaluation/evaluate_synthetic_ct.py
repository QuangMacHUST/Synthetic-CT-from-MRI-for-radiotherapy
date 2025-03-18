#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Evaluation module for synthetic CT.
Provides tools to evaluate synthetic CT quality by comparing with real CT.
"""

import os
import json
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple

from app.utils.io_utils import load_medical_image, SyntheticCT
from app.utils.config_utils import load_config

# Set up logger
logger = logging.getLogger(__name__)


class EvaluationResult:
    """
    Class to hold evaluation results with metrics and visualizations.
    """
    
    def __init__(self, metrics=None, image_paths=None, report_path=None):
        """
        Initialize evaluation result object.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics
            image_paths (list): List of paths to generated images
            report_path (str): Path to evaluation report
        """
        self.metrics = metrics or {}
        self.image_paths = image_paths or []
        self.report_path = report_path
        
    def add_metric(self, name, value):
        """Add or update a metric."""
        self.metrics[name] = value
        
    def add_image_path(self, path):
        """Add image path to results."""
        self.image_paths.append(path)
        
    def set_report_path(self, path):
        """Set report path."""
        self.report_path = path
        
    def save_report(self, output_path):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path (str): Path to save report
        """
        report_data = {
            'metrics': self.metrics,
            'image_paths': self.image_paths,
            'report_path': self.report_path
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Saved evaluation report to {output_path}")
        return output_path


def evaluate_synthetic_ct(synthetic_ct_path, reference_ct_path, metrics=None, regions=None, config=None):
    """
    Evaluate synthetic CT by comparing with reference real CT.
    
    Args:
        synthetic_ct_path: Path to synthetic CT or SyntheticCT object
        reference_ct_path: Path to reference real CT
        metrics: List of metrics to calculate (e.g., ['mae', 'mse', 'psnr', 'ssim'])
        regions: List of regions to evaluate (e.g., ['all', 'bone', 'soft_tissue', 'air'])
        config: Configuration dict or None to use default
        
    Returns:
        EvaluationResult: Object containing evaluation results
    """
    logger.info("Starting synthetic CT evaluation")
    
    # Load configuration
    if config is None:
        config = load_config()
    
    # Set default metrics and regions if not provided
    if metrics is None:
        metrics = config.get('evaluation', {}).get('metrics', ['mae', 'mse', 'psnr', 'ssim'])
    
    if regions is None:
        regions = config.get('evaluation', {}).get('regions', ['all'])
    
    # Load synthetic CT
    if isinstance(synthetic_ct_path, SyntheticCT):
        synthetic_ct = synthetic_ct_path.image
        metadata = synthetic_ct_path.metadata
    else:
        synthetic_ct = load_medical_image(synthetic_ct_path)
        metadata = {}
    
    # Load reference CT
    reference_ct = load_medical_image(reference_ct_path)
    
    # Resample reference CT to match synthetic CT if needed
    if reference_ct.GetSize() != synthetic_ct.GetSize() or reference_ct.GetSpacing() != synthetic_ct.GetSpacing():
        logger.info("Resampling reference CT to match synthetic CT")
        reference_ct = resample_image(reference_ct, synthetic_ct)
    
    # Convert to numpy arrays
    synthetic_array = sitk.GetArrayFromImage(synthetic_ct)
    reference_array = sitk.GetArrayFromImage(reference_ct)
    
    # Create mask for foreground voxels (non-air voxels)
    foreground_mask = reference_array > -950  # Typical HU threshold for air/tissue boundary
    
    # Initialize results
    result = EvaluationResult()
    
    # Calculate global metrics
    logger.info("Calculating global metrics")
    global_metrics = calculate_metrics(synthetic_array, reference_array, metrics, foreground_mask)
    for name, value in global_metrics.items():
        result.add_metric(name, value)
    
    # Calculate tissue-specific metrics if requested
    if len(regions) > 1 or regions[0] != 'all':
        logger.info("Calculating tissue-specific metrics")
        # Create tissue masks
        tissue_masks = create_tissue_masks(reference_array)
        
        # Calculate metrics for each requested region
        tissue_metrics = {}
        for region in regions:
            if region == 'all':
                continue  # Already calculated
                
            if region in tissue_masks:
                mask = tissue_masks[region]
                if np.sum(mask) > 0:  # Only evaluate if mask has voxels
                    region_metrics = calculate_metrics(synthetic_array, reference_array, metrics, mask)
                    tissue_metrics[region] = region_metrics
                else:
                    logger.warning(f"Region '{region}' has no voxels. Skipping evaluation.")
            else:
                logger.warning(f"Region '{region}' not found. Skipping evaluation.")
        
        # Add tissue-specific metrics to results
        if tissue_metrics:
            result.add_metric('by_tissue', tissue_metrics)
    
    # Create visualizations if configured
    if config.get('evaluation', {}).get('reporting', {}).get('generate_visualizations', True):
        logger.info("Generating visualizations")
        output_dir = os.path.dirname(reference_ct_path) if isinstance(reference_ct_path, str) else None
        
        if output_dir:
            try:
                from app.utils.visualization import generate_evaluation_report
                
                # Create output directory for visualizations
                vis_dir = os.path.join(output_dir, 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Generate report with visualizations
                report_info = generate_evaluation_report(
                    None,  # MRI not available in this context
                    reference_ct,
                    synthetic_ct,
                    None,  # Segmentation not available in this context
                    result.metrics,
                    vis_dir
                )
                
                # Update result with visualization information
                result.image_paths.extend(report_info.get('image_paths', []))
                report_path = report_info.get('pdf_path')
                if report_path:
                    result.set_report_path(report_path)
                    
            except Exception as e:
                logger.error(f"Error generating visualizations: {str(e)}")
    
    logger.info("Evaluation completed")
    return result


def calculate_metrics(synthetic_array, reference_array, metrics, mask=None):
    """
    Calculate evaluation metrics between synthetic and reference CT.
    
    Args:
        synthetic_array: Synthetic CT as numpy array
        reference_array: Reference CT as numpy array
        metrics: List of metrics to calculate
        mask: Optional binary mask to restrict evaluation
        
    Returns:
        dict: Dictionary of calculated metrics
    """
    # Apply mask if provided
    if mask is not None:
        synthetic_masked = synthetic_array[mask]
        reference_masked = reference_array[mask]
    else:
        synthetic_masked = synthetic_array.flatten()
        reference_masked = reference_array.flatten()
    
    # Check if arrays have valid data
    if len(synthetic_masked) == 0:
        logger.warning("No valid voxels found for metric calculation")
        return {metric: float('nan') for metric in metrics}
    
    # Calculate metrics
    result = {}
    
    for metric in metrics:
        if metric.lower() == 'mae':
            # Mean Absolute Error (in HU)
            result['mae'] = float(np.mean(np.abs(synthetic_masked - reference_masked)))
            
        elif metric.lower() == 'mse':
            # Mean Squared Error (in HU²)
            result['mse'] = float(np.mean((synthetic_masked - reference_masked) ** 2))
            
        elif metric.lower() == 'rmse':
            # Root Mean Squared Error (in HU)
            result['rmse'] = float(np.sqrt(np.mean((synthetic_masked - reference_masked) ** 2)))
            
        elif metric.lower() == 'psnr':
            # Peak Signal to Noise Ratio (in dB)
            mse = np.mean((synthetic_masked - reference_masked) ** 2)
            if mse == 0:
                result['psnr'] = float('inf')
            else:
                data_range = np.max(reference_masked) - np.min(reference_masked)
                result['psnr'] = float(20 * np.log10(data_range / np.sqrt(mse)))
                
        elif metric.lower() == 'ssim':
            # Structural Similarity Index
            try:
                from skimage.metrics import structural_similarity as ssim
                
                # Reshape to 2D for SSIM calculation (if 1D)
                if synthetic_masked.ndim == 1:
                    size = int(np.sqrt(synthetic_masked.shape[0]))
                    synthetic_2d = synthetic_masked[:size**2].reshape(size, size)
                    reference_2d = reference_masked[:size**2].reshape(size, size)
                else:
                    # Use middle slice for 3D data
                    middle_slice = synthetic_array.shape[0] // 2
                    synthetic_2d = synthetic_array[middle_slice]
                    reference_2d = reference_array[middle_slice]
                    if mask is not None:
                        mask_2d = mask[middle_slice]
                        synthetic_2d = synthetic_2d * mask_2d
                        reference_2d = reference_2d * mask_2d
                
                # Normalize data for SSIM calculation
                data_range = np.max(reference_2d) - np.min(reference_2d)
                result['ssim'] = float(ssim(reference_2d, synthetic_2d, 
                                           data_range=data_range))
            except ImportError:
                logger.warning("scikit-image not found, SSIM calculation skipped")
                result['ssim'] = float('nan')
                
        elif metric.lower() == 'mean_error':
            # Mean Error (in HU)
            result['mean_error'] = float(np.mean(synthetic_masked - reference_masked))
            
        elif metric.lower() == 'max_error':
            # Maximum Absolute Error (in HU)
            result['max_error'] = float(np.max(np.abs(synthetic_masked - reference_masked)))
            
        elif metric.lower() == 'percentage_voxels_within_tolerance':
            # Percentage of voxels within tolerance (default: ±20 HU)
            tolerance = 20  # HU
            within_tolerance = np.abs(synthetic_masked - reference_masked) <= tolerance
            result['percentage_within_20HU'] = float(np.mean(within_tolerance) * 100)
            
        elif metric.lower() == 'correlation':
            # Pearson correlation coefficient
            result['correlation'] = float(np.corrcoef(synthetic_masked, reference_masked)[0, 1])
    
    return result


def create_tissue_masks(ct_array):
    """
    Create binary masks for different tissue types based on HU values.
    
    Args:
        ct_array: CT array in Hounsfield Units
        
    Returns:
        dict: Dictionary of binary masks for each tissue type
    """
    # Define HU thresholds for different tissue types
    hu_thresholds = {
        'air': (-1000, -950),         # Air
        'lung': (-950, -700),          # Lung tissue
        'fat': (-150, -50),           # Fat
        'soft_tissue': (-50, 100),    # Soft tissue
        'bone': (300, 3000)           # Bone
    }
    
    # Create masks
    masks = {}
    for tissue, (min_hu, max_hu) in hu_thresholds.items():
        masks[tissue] = (ct_array >= min_hu) & (ct_array <= max_hu)
    
    # Add foreground mask (all non-air tissues)
    masks['foreground'] = ct_array > -950
    
    return masks


def resample_image(image, reference_image, interpolator=sitk.sitkLinear):
    """
    Resample an image to match the spatial characteristics of a reference image.
    
    Args:
        image: The image to resample
        reference_image: The reference image to match
        interpolator: Interpolation method
        
    Returns:
        SimpleITK.Image: Resampled image
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    resample.SetInterpolator(interpolator)
    return resample.Execute(image)


def visualize_evaluation(synthetic_ct, reference_ct, metrics, output_dir, slice_idx=None):
    """
    Generate visualization of evaluation results.
    
    Args:
        synthetic_ct: Synthetic CT image
        reference_ct: Reference CT image
        metrics: Dictionary of metrics
        output_dir: Output directory for visualizations
        slice_idx: Optional slice index to visualize
        
    Returns:
        list: Paths to generated visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert SimpleITK images to numpy arrays
    synthetic_array = sitk.GetArrayFromImage(synthetic_ct)
    reference_array = sitk.GetArrayFromImage(reference_ct)
    
    # Select slice to visualize
    if slice_idx is None:
        slice_idx = synthetic_array.shape[0] // 2
    
    slice_synthetic = synthetic_array[slice_idx]
    slice_reference = reference_array[slice_idx]
    
    # Calculate difference
    slice_diff = slice_synthetic - slice_reference
    
    # Create figure for comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display reference CT
    axes[0].imshow(slice_reference, cmap='gray', vmin=-1000, vmax=1000)
    axes[0].set_title('Reference CT')
    axes[0].axis('off')
    
    # Display synthetic CT
    axes[1].imshow(slice_synthetic, cmap='gray', vmin=-1000, vmax=1000)
    axes[1].set_title('Synthetic CT')
    axes[1].axis('off')
    
    # Display difference
    diff_max = np.max(np.abs(slice_diff))
    im = axes[2].imshow(slice_diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (Synthetic - Reference)')
    axes[2].axis('off')
    
    # Add colorbar for difference
    cbar = fig.colorbar(im, ax=axes[2], orientation='vertical')
    cbar.set_label('HU Difference')
    
    # Add metrics as text
    metrics_text = '\n'.join(f"{k}: {v:.3f}" for k, v in metrics.items() 
                            if k != 'by_tissue' and not isinstance(v, dict))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.05, 0.05, metrics_text, transform=axes[0].transAxes, 
                fontsize=10, verticalalignment='bottom', bbox=props)
    
    # Save figure
    comparison_path = os.path.join(output_dir, f"comparison_slice_{slice_idx}.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close(fig)
    
    # Create histogram comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Clip outliers for better visualization
    flat_synthetic = synthetic_array.flatten()
    flat_reference = reference_array.flatten()
    
    # Get range that includes most values
    combined = np.concatenate([flat_synthetic, flat_reference])
    p1, p99 = np.percentile(combined, [1, 99])
    
    # Create histogram
    bins = np.linspace(p1, p99, 100)
    ax.hist(flat_reference, bins=bins, alpha=0.5, label='Reference CT')
    ax.hist(flat_synthetic, bins=bins, alpha=0.5, label='Synthetic CT')
    
    ax.set_xlabel('Hounsfield Units')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram Comparison')
    ax.legend()
    
    # Save histogram
    histogram_path = os.path.join(output_dir, "histogram_comparison.png")
    plt.tight_layout()
    plt.savefig(histogram_path, dpi=150)
    plt.close(fig)
    
    # Return paths to generated visualizations
    return [comparison_path, histogram_path] 