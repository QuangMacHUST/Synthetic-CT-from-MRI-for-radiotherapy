#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation module for synthetic CT images.
This module provides functions to evaluate the quality of synthetic CT images
by comparing them with reference CT images using various metrics.
"""

import os
import logging
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from skimage.metrics import structural_similarity

from app.utils.io_utils import SyntheticCT
from app.utils.config_utils import load_config

# Set up logger
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()


def mean_absolute_error(reference_ct: sitk.Image, synthetic_ct: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    """
    Calculate Mean Absolute Error (MAE) between reference and synthetic CT images.
    
    Args:
        reference_ct: Reference CT image as SimpleITK image
        synthetic_ct: Synthetic CT image as SimpleITK image
        mask: Optional mask to limit evaluation to specific regions
        
    Returns:
        MAE value in Hounsfield Units (HU)
    """
    logger.info("Calculating Mean Absolute Error (MAE)")
    
    # Convert to numpy arrays
    reference_array = sitk.GetArrayFromImage(reference_ct)
    synthetic_array = sitk.GetArrayFromImage(synthetic_ct)
    
    # Apply mask if provided
    if mask is not None:
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Ensure mask is boolean
        if mask_array.dtype != bool:
            mask_array = mask_array > 0
        
        # Calculate MAE only for voxels within mask
        mae = np.mean(np.abs(reference_array[mask_array] - synthetic_array[mask_array]))
    else:
        mae = np.mean(np.abs(reference_array - synthetic_array))
    
    logger.info(f"MAE: {mae:.2f} HU")
    return mae


def mean_squared_error(reference_ct: sitk.Image, synthetic_ct: sitk.Image, mask: Optional[sitk.Image] = None) -> float:
    """
    Calculate Mean Squared Error (MSE) between reference and synthetic CT images.
    
    Args:
        reference_ct: Reference CT image as SimpleITK image
        synthetic_ct: Synthetic CT image as SimpleITK image
        mask: Optional mask to limit evaluation to specific regions
        
    Returns:
        MSE value in Hounsfield Units squared (HU²)
    """
    logger.info("Calculating Mean Squared Error (MSE)")
    
    # Convert to numpy arrays
    reference_array = sitk.GetArrayFromImage(reference_ct)
    synthetic_array = sitk.GetArrayFromImage(synthetic_ct)
    
    # Apply mask if provided
    if mask is not None:
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Ensure mask is boolean
        if mask_array.dtype != bool:
            mask_array = mask_array > 0
        
        # Calculate MSE only for voxels within mask
        mse = np.mean(np.square(reference_array[mask_array] - synthetic_array[mask_array]))
    else:
        mse = np.mean(np.square(reference_array - synthetic_array))
    
    logger.info(f"MSE: {mse:.2f} HU²")
    return mse


def peak_signal_to_noise_ratio(reference_ct: sitk.Image, synthetic_ct: sitk.Image, 
                              mask: Optional[sitk.Image] = None, data_range: Optional[float] = None) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between reference and synthetic CT images.
    
    Args:
        reference_ct: Reference CT image as SimpleITK image
        synthetic_ct: Synthetic CT image as SimpleITK image
        mask: Optional mask to limit evaluation to specific regions
        data_range: Data range for PSNR calculation (max - min)
        
    Returns:
        PSNR value in decibels (dB)
    """
    logger.info("Calculating Peak Signal-to-Noise Ratio (PSNR)")
    
    # Calculate MSE
    mse_value = mean_squared_error(reference_ct, synthetic_ct, mask)
    
    # If data_range is not provided, calculate it from reference CT
    if data_range is None:
        reference_array = sitk.GetArrayFromImage(reference_ct)
        if mask is not None:
            mask_array = sitk.GetArrayFromImage(mask)
            if mask_array.dtype != bool:
                mask_array = mask_array > 0
            data_range = np.max(reference_array[mask_array]) - np.min(reference_array[mask_array])
        else:
            data_range = np.max(reference_array) - np.min(reference_array)
    
    # Calculate PSNR
    if mse_value == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(data_range / np.sqrt(mse_value))
    
    logger.info(f"PSNR: {psnr:.2f} dB")
    return psnr


def structural_similarity_index(reference_ct: sitk.Image, synthetic_ct: sitk.Image, 
                              mask: Optional[sitk.Image] = None, data_range: Optional[float] = None) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between reference and synthetic CT images.
    
    Args:
        reference_ct: Reference CT image as SimpleITK image
        synthetic_ct: Synthetic CT image as SimpleITK image
        mask: Optional mask to limit evaluation to specific regions
        data_range: Data range for SSIM calculation (max - min)
        
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    logger.info("Calculating Structural Similarity Index (SSIM)")
    
    # Convert to numpy arrays
    reference_array = sitk.GetArrayFromImage(reference_ct)
    synthetic_array = sitk.GetArrayFromImage(synthetic_ct)
    
    # If data_range is not provided, calculate it from reference CT
    if data_range is None:
        if mask is not None:
            mask_array = sitk.GetArrayFromImage(mask)
            if mask_array.dtype != bool:
                mask_array = mask_array > 0
            data_range = np.max(reference_array[mask_array]) - np.min(reference_array[mask_array])
        else:
            data_range = np.max(reference_array) - np.min(reference_array)
    
    # Calculate SSIM for 3D image (slice by slice and average)
    ssim_values = []
    for z in range(reference_array.shape[0]):
        if mask is not None:
            mask_array = sitk.GetArrayFromImage(mask)
            if mask_array.dtype != bool:
                mask_array = mask_array > 0
            mask_slice = mask_array[z]
            if not np.any(mask_slice):
                continue
            
            # Apply mask to slice
            ref_slice = reference_array[z].copy()
            syn_slice = synthetic_array[z].copy()
            ssim_val = structural_similarity(
                ref_slice, 
                syn_slice,
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                mask=mask_slice
            )
        else:
            ssim_val = structural_similarity(
                reference_array[z], 
                synthetic_array[z],
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False
            )
        
        ssim_values.append(ssim_val)
    
    # Average SSIM across slices
    ssim = np.mean(ssim_values)
    
    logger.info(f"SSIM: {ssim:.4f}")
    return ssim


def calculate_dvh(dose_image: sitk.Image, structure_mask: sitk.Image, dose_grid_scaling: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Dose-Volume Histogram (DVH) for a structure.
    
    Args:
        dose_image: Dose distribution as SimpleITK image
        structure_mask: Binary mask of the structure as SimpleITK image
        dose_grid_scaling: Scaling factor to convert dose grid values to Gy
        
    Returns:
        Tuple of (dose_bins, volume_percent) arrays
    """
    # Convert to numpy arrays
    dose_array = sitk.GetArrayFromImage(dose_image)
    mask_array = sitk.GetArrayFromImage(structure_mask)
    
    # Ensure mask is boolean
    if mask_array.dtype != bool:
        mask_array = mask_array > 0
    
    # Get doses within structure
    structure_doses = dose_array[mask_array] * dose_grid_scaling
    
    # Calculate total structure volume (in number of voxels)
    total_volume = np.sum(mask_array)
    
    if total_volume == 0:
        logger.warning("Structure mask is empty. Cannot calculate DVH.")
        return np.array([]), np.array([])
    
    # Create dose bins (from 0 to max dose with 0.1 Gy increments)
    max_dose = np.max(structure_doses)
    dose_bins = np.arange(0, max_dose + 0.1, 0.1)
    
    # Calculate volume fraction receiving at least each dose
    volume_percent = np.zeros_like(dose_bins)
    
    for i, dose in enumerate(dose_bins):
        volume_percent[i] = np.sum(structure_doses >= dose) / total_volume * 100
    
    return dose_bins, volume_percent


def compare_dvh(reference_dvh: Tuple[np.ndarray, np.ndarray], synthetic_dvh: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compare DVHs from reference and synthetic CT.
    
    Args:
        reference_dvh: DVH from reference CT as (dose_bins, volume_percent)
        synthetic_dvh: DVH from synthetic CT as (dose_bins, volume_percent)
        
    Returns:
        Dictionary of metrics comparing the DVHs
    """
    ref_dose_bins, ref_volume = reference_dvh
    syn_dose_bins, syn_volume = synthetic_dvh
    
    # Ensure dose bins are the same
    if len(ref_dose_bins) == 0 or len(syn_dose_bins) == 0:
        logger.warning("One or both DVHs are empty. Cannot compare.")
        return {'error': 'Empty DVH'}
    
    if not np.array_equal(ref_dose_bins, syn_dose_bins):
        # Interpolate to common dose bins
        logger.info("Interpolating DVHs to common dose bins")
        min_max_dose = min(np.max(ref_dose_bins), np.max(syn_dose_bins))
        common_bins = np.arange(0, min_max_dose + 0.1, 0.1)
        
        # Interpolate reference DVH
        from scipy.interpolate import interp1d
        ref_interp = interp1d(ref_dose_bins, ref_volume, bounds_error=False, fill_value=(100, 0))
        ref_volume_interp = ref_interp(common_bins)
        
        # Interpolate synthetic DVH
        syn_interp = interp1d(syn_dose_bins, syn_volume, bounds_error=False, fill_value=(100, 0))
        syn_volume_interp = syn_interp(common_bins)
        
        # Use interpolated values
        ref_volume = ref_volume_interp
        syn_volume = syn_volume_interp
        ref_dose_bins = common_bins
        syn_dose_bins = common_bins
    
    # Calculate metrics
    mae = np.mean(np.abs(ref_volume - syn_volume))
    mse = np.mean(np.square(ref_volume - syn_volume))
    
    # Calculate dose at specific volume points (D95, D50, D5)
    ref_d95 = np.interp(95, ref_volume[::-1], ref_dose_bins[::-1])
    syn_d95 = np.interp(95, syn_volume[::-1], syn_dose_bins[::-1])
    ref_d50 = np.interp(50, ref_volume[::-1], ref_dose_bins[::-1])
    syn_d50 = np.interp(50, syn_volume[::-1], syn_dose_bins[::-1])
    ref_d5 = np.interp(5, ref_volume[::-1], ref_dose_bins[::-1])
    syn_d5 = np.interp(5, syn_volume[::-1], syn_dose_bins[::-1])
    
    # Calculate volume at specific dose points (V20, V30, V40)
    ref_v20 = np.interp(20, ref_dose_bins, ref_volume) if 20 <= np.max(ref_dose_bins) else 0
    syn_v20 = np.interp(20, syn_dose_bins, syn_volume) if 20 <= np.max(syn_dose_bins) else 0
    ref_v30 = np.interp(30, ref_dose_bins, ref_volume) if 30 <= np.max(ref_dose_bins) else 0
    syn_v30 = np.interp(30, syn_dose_bins, syn_volume) if 30 <= np.max(syn_dose_bins) else 0
    ref_v40 = np.interp(40, ref_dose_bins, ref_volume) if 40 <= np.max(ref_dose_bins) else 0
    syn_v40 = np.interp(40, syn_dose_bins, syn_volume) if 40 <= np.max(syn_dose_bins) else 0
    
    # Create metrics dictionary
    metrics = {
        'mae': mae,
        'mse': mse,
        'd95_diff': syn_d95 - ref_d95,
        'd50_diff': syn_d50 - ref_d50,
        'd5_diff': syn_d5 - ref_d5,
        'v20_diff': syn_v20 - ref_v20,
        'v30_diff': syn_v30 - ref_v30,
        'v40_diff': syn_v40 - ref_v40
    }
    
    return metrics


def create_region_masks(segmentation: sitk.Image, region: str = 'head') -> Dict[str, sitk.Image]:
    """
    Create masks for different tissue regions from segmentation.
    
    Args:
        segmentation: Segmentation image as SimpleITK image
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        
    Returns:
        Dictionary of masks for different regions
    """
    # Get segmentation labels for this region
    seg_config = config.get_segmentation_params(region)
    labels = seg_config.get('labels', {})
    
    # Convert to numpy array
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    
    # Create masks for different regions
    masks = {}
    masks['all'] = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    masks['all'].CopyInformation(segmentation)
    sitk.GetArrayFromImage(masks['all']).fill(1)
    
    # Create mask for bone
    if 'bone' in labels or 'skull' in labels:
        bone_array = np.zeros_like(segmentation_array, dtype=np.uint8)
        
        # Add all bone structures
        bone_labels = []
        if 'bone' in labels:
            bone_labels.append(labels['bone'])
        if 'skull' in labels:
            bone_labels.append(labels['skull'])
        
        for label in bone_labels:
            bone_array[segmentation_array == label] = 1
        
        bone_mask = sitk.GetImageFromArray(bone_array)
        bone_mask.CopyInformation(segmentation)
        masks['bone'] = bone_mask
    
    # Create mask for soft tissue
    if 'soft_tissue' in labels:
        soft_tissue_array = np.zeros_like(segmentation_array, dtype=np.uint8)
        soft_tissue_array[segmentation_array == labels['soft_tissue']] = 1
        
        # Include other soft tissues depending on region
        if region == 'head' and 'brain' in labels:
            soft_tissue_array[segmentation_array == labels['brain']] = 1
        if region == 'thorax' and 'heart' in labels:
            soft_tissue_array[segmentation_array == labels['heart']] = 1
        
        soft_tissue_mask = sitk.GetImageFromArray(soft_tissue_array)
        soft_tissue_mask.CopyInformation(segmentation)
        masks['soft_tissue'] = soft_tissue_mask
    
    # Create mask for air
    if 'air' in labels:
        air_array = np.zeros_like(segmentation_array, dtype=np.uint8)
        air_array[segmentation_array == labels['air']] = 1
        
        # Include lung for thorax
        if region == 'thorax' and 'lung' in labels:
            air_array[segmentation_array == labels['lung']] = 1
        
        air_mask = sitk.GetImageFromArray(air_array)
        air_mask.CopyInformation(segmentation)
        masks['air'] = air_mask
    
    return masks


def evaluate_synthetic_ct(reference_ct: Union[sitk.Image, SyntheticCT, str], 
                         synthetic_ct: Union[sitk.Image, SyntheticCT, str],
                         segmentation: Optional[Union[sitk.Image, str]] = None,
                         region: str = 'head',
                         metrics: Optional[List[str]] = None,
                         dose_images: Optional[Dict[str, sitk.Image]] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate synthetic CT quality compared to reference CT.
    
    Args:
        reference_ct: Reference CT image as SimpleITK image, SyntheticCT object, or path to file
        synthetic_ct: Synthetic CT image as SimpleITK image, SyntheticCT object, or path to file
        segmentation: Optional segmentation image for region-based evaluation
        region: Anatomical region ('head', 'pelvis', or 'thorax')
        metrics: List of metrics to calculate. If None, uses metrics from config.
        dose_images: Optional dictionary of dose images calculated on reference and synthetic CT
        
    Returns:
        Dictionary of metrics for different regions
    """
    logger.info("Starting synthetic CT evaluation")
    
    # Convert input to SimpleITK images if needed
    if isinstance(reference_ct, str):
        # Load image from file
        from app.utils.io_utils import load_medical_image
        reference_ct = load_medical_image(reference_ct)
    
    if isinstance(synthetic_ct, str):
        # Load image from file
        from app.utils.io_utils import load_medical_image
        synthetic_ct = load_medical_image(synthetic_ct)
    
    if segmentation is not None and isinstance(segmentation, str):
        # Load segmentation from file
        from app.utils.io_utils import load_medical_image
        segmentation = load_medical_image(segmentation)
    
    # Extract SimpleITK images from SyntheticCT objects if needed
    if isinstance(reference_ct, SyntheticCT):
        reference_image = reference_ct.image
    else:
        reference_image = reference_ct
    
    if isinstance(synthetic_ct, SyntheticCT):
        synthetic_image = synthetic_ct.image
    else:
        synthetic_image = synthetic_ct
    
    # Check if images have the same size and spacing
    if reference_image.GetSize() != synthetic_image.GetSize():
        logger.warning("Reference and synthetic CTs have different sizes. Resampling...")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        synthetic_image = resampler.Execute(synthetic_image)
    
    # Get evaluation configuration
    eval_config = config.get_evaluation_params()
    
    # Determine metrics to calculate
    if metrics is None:
        metrics = eval_config.get('metrics', ['mae', 'mse', 'psnr', 'ssim'])
    
    # Determine regions to evaluate
    eval_regions = eval_config.get('regions', ['all'])
    
    # Create region masks if segmentation is provided
    region_masks = {}
    if segmentation is not None:
        region_masks = create_region_masks(segmentation, region)
    else:
        # Create an "all" mask covering the entire image
        all_mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
        all_mask.CopyInformation(reference_image)
        sitk.GetArrayFromImage(all_mask).fill(1)
        region_masks['all'] = all_mask
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate for each region
    for region_name in eval_regions:
        if region_name in region_masks:
            logger.info(f"Evaluating region: {region_name}")
            
            region_mask = region_masks[region_name]
            region_results = {}
            
            # Calculate metrics
            if 'mae' in metrics:
                region_results['mae'] = mean_absolute_error(reference_image, synthetic_image, region_mask)
            
            if 'mse' in metrics:
                region_results['mse'] = mean_squared_error(reference_image, synthetic_image, region_mask)
            
            if 'psnr' in metrics:
                region_results['psnr'] = peak_signal_to_noise_ratio(reference_image, synthetic_image, region_mask)
            
            if 'ssim' in metrics:
                region_results['ssim'] = structural_similarity_index(reference_image, synthetic_image, region_mask)
            
            # Calculate DVH comparison if dose images are provided
            if 'dvh' in metrics and dose_images is not None and 'reference' in dose_images and 'synthetic' in dose_images:
                ref_dose = dose_images['reference']
                syn_dose = dose_images['synthetic']
                
                # Calculate DVHs
                ref_dvh = calculate_dvh(ref_dose, region_mask)
                syn_dvh = calculate_dvh(syn_dose, region_mask)
                
                # Compare DVHs
                dvh_metrics = compare_dvh(ref_dvh, syn_dvh)
                
                # Add DVH metrics to results
                for key, value in dvh_metrics.items():
                    region_results[f'dvh_{key}'] = value
            
            # Add region results to overall results
            results[region_name] = region_results
        else:
            logger.warning(f"Region '{region_name}' not found in segmentation. Skipping.")
    
    logger.info("Synthetic CT evaluation completed")
    return results


def evaluate_and_report(reference_ct: Union[sitk.Image, SyntheticCT, str], 
                       synthetic_ct: Union[sitk.Image, SyntheticCT, str],
                       segmentation: Optional[Union[sitk.Image, str]] = None,
                       region: str = 'head',
                       output_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate synthetic CT and generate a comprehensive report.
    
    Args:
        reference_ct: Reference CT image
        synthetic_ct: Synthetic CT image
        segmentation: Optional segmentation image
        region: Anatomical region
        output_dir: Directory to save report files
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Starting evaluation and report generation")
    
    # Get evaluation configuration
    eval_config = config.get_evaluation_params()
    
    # Perform evaluation
    results = evaluate_synthetic_ct(reference_ct, synthetic_ct, segmentation, region)
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate report if requested
    if eval_config.get('reporting', {}).get('generate_pdf', False) and output_dir is not None:
        logger.info("Generating evaluation report")
        
        # Extract SimpleITK images if needed
        if isinstance(reference_ct, SyntheticCT):
            reference_image = reference_ct.image
        elif isinstance(reference_ct, str):
            from app.utils.io_utils import load_medical_image
            reference_image = load_medical_image(reference_ct).image
        else:
            reference_image = reference_ct
        
        if isinstance(synthetic_ct, SyntheticCT):
            synthetic_image = synthetic_ct.image
        elif isinstance(synthetic_ct, str):
            from app.utils.io_utils import load_medical_image
            synthetic_image = load_medical_image(synthetic_ct).image
        else:
            synthetic_image = synthetic_ct
        
        # Save comparison figures if requested
        if eval_config.get('reporting', {}).get('save_figures', False):
            try:
                # Import visualization module
                from app.utils.visualization import (
                    save_comparison_figure, 
                    plot_difference, 
                    plot_evaluation_results
                )
                
                # Save comparison figure
                comparison_path = os.path.join(output_dir, f"{region}_comparison.png")
                save_comparison_figure(
                    reference_image, 
                    synthetic_image, 
                    comparison_path,
                    titles=["Reference CT", "Synthetic CT"]
                )
                logger.info(f"Saved comparison figure to {comparison_path}")
                
                # Save difference figure
                difference_path = os.path.join(output_dir, f"{region}_difference.png")
                fig = plt.figure(figsize=(10, 8))
                plot_difference(
                    reference_image,
                    synthetic_image,
                    title="Difference Map (Reference - Synthetic)"
                )
                plt.savefig(difference_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved difference figure to {difference_path}")
                
                # Save evaluation results figure
                metrics_path = os.path.join(output_dir, f"{region}_metrics.png")
                fig = plt.figure(figsize=(12, 8))
                plot_evaluation_results(results, title=f"Evaluation Results for {region}")
                plt.savefig(metrics_path, dpi=300)
                plt.close(fig)
                logger.info(f"Saved metrics figure to {metrics_path}")
            
            except Exception as e:
                logger.error(f"Error generating figures: {str(e)}")
        
        # Generate PDF report if requested
        if eval_config.get('reporting', {}).get('generate_pdf', False):
            try:
                # Import report generation module
                from app.utils.report_utils import generate_report
                
                # Generate report
                report_path = os.path.join(output_dir, f"{region}_evaluation_report.pdf")
                generate_report(
                    results, 
                    reference_image, 
                    synthetic_image, 
                    region=region,
                    output_path=report_path
                )
                logger.info(f"Generated evaluation report at {report_path}")
            
            except Exception as e:
                logger.error(f"Error generating PDF report: {str(e)}")
    
    logger.info("Evaluation and report generation completed")
    return results 