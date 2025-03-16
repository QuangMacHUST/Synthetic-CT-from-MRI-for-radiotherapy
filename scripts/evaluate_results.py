#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for comprehensive evaluation of synthetic CT results.
This script evaluates image quality metrics and generates detailed reports.
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
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.evaluation.evaluate_synthetic_ct import (
    calculate_mae, calculate_mse, calculate_psnr, calculate_ssim,
    calculate_tissue_specific_metrics, calculate_dvh_metrics,
    generate_visual_report, generate_pdf_report
)
from app.utils.io_utils import load_medical_image, SyntheticCT
from app.core.segmentation import segment_tissues
from app.utils.dicom_utils import load_dicom_series, dicom_to_nifti

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def load_images(synth_ct_path: str, ref_ct_path: str) -> Tuple[sitk.Image, sitk.Image]:
    """
    Load synthetic CT and reference CT images.
    
    Args:
        synth_ct_path: Path to synthetic CT image or directory
        ref_ct_path: Path to reference CT image or directory
        
    Returns:
        Tuple containing synthetic CT and reference CT as SimpleITK images
    """
    logger.info(f"Loading synthetic CT from {synth_ct_path}")
    logger.info(f"Loading reference CT from {ref_ct_path}")
    
    # Check if paths are directories (DICOM) or files (NIfTI)
    if os.path.isdir(synth_ct_path):
        # Load DICOM series
        synth_ct = load_dicom_series(synth_ct_path)
    else:
        # Load NIfTI file
        synth_ct = load_medical_image(synth_ct_path)
    
    if os.path.isdir(ref_ct_path):
        # Load DICOM series
        ref_ct = load_dicom_series(ref_ct_path)
    else:
        # Load NIfTI file
        ref_ct = load_medical_image(ref_ct_path)
    
    # Ensure the images are aligned and have the same dimensions
    # Resample synthetic CT to match reference CT space
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_ct)
    resampler.SetInterpolator(sitk.sitkLinear)
    synth_ct_resampled = resampler.Execute(synth_ct)
    
    return synth_ct_resampled, ref_ct

def generate_segmentation_masks(ct_image: sitk.Image, region: str = 'head') -> Dict[str, sitk.Image]:
    """
    Generate tissue segmentation masks for the CT image.
    
    Args:
        ct_image: CT image to segment
        region: Anatomical region ('head', 'pelvis', 'thorax')
        
    Returns:
        Dictionary of tissue masks {tissue_name: mask_image}
    """
    logger.info(f"Generating segmentation masks for {region} region")
    
    # Segment tissues
    segmentation_result = segment_tissues(ct_image, region=region)
    
    # Extract masks
    masks = {}
    
    if region == 'head':
        tissue_types = ['air', 'soft_tissue', 'bone', 'brain']
    elif region == 'pelvis':
        tissue_types = ['air', 'soft_tissue', 'bone', 'bladder']
    elif region == 'thorax':
        tissue_types = ['air', 'soft_tissue', 'bone', 'lung']
    else:
        tissue_types = ['air', 'soft_tissue', 'bone']
    
    for tissue in tissue_types:
        if hasattr(segmentation_result, tissue):
            masks[tissue] = getattr(segmentation_result, tissue)
    
    logger.info(f"Generated masks for tissues: {list(masks.keys())}")
    return masks

def evaluate_quality_metrics(
    synth_ct: sitk.Image, 
    ref_ct: sitk.Image, 
    masks: Optional[Dict[str, sitk.Image]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate image quality metrics between synthetic CT and reference CT.
    
    Args:
        synth_ct: Synthetic CT image
        ref_ct: Reference CT image
        masks: Optional dictionary of tissue masks
        
    Returns:
        Dictionary of metrics {metric_name: {tissue_name: value}}
    """
    logger.info("Evaluating image quality metrics")
    
    # Calculate global metrics
    mae = calculate_mae(synth_ct, ref_ct)
    mse = calculate_mse(synth_ct, ref_ct)
    psnr = calculate_psnr(synth_ct, ref_ct)
    ssim = calculate_ssim(synth_ct, ref_ct)
    
    metrics = {
        'MAE': {'global': mae},
        'MSE': {'global': mse},
        'PSNR': {'global': psnr},
        'SSIM': {'global': ssim}
    }
    
    # Calculate tissue-specific metrics if masks are provided
    if masks:
        logger.info("Calculating tissue-specific metrics")
        for tissue_name, mask in masks.items():
            tissue_metrics = calculate_tissue_specific_metrics(synth_ct, ref_ct, mask)
            for metric_name, value in tissue_metrics.items():
                metrics[metric_name][tissue_name] = value
    
    return metrics

def evaluate_dvh_metrics(
    synth_ct: sitk.Image, 
    ref_ct: sitk.Image,
    roi_mask: sitk.Image,
    dose_levels: List[int] = [20, 30, 40],
    volume_levels: List[int] = [5, 50, 95]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate dose-volume histogram metrics.
    
    Args:
        synth_ct: Synthetic CT image
        ref_ct: Reference CT image
        roi_mask: Mask of region of interest
        dose_levels: Dose levels for Vx metrics (in Gy)
        volume_levels: Volume levels for Dx metrics (in percentage)
        
    Returns:
        Dictionary of DVH metrics
    """
    logger.info("Evaluating DVH metrics")
    
    dvh_metrics = calculate_dvh_metrics(
        synth_ct, ref_ct, roi_mask, 
        dose_levels=dose_levels, 
        volume_levels=volume_levels
    )
    
    return dvh_metrics

def save_metrics_to_csv(metrics: Dict[str, Dict[str, float]], output_file: str) -> None:
    """
    Save evaluation metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics {metric_name: {tissue_name: value}}
        output_file: Path to output CSV file
    """
    # Convert nested dictionary to DataFrame
    rows = []
    for metric_name, tissue_values in metrics.items():
        for tissue_name, value in tissue_values.items():
            rows.append({'Metric': metric_name, 'Tissue': tissue_name, 'Value': value})
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved metrics to {output_file}")

def save_metrics_to_json(metrics: Dict, output_file: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Saved metrics to {output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate synthetic CT results")
    
    parser.add_argument("--synth_ct", required=True, help="Path to synthetic CT image or directory")
    parser.add_argument("--ref_ct", required=True, help="Path to reference CT image or directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation results")
    parser.add_argument("--region", default="head", choices=["head", "pelvis", "thorax"], 
                       help="Anatomical region")
    parser.add_argument("--roi_mask", help="Path to region of interest mask for DVH calculation")
    parser.add_argument("--generate_report", action="store_true", help="Generate visual and PDF reports")
    parser.add_argument("--no_segmentation", action="store_true", help="Skip tissue segmentation")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images
    synth_ct, ref_ct = load_images(args.synth_ct, args.ref_ct)
    
    # Generate segmentation masks if not skipped
    masks = None
    if not args.no_segmentation:
        masks = generate_segmentation_masks(ref_ct, args.region)
    
    # Evaluate quality metrics
    quality_metrics = evaluate_quality_metrics(synth_ct, ref_ct, masks)
    
    # Save metrics to CSV and JSON
    metrics_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"quality_metrics_{metrics_timestamp}.csv"
    json_path = output_dir / f"quality_metrics_{metrics_timestamp}.json"
    
    save_metrics_to_csv(quality_metrics, csv_path)
    save_metrics_to_json(quality_metrics, json_path)
    
    # Evaluate DVH metrics if ROI mask is provided
    dvh_metrics = None
    if args.roi_mask:
        roi_mask = load_medical_image(args.roi_mask)
        dvh_metrics = evaluate_dvh_metrics(synth_ct, ref_ct, roi_mask)
        
        # Save DVH metrics
        dvh_json_path = output_dir / f"dvh_metrics_{metrics_timestamp}.json"
        save_metrics_to_json(dvh_metrics, dvh_json_path)
    
    # Generate reports if requested
    if args.generate_report:
        logger.info("Generating visual report")
        visual_report_path = output_dir / f"visual_report_{metrics_timestamp}.png"
        generate_visual_report(synth_ct, ref_ct, visual_report_path, masks)
        
        logger.info("Generating PDF report")
        pdf_report_path = output_dir / f"evaluation_report_{metrics_timestamp}.pdf"
        
        # Combine all metrics for report
        all_metrics = quality_metrics.copy()
        if dvh_metrics:
            all_metrics['DVH'] = dvh_metrics
        
        generate_pdf_report(
            synth_ct, ref_ct, 
            metrics=all_metrics, 
            output_path=pdf_report_path,
            masks=masks
        )
    
    logger.info("Evaluation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 