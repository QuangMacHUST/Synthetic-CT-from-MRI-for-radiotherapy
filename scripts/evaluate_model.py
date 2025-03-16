#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate MRI to CT conversion models
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import (
    setup_logging,
    load_medical_image,
    load_config,
    plot_comparison,
    plot_histogram
)
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct
from app.core.evaluation import (
    evaluate_synthetic_ct,
    calculate_mae,
    calculate_mse,
    calculate_psnr,
    calculate_ssim
)


def evaluate_single_case(
    mri_path: str,
    ct_path: str,
    output_dir: str,
    model_type: str = "gan",
    region: str = "head",
    plot_results: bool = True
) -> Dict:
    """
    Evaluate MRI to CT conversion for a single case.

    Args:
        mri_path: Path to MRI image.
        ct_path: Path to reference CT image.
        output_dir: Output directory for results.
        model_type: Type of model to use (atlas, cnn, gan).
        region: Anatomical region (head, pelvis, thorax).
        plot_results: Whether to plot comparison results.

    Returns:
        Dictionary of evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    logging.info(f"Processing MRI: {mri_path}")
    logging.info(f"Reference CT: {ct_path}")
    
    # Preprocess MRI
    preprocessed_mri = preprocess_mri(mri_path)
    
    # Segment tissues
    segmented_tissues = segment_tissues(preprocessed_mri, region=region)
    
    # Convert MRI to CT
    synthetic_ct = convert_mri_to_ct(
        preprocessed_mri, 
        segmented_tissues, 
        model_type=model_type,
        region=region
    )
    
    # Load reference CT
    reference_ct = load_medical_image(ct_path)
    
    # Evaluate synthetic CT
    metrics = evaluate_synthetic_ct(synthetic_ct, reference_ct, segmented_tissues)
    
    # Save synthetic CT
    synthetic_ct_path = os.path.join(output_dir, f"synthetic_ct_{model_type}_{region}.nii.gz")
    sitk.WriteImage(synthetic_ct.image, synthetic_ct_path)
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "PSNR", "SSIM"],
        "Overall": [
            metrics["overall"]["mae"],
            metrics["overall"]["mse"],
            metrics["overall"]["psnr"],
            metrics["overall"]["ssim"]
        ]
    })
    
    # Add tissue-specific metrics if available
    if "by_tissue" in metrics:
        for tissue, tissue_metrics in metrics["by_tissue"].items():
            metrics_df[tissue.capitalize()] = [
                tissue_metrics["mae"],
                tissue_metrics["mse"],
                tissue_metrics["psnr"],
                tissue_metrics.get("ssim", np.nan)  # SSIM may not be available for all tissues
            ]
    
    metrics_df.to_csv(os.path.join(output_dir, f"metrics_{model_type}_{region}.csv"), index=False)
    
    # Plot comparison if requested
    if plot_results:
        # Get mid slices
        synthetic_ct_array = sitk.GetArrayFromImage(synthetic_ct.image)
        reference_ct_array = sitk.GetArrayFromImage(reference_ct)
        
        mid_slice_axial = synthetic_ct_array.shape[0] // 2
        mid_slice_coronal = synthetic_ct_array.shape[1] // 2
        mid_slice_sagittal = synthetic_ct_array.shape[2] // 2
        
        # Plot axial, coronal and sagittal views
        # Axial
        plot_comparison(
            reference_ct_array[mid_slice_axial, :, :],
            synthetic_ct_array[mid_slice_axial, :, :],
            title1="Reference CT (Axial)",
            title2="Synthetic CT (Axial)",
            window_center=40,
            window_width=400,
            filename=os.path.join(output_dir, f"comparison_axial_{model_type}_{region}.png")
        )
        
        # Coronal
        plot_comparison(
            reference_ct_array[:, mid_slice_coronal, :],
            synthetic_ct_array[:, mid_slice_coronal, :],
            title1="Reference CT (Coronal)",
            title2="Synthetic CT (Coronal)",
            window_center=40,
            window_width=400,
            filename=os.path.join(output_dir, f"comparison_coronal_{model_type}_{region}.png")
        )
        
        # Sagittal
        plot_comparison(
            reference_ct_array[:, :, mid_slice_sagittal],
            synthetic_ct_array[:, :, mid_slice_sagittal],
            title1="Reference CT (Sagittal)",
            title2="Synthetic CT (Sagittal)",
            window_center=40,
            window_width=400,
            filename=os.path.join(output_dir, f"comparison_sagittal_{model_type}_{region}.png")
        )
        
        # Plot histogram
        plot_histogram(
            reference_ct_array.flatten(),
            synthetic_ct_array.flatten(),
            title1="Reference CT",
            title2="Synthetic CT",
            bins=100,
            range=(-1000, 1000),
            filename=os.path.join(output_dir, f"histogram_{model_type}_{region}.png")
        )
        
        # Plot difference map
        diff_map = np.abs(reference_ct_array - synthetic_ct_array)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(diff_map[mid_slice_axial, :, :], cmap="hot", vmin=0, vmax=200)
        plt.colorbar(label="HU difference")
        plt.title("Absolute Difference (Axial)")
        
        plt.subplot(132)
        plt.imshow(diff_map[:, mid_slice_coronal, :], cmap="hot", vmin=0, vmax=200)
        plt.colorbar(label="HU difference")
        plt.title("Absolute Difference (Coronal)")
        
        plt.subplot(133)
        plt.imshow(diff_map[:, :, mid_slice_sagittal], cmap="hot", vmin=0, vmax=200)
        plt.colorbar(label="HU difference")
        plt.title("Absolute Difference (Sagittal)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"difference_map_{model_type}_{region}.png"))
        plt.close()
    
    return metrics


def evaluate_batch(
    mri_dir: str,
    ct_dir: str,
    output_dir: str,
    model_type: str = "gan",
    region: str = "head",
    suffix: str = ".nii.gz"
) -> Dict:
    """
    Evaluate MRI to CT conversion for a batch of cases.

    Args:
        mri_dir: Directory containing MRI images.
        ct_dir: Directory containing reference CT images.
        output_dir: Output directory for results.
        model_type: Type of model to use (atlas, cnn, gan).
        region: Anatomical region (head, pelvis, thorax).
        suffix: File suffix to match.

    Returns:
        Dictionary of aggregated evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of MRI files
    mri_files = [
        os.path.join(mri_dir, f) for f in os.listdir(mri_dir)
        if f.endswith(suffix)
    ]
    
    # Get list of CT files
    ct_files = [
        os.path.join(ct_dir, f) for f in os.listdir(ct_dir)
        if f.endswith(suffix)
    ]
    
    # Sort files to ensure matching
    mri_files.sort()
    ct_files.sort()
    
    # Check if MRI and CT lists have the same length
    if len(mri_files) != len(ct_files):
        raise ValueError("Number of MRI and CT files must match.")
    
    # Initialize aggregated metrics
    aggregated_metrics = {
        "overall": {
            "mae": [],
            "mse": [],
            "psnr": [],
            "ssim": []
        },
        "by_tissue": {}
    }
    
    # Process each MRI-CT pair
    for i, (mri_file, ct_file) in enumerate(zip(mri_files, ct_files)):
        logging.info(f"Processing case {i+1}/{len(mri_files)}")
        
        # Create case-specific output directory
        case_name = os.path.basename(mri_file).split(".")[0]
        case_output_dir = os.path.join(output_dir, case_name)
        
        # Evaluate single case
        metrics = evaluate_single_case(
            mri_file,
            ct_file,
            case_output_dir,
            model_type=model_type,
            region=region,
            plot_results=True
        )
        
        # Append metrics to aggregated metrics
        for metric in ["mae", "mse", "psnr", "ssim"]:
            aggregated_metrics["overall"][metric].append(metrics["overall"][metric])
        
        # Append tissue-specific metrics if available
        if "by_tissue" in metrics:
            for tissue, tissue_metrics in metrics["by_tissue"].items():
                if tissue not in aggregated_metrics["by_tissue"]:
                    aggregated_metrics["by_tissue"][tissue] = {
                        "mae": [],
                        "mse": [],
                        "psnr": [],
                        "ssim": []
                    }
                
                for metric in ["mae", "mse", "psnr", "ssim"]:
                    if metric in tissue_metrics:
                        aggregated_metrics["by_tissue"][tissue][metric].append(tissue_metrics[metric])
    
    # Calculate mean and std of metrics
    summary_metrics = {
        "overall": {
            metric: {
                "mean": np.mean(values),
                "std": np.std(values)
            }
            for metric, values in aggregated_metrics["overall"].items()
        },
        "by_tissue": {
            tissue: {
                metric: {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
                for metric, values in tissue_metrics.items()
                if len(values) > 0
            }
            for tissue, tissue_metrics in aggregated_metrics["by_tissue"].items()
        }
    }
    
    # Save summary metrics as CSV
    # Overall metrics
    summary_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "PSNR", "SSIM"],
        "Mean": [
            summary_metrics["overall"]["mae"]["mean"],
            summary_metrics["overall"]["mse"]["mean"],
            summary_metrics["overall"]["psnr"]["mean"],
            summary_metrics["overall"]["ssim"]["mean"]
        ],
        "Std": [
            summary_metrics["overall"]["mae"]["std"],
            summary_metrics["overall"]["mse"]["std"],
            summary_metrics["overall"]["psnr"]["std"],
            summary_metrics["overall"]["ssim"]["std"]
        ]
    })
    
    summary_df.to_csv(os.path.join(output_dir, f"summary_metrics_{model_type}_{region}.csv"), index=False)
    
    # Tissue-specific metrics
    for tissue, tissue_metrics in summary_metrics["by_tissue"].items():
        tissue_df = pd.DataFrame({
            "Metric": ["MAE", "MSE", "PSNR", "SSIM"],
            "Mean": [
                tissue_metrics.get("mae", {}).get("mean", np.nan),
                tissue_metrics.get("mse", {}).get("mean", np.nan),
                tissue_metrics.get("psnr", {}).get("mean", np.nan),
                tissue_metrics.get("ssim", {}).get("mean", np.nan)
            ],
            "Std": [
                tissue_metrics.get("mae", {}).get("std", np.nan),
                tissue_metrics.get("mse", {}).get("std", np.nan),
                tissue_metrics.get("psnr", {}).get("std", np.nan),
                tissue_metrics.get("ssim", {}).get("std", np.nan)
            ]
        })
        
        tissue_df.to_csv(
            os.path.join(output_dir, f"summary_metrics_{tissue}_{model_type}_{region}.csv"),
            index=False
        )
    
    # Plot summary box plots
    plt.figure(figsize=(15, 10))
    
    # MAE box plot
    plt.subplot(2, 2, 1)
    tissue_names = list(aggregated_metrics["by_tissue"].keys())
    mae_values = [aggregated_metrics["overall"]["mae"]]
    mae_labels = ["Overall"]
    
    for tissue in tissue_names:
        if "mae" in aggregated_metrics["by_tissue"][tissue]:
            mae_values.append(aggregated_metrics["by_tissue"][tissue]["mae"])
            mae_labels.append(tissue.capitalize())
    
    plt.boxplot(mae_values, labels=mae_labels)
    plt.title("Mean Absolute Error (MAE)")
    plt.ylabel("MAE (HU)")
    plt.grid(linestyle="--", alpha=0.7)
    
    # MSE box plot
    plt.subplot(2, 2, 2)
    mse_values = [aggregated_metrics["overall"]["mse"]]
    mse_labels = ["Overall"]
    
    for tissue in tissue_names:
        if "mse" in aggregated_metrics["by_tissue"][tissue]:
            mse_values.append(aggregated_metrics["by_tissue"][tissue]["mse"])
            mse_labels.append(tissue.capitalize())
    
    plt.boxplot(mse_values, labels=mse_labels)
    plt.title("Mean Squared Error (MSE)")
    plt.ylabel("MSE (HU²)")
    plt.grid(linestyle="--", alpha=0.7)
    
    # PSNR box plot
    plt.subplot(2, 2, 3)
    psnr_values = [aggregated_metrics["overall"]["psnr"]]
    psnr_labels = ["Overall"]
    
    for tissue in tissue_names:
        if "psnr" in aggregated_metrics["by_tissue"][tissue]:
            psnr_values.append(aggregated_metrics["by_tissue"][tissue]["psnr"])
            psnr_labels.append(tissue.capitalize())
    
    plt.boxplot(psnr_values, labels=psnr_labels)
    plt.title("Peak Signal-to-Noise Ratio (PSNR)")
    plt.ylabel("PSNR (dB)")
    plt.grid(linestyle="--", alpha=0.7)
    
    # SSIM box plot
    plt.subplot(2, 2, 4)
    ssim_values = [aggregated_metrics["overall"]["ssim"]]
    ssim_labels = ["Overall"]
    
    for tissue in tissue_names:
        if "ssim" in aggregated_metrics["by_tissue"][tissue]:
            ssim_values.append(aggregated_metrics["by_tissue"][tissue]["ssim"])
            ssim_labels.append(tissue.capitalize())
    
    plt.boxplot(ssim_values, labels=ssim_labels)
    plt.title("Structural Similarity Index (SSIM)")
    plt.ylabel("SSIM")
    plt.grid(linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_boxplots_{model_type}_{region}.png"))
    plt.close()
    
    return summary_metrics


def compare_models(
    mri_path: str,
    ct_path: str,
    output_dir: str,
    region: str = "head",
    model_types: List[str] = ["atlas", "cnn", "gan"]
) -> None:
    """
    Compare different MRI to CT conversion models for a single case.

    Args:
        mri_path: Path to MRI image.
        ct_path: Path to reference CT image.
        output_dir: Output directory for results.
        region: Anatomical region (head, pelvis, thorax).
        model_types: List of model types to compare.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    logging.info(f"Processing MRI: {mri_path}")
    logging.info(f"Reference CT: {ct_path}")
    
    # Preprocess MRI
    preprocessed_mri = preprocess_mri(mri_path)
    
    # Segment tissues
    segmented_tissues = segment_tissues(preprocessed_mri, region=region)
    
    # Load reference CT
    reference_ct = load_medical_image(ct_path)
    reference_ct_array = sitk.GetArrayFromImage(reference_ct)
    
    # Generate synthetic CTs for each model type
    synthetic_cts = {}
    metrics = {}
    
    for model_type in model_types:
        logging.info(f"Converting using {model_type} model")
        
        # Convert MRI to CT
        synthetic_ct = convert_mri_to_ct(
            preprocessed_mri, 
            segmented_tissues, 
            model_type=model_type,
            region=region
        )
        
        # Save synthetic CT
        synthetic_ct_path = os.path.join(output_dir, f"synthetic_ct_{model_type}_{region}.nii.gz")
        sitk.WriteImage(synthetic_ct.image, synthetic_ct_path)
        
        synthetic_cts[model_type] = synthetic_ct
        
        # Evaluate synthetic CT
        metrics[model_type] = evaluate_synthetic_ct(synthetic_ct, reference_ct, segmented_tissues)
    
    # Save comparative metrics as CSV
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "PSNR", "SSIM"]
    })
    
    for model_type in model_types:
        metrics_df[model_type.capitalize()] = [
            metrics[model_type]["overall"]["mae"],
            metrics[model_type]["overall"]["mse"],
            metrics[model_type]["overall"]["psnr"],
            metrics[model_type]["overall"]["ssim"]
        ]
    
    metrics_df.to_csv(os.path.join(output_dir, f"comparative_metrics_{region}.csv"), index=False)
    
    # Plot comparison
    # Get mid slice
    mid_slice_axial = reference_ct_array.shape[0] // 2
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Reference CT
    plt.subplot(2, 2, 1)
    plt.imshow(reference_ct_array[mid_slice_axial, :, :], cmap="gray", vmin=-200, vmax=200)
    plt.colorbar(label="HU")
    plt.title("Reference CT")
    
    # Synthetic CTs
    for i, model_type in enumerate(model_types):
        plt.subplot(2, 2, i + 2)
        synthetic_ct_array = sitk.GetArrayFromImage(synthetic_cts[model_type].image)
        plt.imshow(synthetic_ct_array[mid_slice_axial, :, :], cmap="gray", vmin=-200, vmax=200)
        plt.colorbar(label="HU")
        mae = metrics[model_type]["overall"]["mae"]
        psnr = metrics[model_type]["overall"]["psnr"]
        plt.title(f"{model_type.capitalize()} (MAE: {mae:.2f} HU, PSNR: {psnr:.2f} dB)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"model_comparison_{region}.png"))
    plt.close()
    
    # Plot difference maps
    plt.figure(figsize=(15, 10))
    
    for i, model_type in enumerate(model_types):
        plt.subplot(1, len(model_types), i + 1)
        synthetic_ct_array = sitk.GetArrayFromImage(synthetic_cts[model_type].image)
        diff_map = np.abs(reference_ct_array - synthetic_ct_array)
        plt.imshow(diff_map[mid_slice_axial, :, :], cmap="hot", vmin=0, vmax=200)
        plt.colorbar(label="HU difference")
        plt.title(f"{model_type.capitalize()} Difference Map")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"difference_comparison_{region}.png"))
    plt.close()


def main():
    """Main function for evaluating MRI to CT conversion models."""
    parser = argparse.ArgumentParser(description="Evaluate MRI to CT conversion models")
    
    # Input arguments
    parser.add_argument("--mri_path", help="Path to MRI image for single case evaluation")
    parser.add_argument("--ct_path", help="Path to reference CT image for single case evaluation")
    parser.add_argument("--mri_dir", help="Directory containing MRI images for batch evaluation")
    parser.add_argument("--ct_dir", help="Directory containing reference CT images for batch evaluation")
    parser.add_argument("--output_dir", default="results/evaluation", help="Output directory for results")
    
    # Model parameters
    parser.add_argument(
        "--model_type",
        default="gan",
        choices=["atlas", "cnn", "gan"],
        help="Type of model to use"
    )
    parser.add_argument(
        "--region",
        default="head",
        choices=["head", "pelvis", "thorax"],
        help="Anatomical region"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "batch", "compare"],
        help="Evaluation mode"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate based on mode
    if args.mode == "single":
        if args.mri_path is None or args.ct_path is None:
            parser.error("--mri_path and --ct_path are required for single case evaluation")
        
        evaluate_single_case(
            args.mri_path,
            args.ct_path,
            args.output_dir,
            model_type=args.model_type,
            region=args.region,
            plot_results=True
        )
    
    elif args.mode == "batch":
        if args.mri_dir is None or args.ct_dir is None:
            parser.error("--mri_dir and --ct_dir are required for batch evaluation")
        
        evaluate_batch(
            args.mri_dir,
            args.ct_dir,
            args.output_dir,
            model_type=args.model_type,
            region=args.region
        )
    
    elif args.mode == "compare":
        if args.mri_path is None or args.ct_path is None:
            parser.error("--mri_path and --ct_path are required for model comparison")
        
        compare_models(
            args.mri_path,
            args.ct_path,
            args.output_dir,
            region=args.region,
            model_types=["atlas", "cnn", "gan"]
        )


if __name__ == "__main__":
    main() 