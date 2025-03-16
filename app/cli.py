#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for MRI to synthetic CT conversion.
This module provides a command-line interface for preprocessing MRI images,
segmenting tissues, converting MRI to synthetic CT, evaluating synthetic CT,
and visualizing results.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.config_utils import load_config, ConfigManager
from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT
from app.utils.dicom_utils import (
    load_dicom_series, save_dicom_series, 
    create_synthetic_dicom_series, anonymize_dicom
)
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct
from app.core.evaluation import evaluate_synthetic_ct
from app.visualization import plot_image_slice, plot_comparison, create_visual_report

# Set up logger
logger = logging.getLogger(__name__)


def setup_logging(args):
    """Set up logging."""
    config = load_config()
    
    # Get logging level from config or command line
    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        log_level = getattr(logging, config.get('general', {}).get('logging_level', 'INFO').upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{args.command}_{Path(args.input).stem}.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting {args.command} with arguments: {args}")


def preprocess_command(args):
    """Handle preprocessing command."""
    # Load MRI image
    logger.info(f"Loading MRI image from {args.input}")
    mri_image = load_medical_image(args.input)
    
    # Preprocess MRI
    logger.info("Preprocessing MRI image")
    preprocessed_mri = preprocess_mri(
        mri_image,
        apply_bias_field_correction=args.bias_correction,
        apply_denoising=args.denoise,
        apply_normalization=args.normalize,
        apply_resampling=args.resample,
        target_spacing=args.target_spacing
    )
    
    # Save preprocessed MRI
    logger.info(f"Saving preprocessed MRI to {args.output}")
    save_medical_image(preprocessed_mri, args.output)
    
    logger.info("Preprocessing completed successfully")


def segment_command(args):
    """Handle segmentation command."""
    # Load MRI image
    logger.info(f"Loading MRI image from {args.input}")
    mri_image = load_medical_image(args.input)
    
    # Segment tissues
    logger.info(f"Segmenting tissues using {args.method} method for {args.region} region")
    segmentation = segment_tissues(
        mri_image,
        method=args.method,
        anatomical_region=args.region
    )
    
    # Save segmentation
    logger.info(f"Saving segmentation to {args.output}")
    save_medical_image(segmentation, args.output)
    
    logger.info("Segmentation completed successfully")


def convert_command(args):
    """Handle conversion command."""
    # Load MRI image
    logger.info(f"Loading MRI image from {args.input}")
    mri_image = load_medical_image(args.input)
    
    # Load segmentation if provided
    segmentation = None
    if args.segmentation:
        logger.info(f"Loading segmentation from {args.segmentation}")
        segmentation = load_medical_image(args.segmentation)
    
    # Convert MRI to synthetic CT
    logger.info(f"Converting MRI to synthetic CT using {args.method} method")
    synthetic_ct = convert_mri_to_ct(
        mri_image,
        segmentation=segmentation,
        method=args.method,
        model_path=args.model
    )
    
    # Save synthetic CT
    logger.info(f"Saving synthetic CT to {args.output}")
    if args.output.lower().endswith('.dcm') or Path(args.output).is_dir():
        # Save as DICOM series
        if isinstance(args.input, str) and Path(args.input).is_dir():
            # If input is a DICOM series, use it as reference
            logger.info("Creating synthetic DICOM series from reference MRI DICOM")
            create_synthetic_dicom_series(
                synthetic_ct.image,
                args.input,
                args.output,
                anonymize=args.anonymize
            )
        else:
            # Otherwise, save as a new DICOM series
            logger.info("Saving as DICOM series")
            save_dicom_series(
                synthetic_ct.image,
                args.output
            )
    else:
        # Save as other format (NIfTI, etc.)
        save_medical_image(synthetic_ct.image, args.output)
    
    logger.info("Conversion completed successfully")


def evaluate_command(args):
    """Handle evaluation command."""
    # Load synthetic CT
    logger.info(f"Loading synthetic CT from {args.synthetic_ct}")
    synthetic_ct = load_medical_image(args.synthetic_ct)
    
    # Load reference CT
    logger.info(f"Loading reference CT from {args.reference_ct}")
    reference_ct = load_medical_image(args.reference_ct)
    
    # Load segmentation/mask if provided
    mask = None
    if args.mask:
        logger.info(f"Loading evaluation mask from {args.mask}")
        mask = load_medical_image(args.mask)
    
    # Evaluate synthetic CT
    logger.info("Evaluating synthetic CT")
    evaluation_results = evaluate_synthetic_ct(
        synthetic_ct,
        reference_ct,
        mask=mask,
        metrics=args.metrics.split(',')
    )
    
    # Print evaluation results
    logger.info("Evaluation results:")
    for metric, value in evaluation_results.items():
        logger.info(f"{metric}: {value}")
    
    # Create visual report if requested
    if args.report:
        logger.info(f"Creating visual report at {args.report}")
        create_visual_report(
            mri=None,  # We don't have MRI here
            synthetic_ct=synthetic_ct,
            reference_ct=reference_ct,
            evaluation_results=evaluation_results,
            output_path=args.report
        )
    
    logger.info("Evaluation completed successfully")


def visualize_command(args):
    """Handle visualization command."""
    # Load images
    images = []
    titles = []
    
    logger.info(f"Loading image from {args.input}")
    image1 = load_medical_image(args.input)
    images.append(image1)
    titles.append(Path(args.input).stem)
    
    if args.compare:
        logger.info(f"Loading comparison image from {args.compare}")
        image2 = load_medical_image(args.compare)
        images.append(image2)
        titles.append(Path(args.compare).stem)
    
    # Create visualization
    if args.compare:
        logger.info("Creating comparison visualization")
        fig = plot_comparison(
            images[0],
            images[1],
            slice_indices=args.slice,
            titles=titles,
            colormap=args.colormap,
            window_center=args.window_center,
            window_width=args.window_width
        )
    else:
        logger.info("Creating single image visualization")
        fig = plot_image_slice(
            images[0],
            slice_index=args.slice,
            axis=args.axis,
            title=titles[0],
            colormap=args.colormap,
            window_center=args.window_center,
            window_width=args.window_width
        )
    
    # Save or show visualization
    if args.output:
        logger.info(f"Saving visualization to {args.output}")
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
    else:
        logger.info("Displaying visualization")
        plt.show()
    
    logger.info("Visualization completed successfully")


def main():
    """Main function for command-line interface."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="MRI to synthetic CT conversion tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument(
        '--config',
        help='Path to configuration file',
        default='configs/default_config.yaml'
    )
    parser.add_argument(
        '--log_level',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='Logging level'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess MRI images'
    )
    preprocess_parser.add_argument(
        'input',
        help='Input MRI image or DICOM directory'
    )
    preprocess_parser.add_argument(
        'output',
        help='Output preprocessed MRI image'
    )
    preprocess_parser.add_argument(
        '--bias_correction',
        action='store_true',
        help='Apply bias field correction'
    )
    preprocess_parser.add_argument(
        '--denoise',
        action='store_true',
        help='Apply denoising'
    )
    preprocess_parser.add_argument(
        '--normalize',
        action='store_true',
        help='Apply intensity normalization'
    )
    preprocess_parser.add_argument(
        '--resample',
        action='store_true',
        help='Apply resampling'
    )
    preprocess_parser.add_argument(
        '--target_spacing',
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help='Target voxel spacing for resampling (x, y, z)'
    )
    
    # Segment command
    segment_parser = subparsers.add_parser(
        'segment',
        help='Segment tissues in MRI images'
    )
    segment_parser.add_argument(
        'input',
        help='Input MRI image or DICOM directory'
    )
    segment_parser.add_argument(
        'output',
        help='Output segmentation image'
    )
    segment_parser.add_argument(
        '--method',
        choices=['atlas', 'deeplearning'],
        default='deeplearning',
        help='Segmentation method'
    )
    segment_parser.add_argument(
        '--region',
        choices=['head', 'pelvis', 'thorax'],
        default='head',
        help='Anatomical region'
    )
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert MRI to synthetic CT'
    )
    convert_parser.add_argument(
        'input',
        help='Input MRI image or DICOM directory'
    )
    convert_parser.add_argument(
        'output',
        help='Output synthetic CT image or DICOM directory'
    )
    convert_parser.add_argument(
        '--method',
        choices=['atlas', 'cnn', 'gan'],
        default='cnn',
        help='Conversion method'
    )
    convert_parser.add_argument(
        '--segmentation',
        help='Input tissue segmentation (optional)'
    )
    convert_parser.add_argument(
        '--model',
        help='Path to CNN or GAN model (for CNN or GAN methods)'
    )
    convert_parser.add_argument(
        '--anonymize',
        action='store_true',
        help='Anonymize output DICOM series'
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate synthetic CT against reference CT'
    )
    evaluate_parser.add_argument(
        'synthetic_ct',
        help='Input synthetic CT image or DICOM directory'
    )
    evaluate_parser.add_argument(
        'reference_ct',
        help='Reference CT image or DICOM directory'
    )
    evaluate_parser.add_argument(
        '--mask',
        help='Evaluation mask for region-specific evaluation (optional)'
    )
    evaluate_parser.add_argument(
        '--metrics',
        default='mae,mse,psnr,ssim',
        help='Comma-separated list of metrics to compute'
    )
    evaluate_parser.add_argument(
        '--report',
        help='Path for saving visual evaluation report (optional)'
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize medical images'
    )
    visualize_parser.add_argument(
        'input',
        help='Input medical image or DICOM directory'
    )
    visualize_parser.add_argument(
        '--compare',
        help='Second image for comparison (optional)'
    )
    visualize_parser.add_argument(
        '--output',
        help='Output image file (optional, displays if not provided)'
    )
    visualize_parser.add_argument(
        '--slice',
        type=int,
        default=-1,
        help='Slice index (-1 for middle slice)'
    )
    visualize_parser.add_argument(
        '--axis',
        type=int,
        choices=[0, 1, 2],
        default=0,
        help='Viewing axis (0=axial, 1=coronal, 2=sagittal)'
    )
    visualize_parser.add_argument(
        '--colormap',
        default='gray',
        help='Colormap for visualization'
    )
    visualize_parser.add_argument(
        '--window_center',
        type=float,
        default=None,
        help='Window center for visualization (Hounsfield units for CT)'
    )
    visualize_parser.add_argument(
        '--window_width',
        type=float,
        default=None,
        help='Window width for visualization (Hounsfield units for CT)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Handle commands
    if args.command == 'preprocess':
        setup_logging(args)
        preprocess_command(args)
    elif args.command == 'segment':
        setup_logging(args)
        segment_command(args)
    elif args.command == 'convert':
        setup_logging(args)
        convert_command(args)
    elif args.command == 'evaluate':
        setup_logging(args)
        evaluate_command(args)
    elif args.command == 'visualize':
        setup_logging(args)
        visualize_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 