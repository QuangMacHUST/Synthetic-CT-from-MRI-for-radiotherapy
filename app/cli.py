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

from app.utils.config_utils import load_config, ConfigManager, create_default_config, update_config_from_args
from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT, validate_input_file, ensure_output_dir, load_multi_sequence_mri
from app.utils.dicom_utils import (
    load_dicom_series, save_dicom_series, 
    create_synthetic_dicom_series, anonymize_dicom
)
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct, available_conversion_methods
from app.core.evaluation import evaluate_synthetic_ct
from app.visualization import plot_image_slice, plot_comparison, create_visual_report
from app.utils.logging_utils import setup_logging

# Set up logger
logger = logging.getLogger(__name__)


def setup_parser():
    """
    Set up command-line argument parser.
    
    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="MRI to synthetic CT conversion for radiotherapy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run entire conversion pipeline on a T1-weighted MRI scan
  python -m app.cli pipeline --input path/to/t1.nii.gz --output path/to/output --region head
  
  # Run preprocessing only
  python -m app.cli preprocess --input path/to/t1.nii.gz --output path/to/output
  
  # Run conversion with CNN model
  python -m app.cli convert --input path/to/preprocessed.nii.gz --output path/to/output --model cnn --region head
  
  # Run evaluation on synthetic CT
  python -m app.cli evaluate --synthetic path/to/synthetic_ct.nii.gz --reference path/to/real_ct.nii.gz --output path/to/output
  
  # Use a custom configuration file
  python -m app.cli pipeline --input path/to/t1.nii.gz --output path/to/output --config path/to/config.yaml
        """
    )
    
    # Global arguments
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set logging level")
    parser.add_argument("--log-file", help="Path to log file")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    
    # Config mode
    config_parser = subparsers.add_parser("config", help="Create or manage configuration files")
    config_parser.add_argument("--create-default", help="Path to create default configuration file")
    
    # Preprocess mode
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess MRI images")
    preprocess_parser.add_argument("--input", required=True, help="Path to input MRI file")
    preprocess_parser.add_argument("--output", required=True, help="Path to output directory")
    preprocess_parser.add_argument("--bias-correction", action="store_true", help="Apply bias field correction")
    preprocess_parser.add_argument("--denoise", action="store_true", help="Apply denoising")
    preprocess_parser.add_argument("--normalize", action="store_true", help="Apply intensity normalization")
    
    # Multi-sequence parser (for handling multiple MRI sequences)
    multi_seq_parser = subparsers.add_parser("multi-sequence", help="Process multiple MRI sequences")
    multi_seq_parser.add_argument("--inputs", required=True, nargs="+", help="Paths to input MRI files")
    multi_seq_parser.add_argument("--sequences", required=True, nargs="+", 
                                 choices=["T1", "T2", "FLAIR", "DWI", "ADC", "DIXON"],
                                 help="MRI sequence types (in the same order as inputs)")
    multi_seq_parser.add_argument("--output", required=True, help="Path to output directory")
    multi_seq_parser.add_argument("--reference", default="T1", 
                                 choices=["T1", "T2", "FLAIR", "DWI", "ADC", "DIXON"],
                                 help="Reference sequence for registration")
    
    # Segment mode
    segment_parser = subparsers.add_parser("segment", help="Segment tissues in MRI images")
    segment_parser.add_argument("--input", required=True, help="Path to preprocessed MRI file")
    segment_parser.add_argument("--output", required=True, help="Path to output directory")
    segment_parser.add_argument("--region", required=True, choices=["head", "pelvis", "thorax"], 
                               help="Anatomical region")
    segment_parser.add_argument("--method", choices=["atlas", "nn"], default="nn",
                              help="Segmentation method (atlas-based or neural network)")
    
    # Convert mode
    convert_parser = subparsers.add_parser("convert", help="Convert MRI to synthetic CT")
    convert_parser.add_argument("--input", required=True, help="Path to preprocessed MRI file")
    convert_parser.add_argument("--segmentation", help="Path to tissue segmentation file (optional)")
    convert_parser.add_argument("--output", required=True, help="Path to output directory")
    convert_parser.add_argument("--model", required=True, choices=available_conversion_methods(),
                              help="Conversion model to use")
    convert_parser.add_argument("--region", required=True, choices=["head", "pelvis", "thorax"],
                              help="Anatomical region")
    convert_parser.add_argument("--multi-sequence", action="store_true", 
                              help="Enable multi-sequence conversion if multiple inputs are available")
    
    # Evaluate mode
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate synthetic CT")
    evaluate_parser.add_argument("--synthetic", required=True, help="Path to synthetic CT file")
    evaluate_parser.add_argument("--reference", required=True, help="Path to reference CT file")
    evaluate_parser.add_argument("--segmentation", help="Path to tissue segmentation for region-specific metrics")
    evaluate_parser.add_argument("--output", required=True, help="Path to output directory")
    evaluate_parser.add_argument("--metrics", default="mae,mse,psnr,ssim",
                               help="Comma-separated list of metrics to calculate")
    evaluate_parser.add_argument("--regions-of-interest", help="Comma-separated list of ROI labels to analyze separately")
    evaluate_parser.add_argument("--generate-report", action="store_true", help="Generate comprehensive PDF report")
    evaluate_parser.add_argument("--export-format", choices=["json", "csv", "both"], default="json",
                               help="Format for exporting evaluation results")
    evaluate_parser.add_argument("--histogram-compare", action="store_true", help="Include histogram comparison in evaluation")
    evaluate_parser.add_argument("--dose-calculation", action="store_true", 
                               help="Perform dose calculation comparison (requires additional RT structure and plan files)")
    evaluate_parser.add_argument("--rt-structures", help="Path to RT structure set for dose calculation")
    evaluate_parser.add_argument("--rt-plan", help="Path to RT plan for dose calculation")
    
    # Pipeline mode (run entire workflow)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run entire MRI to synthetic CT pipeline")
    pipeline_parser.add_argument("--input", required=True, help="Path to input MRI file")
    pipeline_parser.add_argument("--output", required=True, help="Path to output directory")
    pipeline_parser.add_argument("--region", required=True, choices=["head", "pelvis", "thorax"],
                               help="Anatomical region")
    pipeline_parser.add_argument("--model", default="gan", choices=available_conversion_methods(),
                               help="Conversion model to use")
    pipeline_parser.add_argument("--reference-ct", help="Path to reference CT for evaluation (optional)")
    pipeline_parser.add_argument("--skip-preprocessing", action="store_true", 
                               help="Skip preprocessing step (use if input is already preprocessed)")
    pipeline_parser.add_argument("--skip-segmentation", action="store_true",
                               help="Skip segmentation step (use if segmentation is not required)")
    
    # Visualization mode
    visualization_parser = subparsers.add_parser("visualize", help="Visualize results")
    visualization_parser.add_argument("--mri", help="Path to MRI file")
    visualization_parser.add_argument("--synthetic-ct", help="Path to synthetic CT file")
    visualization_parser.add_argument("--reference-ct", help="Path to reference CT file")
    visualization_parser.add_argument("--segmentation", help="Path to segmentation file")
    visualization_parser.add_argument("--output", required=True, help="Path to output directory for visualizations")
    visualization_parser.add_argument("--type", choices=["slice", "compare", "3d", "all"], default="compare",
                                    help="Type of visualization to generate")
    visualization_parser.add_argument("--orientation", choices=["axial", "sagittal", "coronal", "all"], default="axial",
                                    help="Slice orientation for 2D visualizations")
    visualization_parser.add_argument("--slices", type=str, help="Comma-separated list of slice indices to visualize")
    visualization_parser.add_argument("--window-level", type=str, 
                                    help="Window/level values in format 'window,level' for CT visualization")
    visualization_parser.add_argument("--colormap", help="Colormap to use for visualizations")
    visualization_parser.add_argument("--overlay", action="store_true", help="Create overlay visualizations")
    visualization_parser.add_argument("--difference-map", action="store_true", help="Generate difference maps")
    visualization_parser.add_argument("--interactive", action="store_true", 
                                    help="Generate interactive visualizations (HTML output)")
    
    # Patient archive mode
    patient_parser = subparsers.add_parser("patient", help="Manage patient data archive")
    patient_parser.add_argument("--action", choices=["import", "export", "list", "delete", "anonymize"],
                              required=True, help="Action to perform")
    patient_parser.add_argument("--input", help="Path to input file/directory (for import/export actions)")
    patient_parser.add_argument("--output", help="Path to output directory (for export action)")
    patient_parser.add_argument("--patient-id", help="Patient ID (for list/export/delete actions)")
    patient_parser.add_argument("--modalities", help="Comma-separated list of modalities to import/export")
    patient_parser.add_argument("--anonymize", action="store_true", help="Anonymize data during import/export")

    # Deployment mode
    deploy_parser = subparsers.add_parser("deploy", help="Deploy model as service")
    deploy_parser.add_argument("--model", required=True, choices=available_conversion_methods(),
                             help="Model to deploy")
    deploy_parser.add_argument("--region", required=True, choices=["head", "pelvis", "thorax"],
                             help="Anatomical region")
    deploy_parser.add_argument("--port", type=int, default=8000, help="Port to run service on")
    deploy_parser.add_argument("--host", default="0.0.0.0", help="Host to run service on")
    deploy_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    deploy_parser.add_argument("--api-key", help="API key for authentication")
    deploy_parser.add_argument("--ssl-cert", help="Path to SSL certificate")
    deploy_parser.add_argument("--ssl-key", help="Path to SSL key")
    
    return parser


def process_config_command(args):
    """
    Process config-related commands.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        int: Exit code (0 for success)
    """
    if args.create_default:
        try:
            path = create_default_config(args.create_default)
            print(f"Default configuration created at: {path}")
            return 0
        except Exception as e:
            print(f"Error creating default configuration: {str(e)}")
            return 1
    else:
        print("Please specify a config action (e.g., --create-default)")
        return 1


def process_preprocess_command(args, config):
    """
    Process preprocessing command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input file
    if not validate_input_file(args.input):
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Get preprocessing parameters
        params = config.get_preprocessing_params()
        
        # Run preprocessing
        output_path = preprocess_mri(
                args.input,
            output_dir,
            bias_correction=args.bias_correction if args.bias_correction else params.get('bias_field_correction', {}).get('enable', True),
            denoise=args.denoise if args.denoise else params.get('denoising', {}).get('enable', True),
            normalize=args.normalize if args.normalize else params.get('normalization', {}).get('enable', True)
        )
        
        print(f"Preprocessing completed. Output saved to: {output_path}")
        return 0
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        print(f"Error during preprocessing: {str(e)}")
        return 1


def process_multi_sequence_command(args, config):
    """
    Process multi-sequence command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input files
    for input_file in args.inputs:
        if not validate_input_file(input_file):
            return 1
    
    # Check if number of inputs matches number of sequences
    if len(args.inputs) != len(args.sequences):
        print(f"Error: Number of inputs ({len(args.inputs)}) does not match number of sequences ({len(args.sequences)})")
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Load multi-sequence MRI
        multi_seq_mri = load_multi_sequence_mri(args.inputs, args.sequences)
        
        # Set reference sequence
        multi_seq_mri.set_reference(args.reference)
        
        # Register and save
        multi_seq_mri.register_all()
        output_paths = multi_seq_mri.save_all(output_dir)
        
        print(f"Multi-sequence processing completed. Outputs saved to: {output_dir}")
        for seq, path in zip(args.sequences, output_paths):
            print(f"  - {seq}: {path}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error during multi-sequence processing: {str(e)}", exc_info=True)
        print(f"Error during multi-sequence processing: {str(e)}")
        return 1


def process_segment_command(args, config):
    """
    Process segmentation command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input file
    if not validate_input_file(args.input):
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Get segmentation parameters
        params = config.get_segmentation_params(args.region)
        
        # Run segmentation
        output_path = segment_tissues(
            args.input,
            output_dir,
            region=args.region,
            method=args.method if args.method else params.get('method', 'nn')
        )
        
        print(f"Segmentation completed. Output saved to: {output_path}")
        return 0
    
    except Exception as e:
        logging.error(f"Error during segmentation: {str(e)}", exc_info=True)
        print(f"Error during segmentation: {str(e)}")
        return 1


def process_convert_command(args, config):
    """
    Process conversion command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input file
    if not validate_input_file(args.input):
        return 1
    
    # Validate segmentation file if provided
    if args.segmentation and not validate_input_file(args.segmentation):
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Get conversion parameters
        params = config.get_conversion_params(args.model, args.region)
        
        # Run conversion
        output_path = convert_mri_to_ct(
            args.input,
            output_dir,
            method=args.model,
            region=args.region,
            segmentation_path=args.segmentation,
            use_multi_sequence=args.multi_sequence,
            params=params
        )
        
        print(f"Conversion completed. Output saved to: {output_path}")
        return 0
    
    except Exception as e:
        logging.error(f"Error during conversion: {str(e)}", exc_info=True)
        print(f"Error during conversion: {str(e)}")
        return 1


def process_evaluate_command(args, config):
    """
    Process evaluation command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input files
    if not validate_input_file(args.synthetic):
        return 1
    
    if not validate_input_file(args.reference):
        return 1
    
    if args.segmentation and not validate_input_file(args.segmentation):
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Get evaluation parameters
        params = config.get_evaluation_params()
        
        # Parse metrics
        metrics = args.metrics.split(',')
        
        # Run evaluation
        result = evaluate_synthetic_ct(
            args.synthetic,
            args.reference,
            output_dir,
            segmentation_path=args.segmentation,
            metrics=metrics
        )
        
        print(f"Evaluation completed. Results saved to: {output_dir}")
        print("\nMetrics:")
        for metric, value in result.metrics.items():
            print(f"  - {metric}: {value}")
        
        return 0
    
    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}", exc_info=True)
        print(f"Error during evaluation: {str(e)}")
        return 1


def process_pipeline_command(args, config):
    """
    Process pipeline command (run entire workflow).
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Validate input file
    if not validate_input_file(args.input):
        return 1
    
    # Validate reference CT if provided
    if args.reference_ct and not validate_input_file(args.reference_ct):
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Create subdirectories
        preproc_dir = os.path.join(output_dir, "preprocessed")
        segment_dir = os.path.join(output_dir, "segmentation")
        convert_dir = os.path.join(output_dir, "synthetic_ct")
        eval_dir = os.path.join(output_dir, "evaluation")
        
        os.makedirs(preproc_dir, exist_ok=True)
        os.makedirs(segment_dir, exist_ok=True)
        os.makedirs(convert_dir, exist_ok=True)
        if args.reference_ct:
            os.makedirs(eval_dir, exist_ok=True)
        
        # Step 1: Preprocessing
        preprocessed_path = args.input
        if not args.skip_preprocessing:
            print("Step 1/4: Preprocessing MRI...")
            preprocessed_path = preprocess_mri(
                args.input,
                preproc_dir,
                bias_correction=True,
                denoise=True,
                normalize=True
            )
            print(f"Preprocessing completed. Output saved to: {preprocessed_path}")
        else:
            print("Step 1/4: Preprocessing skipped.")
        
        # Step 2: Segmentation
        segmentation_path = None
        if not args.skip_segmentation:
            print("\nStep 2/4: Segmenting tissues...")
            segmentation_path = segment_tissues(
                preprocessed_path,
                segment_dir,
                region=args.region,
                method='nn'
            )
            print(f"Segmentation completed. Output saved to: {segmentation_path}")
        else:
            print("\nStep 2/4: Segmentation skipped.")
        
        # Step 3: Conversion
        print("\nStep 3/4: Converting MRI to synthetic CT...")
        synthetic_ct_path = convert_mri_to_ct(
            preprocessed_path,
            convert_dir,
            method=args.model,
            region=args.region,
            segmentation_path=segmentation_path
        )
        print(f"Conversion completed. Output saved to: {synthetic_ct_path}")
        
        # Step 4: Evaluation (if reference CT provided)
        if args.reference_ct:
            print("\nStep 4/4: Evaluating synthetic CT...")
            result = evaluate_synthetic_ct(
                synthetic_ct_path,
                args.reference_ct,
                eval_dir,
                segmentation_path=segmentation_path
            )
            
            print(f"Evaluation completed. Results saved to: {eval_dir}")
            print("\nMetrics:")
            for metric, value in result.metrics.items():
                print(f"  - {metric}: {value}")
        else:
            print("\nStep 4/4: Evaluation skipped (no reference CT provided).")
        
        print("\nPipeline completed successfully!")
        return 0
    
    except Exception as e:
        logging.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        print(f"Error during pipeline execution: {str(e)}")
        return 1


def process_visualize_command(args, config):
    """
    Process visualization command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Import here to avoid circular imports
    from app.utils.visualization import (
        plot_slice, plot_comparison, plot_3d_rendering, generate_visualization_report
    )
    
    # Validate input files
    if args.mri and not validate_input_file(args.mri):
        return 1
    
    if args.synthetic_ct and not validate_input_file(args.synthetic_ct):
        return 1
    
    if args.reference_ct and not validate_input_file(args.reference_ct):
        return 1
    
    if args.segmentation and not validate_input_file(args.segmentation):
        return 1
    
    # Ensure at least one input file
    if not (args.mri or args.synthetic_ct or args.reference_ct):
        print("Error: At least one input file (MRI, synthetic CT, or reference CT) must be provided")
        return 1
    
    # Ensure output directory exists
    output_dir = ensure_output_dir(args.output)
    if not output_dir:
        return 1
    
    try:
        # Process visualization based on type
        if args.type == "slice" or args.type == "all":
            # Generate slice visualizations
            if args.mri:
                plot_slice(args.mri, os.path.join(output_dir, "mri_slice.png"), title="MRI")
            
            if args.synthetic_ct:
                plot_slice(args.synthetic_ct, os.path.join(output_dir, "synthetic_ct_slice.png"), 
                           title="Synthetic CT", data_type="ct")
            
            if args.reference_ct:
                plot_slice(args.reference_ct, os.path.join(output_dir, "reference_ct_slice.png"), 
                           title="Reference CT", data_type="ct")
            
            if args.segmentation:
                plot_slice(args.segmentation, os.path.join(output_dir, "segmentation_slice.png"), 
                           title="Tissue Segmentation", data_type="segmentation")
        
        if args.type == "compare" or args.type == "all":
            # Generate comparison visualization
            if args.mri and args.synthetic_ct:
                plot_comparison(
                    mri_path=args.mri,
                    synthetic_ct_path=args.synthetic_ct,
                    reference_ct_path=args.reference_ct,
                    segmentation_path=args.segmentation,
                    output_path=os.path.join(output_dir, "comparison.png")
                )
        
        if args.type == "3d" or args.type == "all":
            # Generate 3D rendering
            if args.synthetic_ct:
                plot_3d_rendering(args.synthetic_ct, os.path.join(output_dir, "synthetic_ct_3d.png"))
            
            if args.reference_ct:
                plot_3d_rendering(args.reference_ct, os.path.join(output_dir, "reference_ct_3d.png"))
        
        if args.type == "all":
            # Generate comprehensive report
            generate_visualization_report(
                mri_path=args.mri,
                synthetic_ct_path=args.synthetic_ct,
                reference_ct_path=args.reference_ct,
                segmentation_path=args.segmentation,
                output_dir=output_dir
            )
        
        print(f"Visualization completed. Results saved to: {output_dir}")
        return 0
    
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}", exc_info=True)
        print(f"Error during visualization: {str(e)}")
        return 1


def process_patient_command(args, config):
    """
    Process patient data management command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Import here to avoid circular imports
    from app.utils.patient_utils import (
        import_patient_data, export_patient_data, 
        list_patient_data, delete_patient_data, anonymize_patient_data
    )
    
    try:
        if args.action == "import":
            # Validate input path
            if not args.input or not os.path.exists(args.input):
                print(f"Error: Input path does not exist: {args.input}")
                return 1
                
            # Parse modalities if provided
            modalities = args.modalities.split(',') if args.modalities else None
                
            # Import patient data
            result = import_patient_data(
                args.input, 
                anonymize=args.anonymize,
                modalities=modalities
            )
            
            print(f"Patient data imported successfully.")
            print(f"Patient ID: {result['patient_id']}")
            print(f"Imported files: {len(result['imported_files'])}")
            
            return 0
            
        elif args.action == "export":
            # Validate patient ID
            if not args.patient_id:
                print("Error: Patient ID is required for export action")
                return 1
                
            # Ensure output directory exists
            output_dir = ensure_output_dir(args.output)
            if not output_dir:
                return 1
                
            # Parse modalities if provided
            modalities = args.modalities.split(',') if args.modalities else None
                
            # Export patient data
            result = export_patient_data(
                args.patient_id,
                output_dir,
                anonymize=args.anonymize,
                modalities=modalities
            )
            
            print(f"Patient data exported successfully to: {output_dir}")
            print(f"Exported files: {len(result['exported_files'])}")
            
            return 0
            
        elif args.action == "list":
            # List patient data
            if args.patient_id:
                # List data for specific patient
                result = list_patient_data(args.patient_id)
                
                print(f"Patient ID: {args.patient_id}")
                print("Available data:")
                for item in result:
                    print(f"  - {item['modality']}: {item['date']} ({item['description']})")
            else:
                # List all patients
                patients = list_patient_data()
                
                print("Available patients:")
                for patient in patients:
                    print(f"  - {patient['patient_id']}: {patient['name']} ({patient['study_count']} studies)")
                    
            return 0
            
        elif args.action == "delete":
            # Validate patient ID
            if not args.patient_id:
                print("Error: Patient ID is required for delete action")
                return 1
                
            # Delete patient data
            result = delete_patient_data(args.patient_id)
            
            print(f"Patient data deleted successfully.")
            print(f"Deleted files: {result['deleted_files']}")
            
            return 0
            
        elif args.action == "anonymize":
            # Validate patient ID
            if not args.patient_id:
                print("Error: Patient ID is required for anonymize action")
                return 1
                
            # Anonymize patient data
            result = anonymize_patient_data(args.patient_id)
            
            print(f"Patient data anonymized successfully.")
            print(f"Anonymized files: {result['anonymized_files']}")
            
            return 0
        
        else:
            print(f"Error: Unknown action: {args.action}")
            return 1
    
    except Exception as e:
        logging.error(f"Error during patient data management: {str(e)}", exc_info=True)
        print(f"Error during patient data management: {str(e)}")
        return 1


def process_deploy_command(args, config):
    """
    Process deployment command.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration manager
        
    Returns:
        int: Exit code (0 for success)
    """
    # Import here to avoid circular imports
    from app.deployment.api_server import start_server
    
    try:
        print(f"Deploying {args.model} model for {args.region} region...")
        print(f"Server will be available at http://{args.host}:{args.port}")
        
        # Check if SSL is enabled
        use_ssl = args.ssl_cert is not None and args.ssl_key is not None
        if use_ssl:
            print("SSL enabled")
            
        # Start API server
        start_server(
            model=args.model,
            region=args.region,
            host=args.host,
            port=args.port,
            workers=args.workers,
            api_key=args.api_key,
            ssl_cert=args.ssl_cert,
            ssl_key=args.ssl_key,
            config=config
        )
        
        # Note: This function should never return as it starts a server
        # But just in case it does return, we'll consider it a success
        return 0
        
    except KeyboardInterrupt:
        print("\nServer shutdown requested via keyboard interrupt")
        return 0
        
    except Exception as e:
        logging.error(f"Error during server deployment: {str(e)}", exc_info=True)
        print(f"Error during server deployment: {str(e)}")
        return 1


def main():
    """
    Main entry point for the CLI.
    
    Returns:
        int: Exit code (0 for success)
    """
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # Handle no arguments case
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Load configuration
    config = load_config(args.config)
    update_config_from_args(args)
    
    # Process commands based on mode
    if args.mode == "config":
        return process_config_command(args)
    elif args.mode == "preprocess":
        return process_preprocess_command(args, config)
    elif args.mode == "multi-sequence":
        return process_multi_sequence_command(args, config)
    elif args.mode == "segment":
        return process_segment_command(args, config)
    elif args.mode == "convert":
        return process_convert_command(args, config)
    elif args.mode == "evaluate":
        return process_evaluate_command(args, config)
    elif args.mode == "pipeline":
        return process_pipeline_command(args, config)
    elif args.mode == "visualize":
        return process_visualize_command(args, config)
    elif args.mode == "patient":
        return process_patient_command(args, config)
    elif args.mode == "deploy":
        return process_deploy_command(args, config)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main()) 