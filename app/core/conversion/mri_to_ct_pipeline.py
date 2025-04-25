#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI to CT Pipeline module.

This module provides a unified interface for the complete pipeline from
MRI preprocessing to synthetic CT generation, including segmentation and conversion.
It serves as the main integration point between all core components.
"""

import os
import logging
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List

from app.core.preprocessing.preprocess_mri import preprocess_mri
from app.core.segmentation.segment_tissues import segment_tissues
from app.core.segmentation.bone_segmentation import segment_bones
from app.core.segmentation.air_cavity_segmentation import segment_air_cavities
from app.core.segmentation.soft_tissue_segmentation import segment_soft_tissues
from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct
from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT
from app.utils.config_utils import get_config

# Set up logger
logger = logging.getLogger(__name__)

class MRItoCTPipeline:
    """
    Complete MRI to CT conversion pipeline with preprocessing, segmentation, and conversion steps.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MRI to CT pipeline.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or get_config()
        self.segmentation_cache = {}
        self.preprocessed_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def preprocess(self, mri_path: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None,
                  apply_bias_field_correction: bool = True, 
                  apply_denoising: bool = True,
                  apply_normalization: bool = True,
                  params: Optional[Dict[str, Any]] = None) -> SyntheticCT:
        """
        Preprocess MRI image.
        
        Args:
            mri_path: Path to MRI image
            output_path: Path to save preprocessed image (optional)
            apply_bias_field_correction: Apply bias field correction
            apply_denoising: Apply denoising
            apply_normalization: Apply intensity normalization
            params: Additional parameters
            
        Returns:
            Preprocessed MRI as SyntheticCT object
        """
        self.logger.info(f"Preprocessing MRI: {mri_path}")
        
        # Convert path to string if needed
        if isinstance(mri_path, Path):
            mri_path = str(mri_path)
            
        if isinstance(output_path, Path):
            output_path = str(output_path)
            
        # Get parameters from config if not provided
        if params is None:
            params = self.config.get_preprocessing_params()
            
        # Preprocess MRI
        try:
            preprocessed_mri = preprocess_mri(
                mri_path,
                apply_bias_field_correction=apply_bias_field_correction,
                apply_denoising=apply_denoising,
                apply_normalization=apply_normalization,
                params=params
            )
            
            # Cache the preprocessed MRI
            self.preprocessed_cache[mri_path] = preprocessed_mri
            
            # Save if output path is provided
            if output_path:
                preprocessed_mri.save(output_path)
                self.logger.info(f"Saved preprocessed MRI to: {output_path}")
                
            return preprocessed_mri
            
        except Exception as e:
            self.logger.error(f"Error preprocessing MRI: {str(e)}")
            raise
            
    def segment(self, mri_data: Union[str, Path, SyntheticCT], 
               output_dir: Optional[Union[str, Path]] = None,
               method: str = 'auto',
               region: str = 'head',
               params: Optional[Dict[str, Any]] = None) -> Dict[str, SyntheticCT]:
        """
        Segment tissues from MRI.
        
        Args:
            mri_data: Path to MRI or preprocessed MRI object
            output_dir: Directory to save segmented tissues (optional)
            method: Segmentation method
            region: Anatomical region
            params: Additional parameters
            
        Returns:
            Dictionary of segmented tissues as SyntheticCT objects
        """
        self.logger.info(f"Segmenting tissues from MRI using method: {method}")
        
        # Load MRI if path is provided
        if isinstance(mri_data, (str, Path)):
            # Check if already preprocessed
            mri_path = str(mri_data) if isinstance(mri_data, Path) else mri_data
            if mri_path in self.preprocessed_cache:
                mri = self.preprocessed_cache[mri_path]
            else:
                # Load and preprocess
                mri = self.preprocess(mri_path)
        else:
            mri = mri_data
            
        # Get parameters from config if not provided
        if params is None:
            params = self.config.get_segmentation_params(method, region)
            
        # Segment tissues
        try:
            # First, segment bones
            self.logger.info("Segmenting bone structures")
            bone_mask = segment_bones(mri.image, method=method, params=params.get('bone_params', {}))
            
            # Segment air cavities
            self.logger.info("Segmenting air cavities")
            air_mask = segment_air_cavities(mri.image, method=method, bone_mask=bone_mask, 
                                           params=params.get('air_params', {}))
            
            # Segment soft tissues
            self.logger.info("Segmenting soft tissues")
            tissue_mask = segment_soft_tissues(mri.image, method=method, bone_mask=bone_mask, 
                                              air_mask=air_mask, params=params.get('tissue_params', {}))
            
            # Combined segmentation
            self.logger.info("Combining segmentations")
            segmentations = segment_tissues(mri.image, method=method, region=region, params=params)
            
            # Create SyntheticCT objects for each segmentation
            result = {}
            
            # Add bone mask
            bone_ct = SyntheticCT()
            bone_ct.image = bone_mask
            bone_ct.metadata = mri.metadata.copy()
            bone_ct.metadata['SegmentationType'] = 'bone'
            result['bone'] = bone_ct
            
            # Add air mask
            air_ct = SyntheticCT()
            air_ct.image = air_mask
            air_ct.metadata = mri.metadata.copy()
            air_ct.metadata['SegmentationType'] = 'air'
            result['air'] = air_ct
            
            # Add tissue mask
            tissue_ct = SyntheticCT()
            tissue_ct.image = tissue_mask
            tissue_ct.metadata = mri.metadata.copy()
            tissue_ct.metadata['SegmentationType'] = 'soft_tissue'
            result['soft_tissue'] = tissue_ct
            
            # Add other segmentations
            for tissue_name, mask in segmentations.items():
                if tissue_name not in result:
                    tissue_ct = SyntheticCT()
                    tissue_ct.image = mask
                    tissue_ct.metadata = mri.metadata.copy()
                    tissue_ct.metadata['SegmentationType'] = tissue_name
                    result[tissue_name] = tissue_ct
            
            # Cache segmentations
            self.segmentation_cache[str(mri_data)] = result
            
            # Save if output directory is provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                for tissue_name, segmentation in result.items():
                    output_path = output_dir / f"{tissue_name}_segmentation.nii.gz"
                    segmentation.save(str(output_path))
                    self.logger.info(f"Saved {tissue_name} segmentation to: {output_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error segmenting tissues: {str(e)}")
            raise
            
    def convert(self, mri_data: Union[str, Path, SyntheticCT],
               segmentation: Optional[Union[Dict[str, SyntheticCT], SyntheticCT]] = None,
               output_path: Optional[Union[str, Path]] = None,
               model_type: str = 'gan',
               region: str = 'head',
               params: Optional[Dict[str, Any]] = None) -> SyntheticCT:
        """
        Convert MRI to synthetic CT.
        
        Args:
            mri_data: Path to MRI or preprocessed MRI object
            segmentation: Segmentation data (optional)
            output_path: Path to save synthetic CT (optional)
            model_type: Conversion model type
            region: Anatomical region
            params: Additional parameters
            
        Returns:
            Synthetic CT
        """
        self.logger.info(f"Converting MRI to synthetic CT using model: {model_type}")
        
        # Load MRI if path is provided
        if isinstance(mri_data, (str, Path)):
            # Check if already preprocessed
            mri_path = str(mri_data) if isinstance(mri_data, Path) else mri_data
            if mri_path in self.preprocessed_cache:
                mri = self.preprocessed_cache[mri_path]
            else:
                # Load and preprocess
                mri = self.preprocess(mri_path)
        else:
            mri = mri_data
            
        # Get segmentation if not provided
        if segmentation is None:
            # Check if segmentation is in cache
            if str(mri_data) in self.segmentation_cache:
                segmentation = self.segmentation_cache[str(mri_data)]
            else:
                # Segment tissues
                segmentation = self.segment(mri, method='auto', region=region)
        
        # Handle case where segmentation is a dictionary
        seg_image = None
        if isinstance(segmentation, dict):
            # Combine segmentations into a single image
            if 'combined' in segmentation:
                seg_image = segmentation['combined'].image
            else:
                # Use most comprehensive segmentation
                for seg_name in ['all', 'full', 'complete', 'soft_tissue', 'bone']:
                    if seg_name in segmentation:
                        seg_image = segmentation[seg_name].image
                        break
        elif isinstance(segmentation, SyntheticCT):
            seg_image = segmentation.image
            
        # Get parameters from config if not provided
        if params is None:
            params = self.config.get_conversion_params(model_type, region)
            
        # Convert MRI to CT
        try:
            synthetic_ct = convert_mri_to_ct(
                mri.image,
                segmentation=seg_image,
                model_type=model_type,
                region=region,
                params=params
            )
            
            # Create SyntheticCT object
            sct = SyntheticCT()
            sct.image = synthetic_ct
            sct.metadata = mri.metadata.copy()
            sct.metadata['ConversionModel'] = model_type
            sct.metadata['AnatomicalRegion'] = region
            
            # Save if output path is provided
            if output_path:
                sct.save(output_path)
                self.logger.info(f"Saved synthetic CT to: {output_path}")
                
            return sct
            
        except Exception as e:
            self.logger.error(f"Error converting MRI to CT: {str(e)}")
            raise
            
    def run_complete_pipeline(self, mri_path: Union[str, Path],
                          output_dir: Union[str, Path],
                          reference_ct_path: Optional[Union[str, Path]] = None,
                          model_type: str = 'gan',
                          region: str = 'head',
                          apply_bias_field_correction: bool = True,
                          apply_denoising: bool = True,
                          apply_normalization: bool = True,
                          preprocessing_params: Optional[Dict[str, Any]] = None,
                          segmentation_params: Optional[Dict[str, Any]] = None,
                          conversion_params: Optional[Dict[str, Any]] = None,
                          evaluation_params: Optional[Dict[str, Any]] = None,
                          progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run complete end-to-end pipeline from MRI to synthetic CT with optional evaluation.
        
        Args:
            mri_path: Path to MRI image
            output_dir: Output directory for all results
            reference_ct_path: Path to reference CT for evaluation (optional)
            model_type: Conversion model type ('gan', 'cnn', 'atlas')
            region: Anatomical region ('head', 'pelvis', 'thorax')
            apply_bias_field_correction: Apply bias field correction in preprocessing
            apply_denoising: Apply denoising in preprocessing
            apply_normalization: Apply intensity normalization in preprocessing
            preprocessing_params: Preprocessing parameters (optional)
            segmentation_params: Segmentation parameters (optional)
            conversion_params: Conversion parameters (optional)
            evaluation_params: Evaluation parameters (optional)
            progress_callback: Callback function for progress updates (optional)
            
        Returns:
            Dictionary containing the pipeline results:
                - 'preprocessed_mri': Preprocessed MRI
                - 'segmentations': Dictionary of segmentation masks
                - 'synthetic_ct': Synthetic CT
                - 'evaluation_results': Evaluation results (if reference_ct_path provided)
                - 'output_paths': Dictionary of output file paths
        """
        self.logger.info(f"Running complete MRI to CT pipeline for: {mri_path}")
        
        # Function to update progress if callback provided
        def update_progress(value, message=None):
            if progress_callback:
                progress_callback(value, message)
            
        # Create output directory structure
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        preprocessed_dir = output_dir / "preprocessed"
        segmentation_dir = output_dir / "segmentation"
        synthetic_ct_dir = output_dir / "synthetic_ct"
        evaluation_dir = output_dir / "evaluation"
        
        for dir_path in [preprocessed_dir, segmentation_dir, synthetic_ct_dir, evaluation_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Track results and output paths
        results = {}
        output_paths = {}
        
        try:
            # Step 1: Preprocess MRI
            update_progress(10, "Step 1: Preprocessing MRI...")
            self.logger.info("Step 1: Preprocessing MRI")
            preprocessed_output = preprocessed_dir / "preprocessed_mri.nii.gz"
            preprocessed_mri = self.preprocess(
                mri_path,
                output_path=preprocessed_output,
                apply_bias_field_correction=apply_bias_field_correction,
                apply_denoising=apply_denoising,
                apply_normalization=apply_normalization,
                params=preprocessing_params
            )
            results['preprocessed_mri'] = preprocessed_mri
            output_paths['preprocessed_mri'] = str(preprocessed_output)
            
            # Step 2: Segment tissues
            update_progress(30, "Step 2: Segmenting tissues...")
            self.logger.info("Step 2: Segmenting tissues")
            segmentations = self.segment(
                preprocessed_mri,
                output_dir=segmentation_dir,
                method='auto',
                region=region,
                params=segmentation_params
            )
            results['segmentations'] = segmentations
            output_paths['segmentations'] = {
                tissue: str(segmentation_dir / f"{tissue}_segmentation.nii.gz") 
                for tissue in segmentations.keys()
            }
            
            # Step 3: Convert MRI to CT
            update_progress(60, f"Step 3: Converting MRI to CT using {model_type} model...")
            self.logger.info("Step 3: Converting MRI to CT")
            synthetic_ct_output = synthetic_ct_dir / "synthetic_ct.nii.gz"
            synthetic_ct = self.convert(
                preprocessed_mri,
                segmentation=segmentations,
                output_path=synthetic_ct_output,
                model_type=model_type,
                region=region,
                params=conversion_params
            )
            results['synthetic_ct'] = synthetic_ct
            output_paths['synthetic_ct'] = str(synthetic_ct_output)
            
            # Save main output
            complete_output = output_dir / "synthetic_ct.nii.gz"
            synthetic_ct.save(str(complete_output))
            output_paths['final_synthetic_ct'] = str(complete_output)
            
            # Step 4: Evaluate if reference CT is provided
            if reference_ct_path:
                update_progress(80, "Step 4: Evaluating synthetic CT...")
                self.logger.info("Step 4: Evaluating synthetic CT")
                
                # Default metrics if not specified
                if evaluation_params is None:
                    evaluation_params = {}
                
                metrics = evaluation_params.get('metrics', ['mae', 'mse', 'psnr', 'ssim'])
                regions = evaluation_params.get('regions', ['all', 'bone', 'soft_tissue', 'air'])
                
                try:
                    # Import evaluation module
                    from app.core.evaluation.evaluate_synthetic_ct import (
                        evaluate_synthetic_ct, 
                        create_tissue_masks,
                        EvaluationResult
                    )
                    
                    # Load reference CT
                    reference_ct = load_medical_image(reference_ct_path)
                    
                    # Evaluate synthetic CT
                    eval_results = evaluate_synthetic_ct(
                        synthetic_ct,
                        reference_ct,
                        metrics=metrics,
                        regions=regions,
                        config=self.config.config if hasattr(self.config, 'config') else None
                    )
                    
                    # Save evaluation results
                    results['evaluation_results'] = eval_results
                    
                    # Generate report and visualizations
                    eval_report_path = evaluation_dir / "evaluation_report.json"
                    
                    if isinstance(eval_results, EvaluationResult):
                        eval_results.save_report(eval_report_path)
                    else:
                        # If we got a dict instead of EvaluationResult object
                        import json
                        with open(eval_report_path, 'w') as f:
                            json.dump(eval_results, f, indent=2)
                            
                    output_paths['evaluation_report'] = str(eval_report_path)
                    
                    # Generate visualizations if applicable
                    try:
                        from app.core.evaluation.evaluate_synthetic_ct import visualize_evaluation
                        
                        # Generate comparison visualizations
                        vis_output_dir = evaluation_dir / "visualizations"
                        vis_output_dir.mkdir(exist_ok=True)
                        
                        vis_paths = visualize_evaluation(
                            synthetic_ct, 
                            reference_ct, 
                            eval_results,
                            str(vis_output_dir)
                        )
                        
                        if vis_paths:
                            output_paths['evaluation_visualizations'] = vis_paths
                        
                    except ImportError:
                        self.logger.warning("Visualization module not available")
                    except Exception as vis_error:
                        self.logger.warning(f"Error generating visualizations: {str(vis_error)}")
                        
                except ImportError as e:
                    self.logger.warning(f"Evaluation module not available: {str(e)}")
                except Exception as eval_error:
                    self.logger.error(f"Error evaluating synthetic CT: {str(eval_error)}")
                    results['evaluation_error'] = str(eval_error)
            
            # Final step: Update progress to complete
            update_progress(100, "Pipeline completed successfully!")
            self.logger.info("Pipeline completed successfully!")
            
            # Add output paths to results
            results['output_paths'] = output_paths
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            # Include error information in results
            results['error'] = str(e)
            # Return partial results
            return results


def run_pipeline(mri_path: Union[str, Path],
                output_dir: Union[str, Path],
                model_type: str = 'gan',
                region: str = 'head',
                config: Optional[Dict[str, Any]] = None) -> SyntheticCT:
    """
    Run the complete MRI to CT pipeline (convenience function).
    
    Args:
        mri_path: Path to MRI image
        output_dir: Output directory for all results
        model_type: Conversion model type
        region: Anatomical region
        config: Configuration dictionary
        
    Returns:
        Synthetic CT
    """
    pipeline = MRItoCTPipeline(config)
    return pipeline.run_complete_pipeline(
        mri_path,
        output_dir,
        model_type=model_type,
        region=region
    )


def run_complete_pipeline_with_evaluation(
    mri_path: Union[str, Path],
    output_dir: Union[str, Path],
    reference_ct_path: Optional[Union[str, Path]] = None,
    model_type: str = 'gan',
    region: str = 'head',
    apply_bias_field_correction: bool = True,
    apply_denoising: bool = True,
    apply_normalization: bool = True,
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Run the complete MRI to CT pipeline with evaluation (convenience function).
    
    Args:
        mri_path: Path to MRI image
        output_dir: Output directory for all results
        reference_ct_path: Path to reference CT for evaluation (optional)
        model_type: Conversion model type ('gan', 'cnn', 'atlas')
        region: Anatomical region ('head', 'pelvis', 'thorax')
        apply_bias_field_correction: Apply bias field correction
        apply_denoising: Apply denoising
        apply_normalization: Apply intensity normalization
        config: Configuration dictionary
        progress_callback: Callback function for progress updates
        
    Returns:
        Dictionary with pipeline results
    """
    pipeline = MRItoCTPipeline(config)
    return pipeline.run_complete_pipeline(
        mri_path=mri_path,
        output_dir=output_dir,
        reference_ct_path=reference_ct_path,
        model_type=model_type,
        region=region,
        apply_bias_field_correction=apply_bias_field_correction,
        apply_denoising=apply_denoising,
        apply_normalization=apply_normalization,
        progress_callback=progress_callback
    ) 