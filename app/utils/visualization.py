#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Visualization utilities for MRI to CT conversion.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider, Button
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
from matplotlib.colors import LinearSegmentedColormap

from app.utils.io_utils import SyntheticCT, load_medical_image

# Set up logger
logger = logging.getLogger(__name__)

# Định nghĩa các giá trị window/level mặc định cho hiển thị
DEFAULT_WINDOW_LEVEL = {
    'mri': {'width': 500, 'level': 250},  # Window/level for MRI
    'ct': {'width': 400, 'level': 40},    # Window/level for CT (soft tissue)
    'ct_bone': {'width': 1800, 'level': 400},  # Window/level for CT (bone)
    'ct_lung': {'width': 1500, 'level': -600}  # Window/level for CT (lung)
}

# Định nghĩa các colormap
COLORMAPS = {
    'mri': 'gray',
    'ct': 'gray',
    'difference': 'RdBu_r',
    'segmentation': None  # Sẽ được tạo riêng
}


def get_slice_idx(image, orientation='axial', position=0.5):
    """
    Get slice index for a given position along an orientation.
    
    Args:
        image: SimpleITK image or numpy array
        orientation: 'axial', 'coronal', or 'sagittal'
        position: Relative position (0.0 to 1.0) along the axis
        
    Returns:
        Slice index
    """
    # Convert SimpleITK image to numpy array if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    
    # Get axis based on orientation
    axis = orientation_to_axis(orientation)
    
    # Calculate slice index
    n_slices = array.shape[axis]
    slice_idx = int(position * n_slices)
    
    # Ensure valid slice index
    slice_idx = max(0, min(slice_idx, n_slices - 1))
    
    return slice_idx


def orientation_to_axis(orientation):
    """
    Convert orientation to axis index.
    
    Args:
        orientation: 'axial', 'coronal', or 'sagittal'
        
    Returns:
        Axis index (0, 1, or 2)
    """
    orientation = orientation.lower()
    if orientation == 'axial':
        return 0  # Z-axis
    elif orientation == 'coronal':
        return 1  # Y-axis
    elif orientation == 'sagittal':
        return 2  # X-axis
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")


def apply_window_level(data, window=None, level=None, data_type='ct'):
    """
    Apply window/level to image data.
    
    Args:
        data: Image data as numpy array
        window: Window width (range of values to display)
        level: Window level (center of the window)
        data_type: Type of data ('ct', 'mri', 'segmentation')
        
    Returns:
        Windowed data normalized to [0, 1]
    """
    # Set default window/level based on data type
    if window is None or level is None:
        if data_type == 'ct':
            window = 500  # HU
            level = 50    # HU
        elif data_type == 'mri':
            window = np.max(data) - np.min(data)
            level = (np.max(data) + np.min(data)) / 2
        elif data_type == 'segmentation':
            return data  # No windowing for segmentation
    
    # Apply window/level
    data_windowed = np.clip(data, level - window/2, level + window/2)
    
    # Normalize to [0, 1]
    data_normalized = (data_windowed - (level - window/2)) / window
    
    return data_normalized


def create_segmentation_colormap(num_classes):
    """
    Create a colormap for segmentation visualization.
    
    Args:
        num_classes: Number of segmentation classes
        
    Returns:
        Matplotlib colormap
    """
    import matplotlib.colors as mcolors
    
    # Define base colors (excluding background which is transparent)
    base_colors = [
        [0, 0, 0, 0],       # Background (transparent)
        [1, 0, 0, 0.7],     # Class 1 (red)
        [0, 1, 0, 0.7],     # Class 2 (green)
        [0, 0, 1, 0.7],     # Class 3 (blue)
        [1, 1, 0, 0.7],     # Class 4 (yellow)
        [1, 0, 1, 0.7],     # Class 5 (magenta)
        [0, 1, 1, 0.7],     # Class 6 (cyan)
        [1, 0.5, 0, 0.7],   # Class 7 (orange)
        [0.5, 0, 1, 0.7],   # Class 8 (purple)
        [0, 0.5, 0, 0.7],   # Class 9 (dark green)
        [0.5, 0.5, 0.5, 0.7] # Class 10 (gray)
    ]
    
    # Extend colors if needed
    if num_classes > len(base_colors):
        import colorsys
        
        for i in range(len(base_colors), num_classes):
            # Generate evenly spaced HSV colors and convert to RGB
            h = (i - len(base_colors)) / (num_classes - len(base_colors))
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.8)
            base_colors.append([r, g, b, 0.7])
    
    # Create colormap
    cmap = mcolors.ListedColormap(base_colors[:num_classes])
    
    return cmap


def extract_slice(image, orientation='axial', slice_idx=None, position=0.5):
    """
    Extract a 2D slice from a 3D volume.
    
    Args:
        image: SimpleITK image or numpy array
        orientation: 'axial', 'coronal', or 'sagittal'
        slice_idx: Slice index (if None, use position)
        position: Relative position (0.0 to 1.0) along the axis
        
    Returns:
        2D slice as numpy array
    """
    # Convert SimpleITK image to numpy array if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    
    # Get axis based on orientation
    axis = orientation_to_axis(orientation)
    
    # Get slice index if not provided
    if slice_idx is None:
        slice_idx = get_slice_idx(array, orientation, position)
    
    # Extract slice
    if axis == 0:  # Axial
        slice_data = array[slice_idx, :, :]
    elif axis == 1:  # Coronal
        slice_data = array[:, slice_idx, :]
    elif axis == 2:  # Sagittal
        slice_data = array[:, :, slice_idx]
    
    return slice_data


def plot_slice(image_path, output_path=None, orientation='axial', slice_idx=None, position=0.5, 
              title=None, window=None, level=None, cmap=None, 
              data_type='ct', alpha=1.0, figsize=(8, 8), dpi=150):
    """
    Plot a 2D slice from a 3D volume.
    
    Args:
        image_path: Path to the image file or SimpleITK image
        output_path: Path to save the plot (None to display)
        orientation: 'axial', 'coronal', or 'sagittal'
        slice_idx: Slice index (if None, use position)
        position: Relative position (0.0 to 1.0) along the axis
        title: Plot title
        window: Window width for visualization
        level: Window level for visualization
        cmap: Colormap (None for default based on data_type)
        data_type: Type of data ('ct', 'mri', 'segmentation')
        alpha: Transparency level
        figsize: Figure size in inches
        dpi: Resolution in dots per inch
        
    Returns:
        Figure and axes objects
    """
    # Load image if path is provided
    if isinstance(image_path, str) or isinstance(image_path, Path):
        image = load_medical_image(image_path)
    else:
        image = image_path
    
    # Extract slice
    slice_data = extract_slice(image, orientation, slice_idx, position)
    
    # Apply window/level if needed
    if data_type != 'segmentation':
        slice_data = apply_window_level(slice_data, window, level, data_type)
    
    # Set colormap based on data type if not provided
    if cmap is None:
        if data_type == 'ct':
            cmap = 'gray'
        elif data_type == 'mri':
            cmap = 'gray'
        elif data_type == 'segmentation':
            # Count unique values for segmentation
            num_classes = len(np.unique(slice_data))
            cmap = create_segmentation_colormap(num_classes)
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if data_type == 'segmentation':
        # Plot segmentation with discrete values
        im = ax.imshow(slice_data, cmap=cmap, interpolation='nearest', alpha=alpha)
    else:
        # Plot medical image
        im = ax.imshow(slice_data, cmap=cmap, alpha=alpha)
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Remove axes
    ax.axis('off')
    
    # Add colorbar for segmentation
    if data_type == 'segmentation':
        plt.colorbar(im, ax=ax)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.tight_layout()
        return fig, ax


def plot_comparison(mri, ct, segmentation=None, synthetic_ct=None, orientation='axial', 
                   slice_idx=None, position=0.5, show_difference=True, figsize=(15, 10),
                   output_path=None, dpi=150, window=None, level=None):
    """
    Plot comparison of MRI, CT, synthetic CT, and segmentation.
    
    Args:
        mri: MRI image (SimpleITK image, numpy array, or path)
        ct: CT image (SimpleITK image, numpy array, or path)
        segmentation: Segmentation image (optional)
        synthetic_ct: Synthetic CT image (optional)
        orientation: 'axial', 'coronal', or 'sagittal'
        slice_idx: Slice index (if None, use position)
        position: Relative position (0.0 to 1.0) along the axis
        show_difference: Whether to show difference between CT and synthetic CT
        figsize: Figure size in inches
        output_path: Path to save the plot (None to display)
        dpi: Resolution in dots per inch
        window: Window width for visualization
        level: Window level for visualization
        
    Returns:
        Figure and axes objects or output path
    """
    # Load images if paths are provided
    if isinstance(mri, str) or isinstance(mri, Path):
        mri = load_medical_image(mri)
    
    if isinstance(ct, str) or isinstance(ct, Path):
        ct = load_medical_image(ct)
    
    if segmentation is not None and (isinstance(segmentation, str) or isinstance(segmentation, Path)):
        segmentation = load_medical_image(segmentation)
    
    if synthetic_ct is not None and (isinstance(synthetic_ct, str) or isinstance(synthetic_ct, Path)):
        synthetic_ct = load_medical_image(synthetic_ct)
    
    # Determine number of plots
    n_plots = 2  # MRI and CT
    if synthetic_ct is not None:
        n_plots += 1  # Synthetic CT
        if show_difference:
            n_plots += 1  # Difference
    
    # Create figure and axes
    if synthetic_ct is not None and show_difference:
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Extract slices
    mri_slice = extract_slice(mri, orientation, slice_idx, position)
    ct_slice = extract_slice(ct, orientation, slice_idx, position)
    
    # Apply window/level to MRI and CT
    mri_display = apply_window_level(mri_slice, window, level, 'mri')
    ct_display = apply_window_level(ct_slice, window, level, 'ct')
    
    # Plot MRI
    axes[0].imshow(mri_display, cmap='gray')
    axes[0].set_title('MRI')
    axes[0].axis('off')
    
    # Plot CT
    axes[1].imshow(ct_display, cmap='gray')
    axes[1].set_title('Reference CT')
    axes[1].axis('off')
    
    # Plot segmentation as overlay on MRI if provided
    if segmentation is not None:
        seg_slice = extract_slice(segmentation, orientation, slice_idx, position)
        num_classes = len(np.unique(seg_slice))
        seg_cmap = create_segmentation_colormap(num_classes)
        axes[0].imshow(seg_slice, cmap=seg_cmap, alpha=0.5)
    
    # Plot synthetic CT if provided
    if synthetic_ct is not None:
        synth_slice = extract_slice(synthetic_ct, orientation, slice_idx, position)
        synth_display = apply_window_level(synth_slice, window, level, 'ct')
        
        axes[2].imshow(synth_display, cmap='gray')
        axes[2].set_title('Synthetic CT')
        axes[2].axis('off')
        
        # Plot difference if requested
        if show_difference:
            diff_slice = synth_slice - ct_slice
            diff_max = max(abs(np.percentile(diff_slice, 1)), abs(np.percentile(diff_slice, 99)))
            im = axes[3].imshow(diff_slice, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
            axes[3].set_title('Difference')
            axes[3].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[3], label='HU Difference')
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.tight_layout()
        return fig, axes


def plot_3d_rendering(image_path, output_path=None, threshold=300, dpi=150):
    """
    Create a 3D surface rendering of a medical image.
    
    Args:
        image_path: Path to the image file or SimpleITK image
        output_path: Path to save the rendering (None to display)
        threshold: HU threshold for rendering (default: 300 for bone)
        dpi: Resolution in dots per inch
        
    Returns:
        Figure object or output path
    """
    try:
        # Import 3D visualization libraries
        from skimage import measure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = load_medical_image(image_path)
        else:
            image = image_path
        
        # Convert to numpy array
        array = sitk.GetArrayFromImage(image)
        
        # Create 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract surface mesh at the given threshold value
        verts, faces, _, _ = measure.marching_cubes(array, threshold)
        
        # Create mesh
        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_edgecolor('k')
        
        # Add mesh to plot
        ax.add_collection3d(mesh)
        
        # Set axis limits
        ax.set_xlim(0, array.shape[0])
        ax.set_ylim(0, array.shape[1])
        ax.set_zlim(0, array.shape[2])
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title
        ax.set_title(f'3D Rendering (Threshold: {threshold} HU)')
        
        # Save or display
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        else:
            return fig
            
    except ImportError as e:
        logger.error(f"Required libraries for 3D rendering not found: {str(e)}")
        raise ImportError("Required libraries for 3D rendering not found. Please install scikit-image.")
    except Exception as e:
        logger.error(f"Error creating 3D rendering: {str(e)}")
        raise


def generate_visualization_report(mri_path, synthetic_ct_path, reference_ct_path, segmentation_path, output_dir):
    """
    Generate comprehensive visualization report.
    
    Args:
        mri_path: Path to MRI file
        synthetic_ct_path: Path to synthetic CT file
        reference_ct_path: Path to reference CT file
        segmentation_path: Path to segmentation file
        output_dir: Output directory for report
        
    Returns:
        Dictionary with report information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        "image_paths": []
    }
    
    # Load images
    if mri_path:
        mri = load_medical_image(mri_path)
    else:
        mri = None
    
    if synthetic_ct_path:
        synthetic_ct = load_medical_image(synthetic_ct_path)
    else:
        synthetic_ct = None
    
    if reference_ct_path:
        reference_ct = load_medical_image(reference_ct_path)
    else:
        reference_ct = None
    
    if segmentation_path:
        segmentation = load_medical_image(segmentation_path)
    else:
        segmentation = None
    
    # Generate slice visualizations for each orientation
    orientations = ['axial', 'coronal', 'sagittal']
    positions = [0.3, 0.5, 0.7]  # Multiple positions
    
    for orientation in orientations:
        for position in positions:
            if mri is not None:
                # Create MRI slice
                output_path = os.path.join(output_dir, f"mri_{orientation}_{int(position*100)}.png")
                plot_slice(mri, output_path, orientation=orientation, position=position, 
                          title=f"MRI ({orientation}, position {int(position*100)}%)", 
                          data_type='mri')
                result["image_paths"].append(output_path)
            
            if synthetic_ct is not None:
                # Create synthetic CT slice
                output_path = os.path.join(output_dir, f"synthetic_ct_{orientation}_{int(position*100)}.png")
                plot_slice(synthetic_ct, output_path, orientation=orientation, position=position, 
                          title=f"Synthetic CT ({orientation}, position {int(position*100)}%)", 
                          data_type='ct')
                result["image_paths"].append(output_path)
            
            if reference_ct is not None:
                # Create reference CT slice
                output_path = os.path.join(output_dir, f"reference_ct_{orientation}_{int(position*100)}.png")
                plot_slice(reference_ct, output_path, orientation=orientation, position=position, 
                          title=f"Reference CT ({orientation}, position {int(position*100)}%)", 
                          data_type='ct')
                result["image_paths"].append(output_path)
            
            # Create comparison visualization
            if synthetic_ct is not None and (mri is not None or reference_ct is not None):
                output_path = os.path.join(output_dir, f"comparison_{orientation}_{int(position*100)}.png")
                plot_comparison(
                    mri if mri is not None else None,
                    reference_ct if reference_ct is not None else None,
                    segmentation,
                    synthetic_ct,
                    orientation=orientation,
                    position=position,
                    output_path=output_path
                )
                result["image_paths"].append(output_path)
    
    # Generate 3D visualizations if available
    if synthetic_ct is not None:
        try:
            output_path = os.path.join(output_dir, "synthetic_ct_3d.png")
            plot_3d_rendering(synthetic_ct, output_path)
            result["image_paths"].append(output_path)
        except Exception as e:
            logger.warning(f"Error generating 3D rendering for synthetic CT: {str(e)}")
    
    if reference_ct is not None:
        try:
            output_path = os.path.join(output_dir, "reference_ct_3d.png")
            plot_3d_rendering(reference_ct, output_path)
            result["image_paths"].append(output_path)
        except Exception as e:
            logger.warning(f"Error generating 3D rendering for reference CT: {str(e)}")
    
    return result


def generate_evaluation_report(mri, real_ct, synthetic_ct, segmentation=None, metrics=None, output_dir=None, dpi=150):
    """
    Generate evaluation report with visualizations and metrics.
    
    Args:
        mri: MRI image or path
        real_ct: Real CT image or path
        synthetic_ct: Synthetic CT image or path
        segmentation: Segmentation image or path (optional)
        metrics: Dictionary of evaluation metrics
        output_dir: Output directory for report
        dpi: Resolution in dots per inch
        
    Returns:
        Dictionary with report information
    """
    if output_dir is None:
        return {"image_paths": []}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    vis_result = generate_visualization_report(
        mri,
        synthetic_ct,
        real_ct,
        segmentation,
        output_dir
    )
    
    # Return report information
    return {
        "image_paths": vis_result["image_paths"],
        "metrics": metrics
    } 