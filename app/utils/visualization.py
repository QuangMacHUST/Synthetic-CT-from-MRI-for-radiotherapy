#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for medical images
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

from app.utils.io_utils import SyntheticCT

# Set up logger
logger = logging.getLogger(__name__)


def plot_image_slice(image, slice_idx=None, axis=0, cmap='gray', 
                   window_center=None, window_width=None, title=None):
    """
    Plot a single slice from a 3D medical image.
    
    Args:
        image: SimpleITK image or numpy array
        slice_idx: Index of the slice to display (default: middle slice)
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display
        window_center: Window center for display (default: auto)
        window_width: Window width for display (default: auto)
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK image to numpy if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
    else:
        array = image
        spacing = [1.0, 1.0, 1.0]
    
    # Get array dimensions and set default slice if not specified
    if slice_idx is None:
        slice_idx = array.shape[axis] // 2
    
    # Get slice based on axis
    if axis == 0:
        slice_data = array[slice_idx, :, :]
        aspect_ratio = spacing[2] / spacing[1]
    elif axis == 1:
        slice_data = array[:, slice_idx, :]
        aspect_ratio = spacing[2] / spacing[0]
    else:  # axis == 2
        slice_data = array[:, :, slice_idx]
        aspect_ratio = spacing[1] / spacing[0]
    
    # Set window level if not provided
    if window_center is None or window_width is None:
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if window_center is None:
            window_center = (max_val + min_val) / 2
        if window_width is None:
            window_width = max_val - min_val
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display slice with window level
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    im = ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratio)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax_names = ['Sagittal', 'Coronal', 'Axial']
        ax.set_title(f"{ax_names[axis]} view - Slice {slice_idx}")
    
    # Display grid
    ax.grid(False)
    
    return fig


def plot_comparison(image1, image2, slice_idx=None, axis=0, cmap='gray', 
                   window_center=None, window_width=None, titles=None):
    """
    Plot comparison of two 3D medical images.
    
    Args:
        image1: First SimpleITK image or numpy array
        image2: Second SimpleITK image or numpy array
        slice_idx: Index of the slice to display (default: middle slice)
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display
        window_center: Window center for display (default: auto)
        window_width: Window width for display (default: auto)
        titles: Titles for the two images (default: ['Image 1', 'Image 2'])
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK images to numpy if needed
    if isinstance(image1, sitk.Image):
        array1 = sitk.GetArrayFromImage(image1)
        spacing1 = image1.GetSpacing()
    else:
        array1 = image1
        spacing1 = [1.0, 1.0, 1.0]
    
    if isinstance(image2, sitk.Image):
        array2 = sitk.GetArrayFromImage(image2)
        spacing2 = image2.GetSpacing()
    else:
        array2 = image2
        spacing2 = [1.0, 1.0, 1.0]
    
    # Check if arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError(f"Images have different shapes: {array1.shape} vs {array2.shape}")
    
    # Get array dimensions and set default slice if not specified
    if slice_idx is None:
        slice_idx = array1.shape[axis] // 2
    
    # Get slices based on axis
    if axis == 0:
        slice_data1 = array1[slice_idx, :, :]
        slice_data2 = array2[slice_idx, :, :]
        aspect_ratio = spacing1[2] / spacing1[1]
    elif axis == 1:
        slice_data1 = array1[:, slice_idx, :]
        slice_data2 = array2[:, slice_idx, :]
        aspect_ratio = spacing1[2] / spacing1[0]
    else:  # axis == 2
        slice_data1 = array1[:, :, slice_idx]
        slice_data2 = array2[:, :, slice_idx]
        aspect_ratio = spacing1[1] / spacing1[0]
    
    # Set window level if not provided
    if window_center is None or window_width is None:
        min_val = min(np.min(slice_data1), np.min(slice_data2))
        max_val = max(np.max(slice_data1), np.max(slice_data2))
        if window_center is None:
            window_center = (max_val + min_val) / 2
        if window_width is None:
            window_width = max_val - min_val
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display slices with window level
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    
    im1 = axes[0].imshow(slice_data1, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratio)
    im2 = axes[1].imshow(slice_data2, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratio)
    
    # Add colorbar
    plt.colorbar(im1, ax=axes[0], label='Intensity')
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # Set titles
    if titles is None:
        titles = ['Image 1', 'Image 2']
    
    ax_names = ['Sagittal', 'Coronal', 'Axial']
    axes[0].set_title(f"{titles[0]} - {ax_names[axis]} view - Slice {slice_idx}")
    axes[1].set_title(f"{titles[1]} - {ax_names[axis]} view - Slice {slice_idx}")
    
    # Display grid
    axes[0].grid(False)
    axes[1].grid(False)
    
    # Add global title
    plt.suptitle(f"Comparison - {ax_names[axis]} view - Slice {slice_idx}")
    
    plt.tight_layout()
    
    return fig


def plot_difference(image1, image2, slice_idx=None, axis=0, cmap='RdBu_r', 
                   window_center=0, window_width=None, title=None):
    """
    Plot difference between two 3D medical images.
    
    Args:
        image1: First SimpleITK image or numpy array
        image2: Second SimpleITK image or numpy array
        slice_idx: Index of the slice to display (default: middle slice)
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display (default: red-blue for differences)
        window_center: Window center for display (default: 0 for difference)
        window_width: Window width for display (default: auto)
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK images to numpy if needed
    if isinstance(image1, sitk.Image):
        array1 = sitk.GetArrayFromImage(image1)
        spacing = image1.GetSpacing()
    else:
        array1 = image1
        spacing = [1.0, 1.0, 1.0]
    
    if isinstance(image2, sitk.Image):
        array2 = sitk.GetArrayFromImage(image2)
    else:
        array2 = image2
    
    # Check if arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError(f"Images have different shapes: {array1.shape} vs {array2.shape}")
    
    # Calculate difference
    diff_array = array1 - array2
    
    # Get array dimensions and set default slice if not specified
    if slice_idx is None:
        slice_idx = diff_array.shape[axis] // 2
    
    # Get slice based on axis
    if axis == 0:
        slice_data = diff_array[slice_idx, :, :]
        aspect_ratio = spacing[2] / spacing[1]
    elif axis == 1:
        slice_data = diff_array[:, slice_idx, :]
        aspect_ratio = spacing[2] / spacing[0]
    else:  # axis == 2
        slice_data = diff_array[:, :, slice_idx]
        aspect_ratio = spacing[1] / spacing[0]
    
    # Set window width if not provided
    if window_width is None:
        max_abs_diff = np.max(np.abs(slice_data))
        window_width = 2 * max_abs_diff
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display difference with symmetric colormap around zero
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    
    im = ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratio)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Difference')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax_names = ['Sagittal', 'Coronal', 'Axial']
        ax.set_title(f"Difference - {ax_names[axis]} view - Slice {slice_idx}")
    
    # Calculate statistics
    mean_diff = np.mean(diff_array)
    std_diff = np.std(diff_array)
    max_diff = np.max(diff_array)
    min_diff = np.min(diff_array)
    mae = np.mean(np.abs(diff_array))
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_diff:.2f}\n"
        f"Std: {std_diff:.2f}\n"
        f"Min: {min_diff:.2f}\n"
        f"Max: {max_diff:.2f}\n"
        f"MAE: {mae:.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    # Display grid
    ax.grid(False)
    
    return fig


def create_interactive_viewer(image, axis=0, cmap='gray', window_center=None, window_width=None, title=None):
    """
    Create an interactive slice viewer for a 3D medical image.
    
    Args:
        image: SimpleITK image or numpy array
        axis: Initial axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display
        window_center: Initial window center for display (default: auto)
        window_width: Initial window width for display (default: auto)
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK image to numpy if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
    else:
        array = image
        spacing = [1.0, 1.0, 1.0]
    
    # Set default window level if not provided
    if window_center is None:
        window_center = (np.max(array) + np.min(array)) / 2
    if window_width is None:
        window_width = np.max(array) - np.min(array)
    
    # Calculate aspect ratio based on spacing
    aspect_ratios = [
        spacing[2] / spacing[1],  # Sagittal
        spacing[2] / spacing[0],  # Coronal
        spacing[1] / spacing[0]   # Axial
    ]
    
    # Create figure and initial plot
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Get initial slice based on axis
    slice_idx = array.shape[axis] // 2
    
    if axis == 0:
        slice_data = array[slice_idx, :, :]
    elif axis == 1:
        slice_data = array[:, slice_idx, :]
    else:  # axis == 2
        slice_data = array[:, :, slice_idx]
    
    # Display initial slice
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    im = ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratios[axis])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Intensity')
    
    # Set title
    ax_names = ['Sagittal', 'Coronal', 'Axial']
    if title:
        ax.set_title(f"{title} - {ax_names[axis]} view - Slice {slice_idx}")
    else:
        ax.set_title(f"{ax_names[axis]} view - Slice {slice_idx}")
    
    # Create axes for slice slider
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_slice = Slider(
        ax=ax_slice,
        label='Slice',
        valmin=0,
        valmax=array.shape[axis] - 1,
        valinit=slice_idx,
        valstep=1
    )
    
    # Create axes for window level sliders
    ax_center = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_width = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    # Create window level sliders
    data_min = np.min(array)
    data_max = np.max(array)
    data_range = data_max - data_min
    
    slider_center = Slider(
        ax=ax_center,
        label='Window Center',
        valmin=data_min,
        valmax=data_max,
        valinit=window_center
    )
    
    slider_width = Slider(
        ax=ax_width,
        label='Window Width',
        valmin=0,
        valmax=data_range * 2,
        valinit=window_width
    )
    
    # Create axes for axis selection buttons
    btn_axes = []
    for i in range(3):
        btn_axes.append(plt.axes([0.05 + i * 0.05, 0.1, 0.04, 0.04]))
    
    # Create axis selection buttons
    buttons = [Button(ax, label) for ax, label in zip(btn_axes, ['S', 'C', 'A'])]
    
    # Create update function for sliders and buttons
    def update(val):
        # Get current values
        current_slice = int(slider_slice.val)
        current_center = slider_center.val
        current_width = slider_width.val
        
        # Get slice data based on current axis
        nonlocal axis
        if axis == 0:
            slice_data = array[current_slice, :, :]
        elif axis == 1:
            slice_data = array[:, current_slice, :]
        else:  # axis == 2
            slice_data = array[:, :, current_slice]
        
        # Update image
        vmin = current_center - current_width / 2
        vmax = current_center + current_width / 2
        im.set_data(slice_data)
        im.set_clim(vmin, vmax)
        
        # Update title
        if title:
            ax.set_title(f"{title} - {ax_names[axis]} view - Slice {current_slice}")
        else:
            ax.set_title(f"{ax_names[axis]} view - Slice {current_slice}")
        
        fig.canvas.draw_idle()
    
    # Create function to handle axis button clicks
    def change_axis(event):
        nonlocal axis
        if event.inaxes == btn_axes[0]:
            new_axis = 0  # Sagittal
        elif event.inaxes == btn_axes[1]:
            new_axis = 1  # Coronal
        else:
            new_axis = 2  # Axial
        
        if new_axis != axis:
            axis = new_axis
            
            # Update slider range
            slider_slice.valmax = array.shape[axis] - 1
            slider_slice.ax.set_xlim(0, array.shape[axis] - 1)
            slider_slice.set_val(array.shape[axis] // 2)
            
            # Update aspect ratio
            im.set_aspect(aspect_ratios[axis])
            
            # Update display
            update(None)
    
    # Register callbacks
    slider_slice.on_changed(update)
    slider_center.on_changed(update)
    slider_width.on_changed(update)
    
    for button in buttons:
        button.on_clicked(change_axis)
    
    # Store sliders and buttons in figure to prevent garbage collection
    fig.sliders = [slider_slice, slider_center, slider_width]
    fig.buttons = buttons
    
    return fig


def save_comparison_figure(image1, image2, output_path, slice_idx=None, axis=0, cmap='gray',
                          window_center=None, window_width=None, titles=None, dpi=300):
    """
    Create and save a comparison figure of two 3D medical images.
    
    Args:
        image1: First SimpleITK image or numpy array
        image2: Second SimpleITK image or numpy array
        output_path: Path to save the figure
        slice_idx: Index of the slice to display (default: middle slice)
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display
        window_center: Window center for display (default: auto)
        window_width: Window width for display (default: auto)
        titles: Titles for the two images (default: ['Image 1', 'Image 2'])
        dpi: DPI for the output figure
        
    Returns:
        Path to the saved figure
    """
    # Create comparison figure
    fig = plot_comparison(
        image1, image2, 
        slice_idx=slice_idx, 
        axis=axis, 
        cmap=cmap, 
        window_center=window_center, 
        window_width=window_width, 
        titles=titles
    )
    
    # Create difference figure
    diff_fig = plot_difference(
        image1, image2, 
        slice_idx=slice_idx, 
        axis=axis
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save comparison figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    # Save difference figure
    diff_path = os.path.splitext(output_path)[0] + '_diff' + os.path.splitext(output_path)[1]
    diff_fig.savefig(diff_path, dpi=dpi, bbox_inches='tight')
    plt.close(diff_fig)
    
    logger.info(f"Saved comparison figure to {output_path}")
    logger.info(f"Saved difference figure to {diff_path}")
    
    return output_path


def create_montage(image, n_slices=9, axis=0, cmap='gray', window_center=None, window_width=None, title=None):
    """
    Create a montage of slices from a 3D medical image.
    
    Args:
        image: SimpleITK image or numpy array
        n_slices: Number of slices to include in the montage
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        cmap: Colormap for display
        window_center: Window center for display (default: auto)
        window_width: Window width for display (default: auto)
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK image to numpy if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
    else:
        array = image
        spacing = [1.0, 1.0, 1.0]
    
    # Get aspect ratio based on spacing
    if axis == 0:
        aspect_ratio = spacing[2] / spacing[1]
    elif axis == 1:
        aspect_ratio = spacing[2] / spacing[0]
    else:  # axis == 2
        aspect_ratio = spacing[1] / spacing[0]
    
    # Determine grid layout
    grid_size = int(np.ceil(np.sqrt(n_slices)))
    
    # Calculate slice indices
    slice_count = array.shape[axis]
    if n_slices > slice_count:
        n_slices = slice_count
        logger.warning(f"Reduced number of slices to {n_slices} (total available)")
    
    # Select evenly spaced slices
    if slice_count <= n_slices:
        indices = list(range(slice_count))
    else:
        step = slice_count / n_slices
        indices = [int(i * step) for i in range(n_slices)]
    
    # Set window level if not provided
    if window_center is None or window_width is None:
        min_val = np.min(array)
        max_val = np.max(array)
        if window_center is None:
            window_center = (max_val + min_val) / 2
        if window_width is None:
            window_width = max_val - min_val
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Display slices
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2
    
    for i, slice_idx in enumerate(indices):
        if i >= len(axes):
            break
            
        if axis == 0:
            slice_data = array[slice_idx, :, :]
        elif axis == 1:
            slice_data = array[:, slice_idx, :]
        else:  # axis == 2
            slice_data = array[:, :, slice_idx]
        
        im = axes[i].imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect_ratio)
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')
    
    # Add colorbar
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    plt.colorbar(im, cax=cb_ax, label='Intensity')
    
    # Set global title
    ax_names = ['Sagittal', 'Coronal', 'Axial']
    if title:
        plt.suptitle(f"{title} - {ax_names[axis]} view Montage")
    else:
        plt.suptitle(f"{ax_names[axis]} view Montage")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.9)
    
    return fig


def plot_histogram(image, mask=None, bins=100, title=None):
    """
    Plot histogram of intensities in a medical image.
    
    Args:
        image: SimpleITK image or numpy array
        mask: Optional binary mask to select specific voxels
        bins: Number of histogram bins
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Convert SimpleITK image to numpy if needed
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    
    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, sitk.Image):
            mask_array = sitk.GetArrayFromImage(mask)
        else:
            mask_array = mask
        
        # Ensure mask is binary
        if not np.all(np.unique(mask_array) == np.array([0, 1])):
            mask_array = (mask_array > 0).astype(int)
        
        # Apply mask
        masked_array = array[mask_array > 0]
    else:
        masked_array = array.flatten()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(masked_array, bins=bins, alpha=0.7, color='blue')
    
    # Add vertical lines for key statistics
    mean_val = np.mean(masked_array)
    median_val = np.median(masked_array)
    std_val = np.std(masked_array)
    
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.1f}')
    
    # Add statistics as text
    stats_text = (
        f"Mean: {mean_val:.1f}\n"
        f"Median: {median_val:.1f}\n"
        f"Std: {std_val:.1f}\n"
        f"Min: {np.min(masked_array):.1f}\n"
        f"Max: {np.max(masked_array):.1f}\n"
        f"Count: {len(masked_array)}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
    
    # Set labels
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Intensity Histogram')
    
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_evaluation_results(metrics_dict, output_path=None, title="Evaluation Results"):
    """
    Plot evaluation results for synthetic CT comparison.
    
    Args:
        metrics_dict: Dictionary containing evaluation metrics
        output_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    # Extract metrics
    metrics = metrics_dict.copy()
    
    # Handle different metric structures
    if 'by_tissue' in metrics:
        # Tissue-specific metrics
        tissue_metrics = metrics.pop('by_tissue')
        
        # Create figure with subplots for overall and tissue-specific metrics
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot overall metrics
        metrics_names = list(metrics.keys())
        metrics_values = [metrics[m] for m in metrics_names]
        
        axes[0].bar(metrics_names, metrics_values, color='skyblue')
        axes[0].set_title('Overall Metrics')
        axes[0].set_ylabel('Value')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels for readability
        axes[0].set_xticklabels(metrics_names, rotation=45, ha='right')
        
        # Plot tissue-specific metrics
        tissues = list(tissue_metrics.keys())
        metric_types = ['mae', 'mse', 'psnr']  # Common metrics to plot
        
        # Prepare data for grouped bar chart
        x = np.arange(len(tissues))
        width = 0.25
        
        # Plot each metric type as a group
        for i, metric in enumerate(metric_types):
            if all(metric in tissue_metrics[t] for t in tissues):
                values = [tissue_metrics[t][metric] for t in tissues]
                axes[1].bar(x + (i - 1) * width, values, width, label=metric.upper())
        
        # Set axis labels and title
        axes[1].set_xlabel('Tissue Type')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Metrics by Tissue Type')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tissues, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
    else:
        # Only overall metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = list(metrics.keys())
        metrics_values = [metrics[m] for m in metrics_names]
        
        ax.bar(metrics_names, metrics_values, color='skyblue')
        ax.set_title('Evaluation Metrics')
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotate x labels for readability
        ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    
    # Set global title
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved evaluation results plot to {output_path}")
    
    return fig 