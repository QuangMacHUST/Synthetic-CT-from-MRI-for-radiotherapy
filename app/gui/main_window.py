#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Main GUI window for Synthetic CT application
"""

import os
import sys
import logging
import time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QTabWidget,
    QProgressBar, QSlider, QSpinBox, QDoubleSpinBox, QGroupBox,
    QRadioButton, QCheckBox, QMessageBox, QSplitter, QScrollArea,
    QGridLayout, QFormLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize
from PySide6.QtGui import QIcon, QPixmap, QFont

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils import (
    load_medical_image, 
    save_medical_image, 
    load_config,
    plot_image_slice,
    SyntheticCT
)
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues
from app.core.conversion import convert_mri_to_ct
from app.core.evaluation import evaluate_synthetic_ct


class SliceViewer(QWidget):
    """Widget to display and navigate through 3D medical image slices."""
    
    def __init__(self, parent=None):
        """Initialize the slice viewer widget."""
        super().__init__(parent)
        self.image_data = None
        self.current_slice = 0
        self.axis = 0  # 0: axial, 1: coronal, 2: sagittal
        self.axes_names = ["Axial", "Coronal", "Sagittal"]
        self.window_center = 40
        self.window_width = 400
        self.image_display = None
        self.initUI()
    
    def initUI(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        
        # Create figure and canvas for displaying images
        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Add canvas to layout
        layout.addWidget(self.canvas)
        
        # Axis selection
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("View:"))
        
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(self.axes_names)
        self.axis_combo.currentIndexChanged.connect(self.change_axis)
        axis_layout.addWidget(self.axis_combo)
        
        layout.addLayout(axis_layout)
        
        # Slice navigation
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel("Slice:"))
        
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.update_slice)
        slice_layout.addWidget(self.slice_slider)
        
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(0)
        self.slice_spinbox.valueChanged.connect(self.slice_slider.setValue)
        slice_layout.addWidget(self.slice_spinbox)
        
        layout.addLayout(slice_layout)
        
        # Window level adjustment
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("Window:"))
        
        self.window_center_spin = QSpinBox()
        self.window_center_spin.setRange(-1000, 3000)
        self.window_center_spin.setValue(self.window_center)
        self.window_center_spin.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(QLabel("Center:"))
        window_layout.addWidget(self.window_center_spin)
        
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(1, 4000)
        self.window_width_spin.setValue(self.window_width)
        self.window_width_spin.valueChanged.connect(self.update_window_level)
        window_layout.addWidget(QLabel("Width:"))
        window_layout.addWidget(self.window_width_spin)
        
        layout.addLayout(window_layout)
        
        # Preset buttons for common window levels
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        # CT presets
        brain_btn = QPushButton("Brain")
        brain_btn.clicked.connect(lambda: self.set_window_preset(40, 80))
        preset_layout.addWidget(brain_btn)
        
        bone_btn = QPushButton("Bone")
        bone_btn.clicked.connect(lambda: self.set_window_preset(400, 1500))
        preset_layout.addWidget(bone_btn)
        
        lung_btn = QPushButton("Lung")
        lung_btn.clicked.connect(lambda: self.set_window_preset(-600, 1500))
        preset_layout.addWidget(lung_btn)
        
        soft_tissue_btn = QPushButton("Soft Tissue")
        soft_tissue_btn.clicked.connect(lambda: self.set_window_preset(40, 400))
        preset_layout.addWidget(soft_tissue_btn)
        
        layout.addLayout(preset_layout)
        
        self.setLayout(layout)
    
    def set_image(self, image):
        """
        Set the image to display.
        
        Args:
            image: SimpleITK image or SyntheticCT object
        """
        if isinstance(image, SyntheticCT):
            image = image.image
        
        if image is not None:
            self.image_data = sitk.GetArrayFromImage(image)
            
            # Update slider and spinbox ranges
            max_slices = [self.image_data.shape[i] - 1 for i in range(3)]
            self.slice_slider.setMaximum(max_slices[self.axis])
            self.slice_spinbox.setMaximum(max_slices[self.axis])
            
            # Set current slice to middle
            self.current_slice = max_slices[self.axis] // 2
            self.slice_slider.setValue(self.current_slice)
            self.slice_spinbox.setValue(self.current_slice)
            
            # Update display
            self.update_display()
        else:
            self.image_data = None
            self.slice_slider.setMaximum(0)
            self.slice_spinbox.setMaximum(0)
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw()
    
    def change_axis(self, axis_idx):
        """
        Change the viewing axis.
        
        Args:
            axis_idx: Index of the axis to view (0: axial, 1: coronal, 2: sagittal)
        """
        if self.image_data is not None:
            self.axis = axis_idx
            
            # Update slider and spinbox ranges
            max_slice = self.image_data.shape[self.axis] - 1
            self.slice_slider.setMaximum(max_slice)
            self.slice_spinbox.setMaximum(max_slice)
            
            # Set current slice to middle
            self.current_slice = max_slice // 2
            self.slice_slider.setValue(self.current_slice)
            self.slice_spinbox.setValue(self.current_slice)
            
            # Update display
            self.update_display()
    
    def update_slice(self, slice_idx):
        """
        Update the displayed slice.
        
        Args:
            slice_idx: Index of the slice to display
        """
        if self.image_data is not None and slice_idx != self.current_slice:
            self.current_slice = slice_idx
            self.slice_spinbox.setValue(slice_idx)
            self.update_display()
    
    def update_window_level(self):
        """Update the window level settings."""
        self.window_center = self.window_center_spin.value()
        self.window_width = self.window_width_spin.value()
        self.update_display()
    
    def update_display(self):
        """Update the displayed image slice."""
        if self.image_data is None:
            return
        
        # Get the slice based on the current axis
        if self.axis == 0:  # Axial
            slice_data = self.image_data[self.current_slice, :, :]
        elif self.axis == 1:  # Coronal
            slice_data = self.image_data[:, self.current_slice, :]
        else:  # Sagittal
            slice_data = self.image_data[:, :, self.current_slice]
        
        # Clear previous plot
        self.ax.clear()
        
        # Calculate window level
        vmin = self.window_center - self.window_width // 2
        vmax = self.window_center + self.window_width // 2
        
        # Display the slice with window level
        self.image_display = self.ax.imshow(
            slice_data, 
            cmap='gray', 
            vmin=vmin, 
            vmax=vmax,
            interpolation='nearest'
        )
        
        # Add colorbar
        if hasattr(self, 'cbar'):
            self.cbar.remove()
        self.cbar = self.figure.colorbar(self.image_display, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)
        self.cbar.set_label('HU')
        
        # Set title
        self.ax.set_title(f"{self.axes_names[self.axis]} View - Slice {self.current_slice + 1}/{self.image_data.shape[self.axis]}")
        
        # Turn off axis
        self.ax.set_axis_off()
        
        # Update canvas
        self.canvas.draw()
    
    def set_window_preset(self, center, width):
        """
        Set a window level preset.
        
        Args:
            center: Window center value
            width: Window width value
        """
        self.window_center_spin.setValue(center)
        self.window_width_spin.setValue(width)


class ConversionThread(QThread):
    """Thread for running conversion process."""
    
    progress_signal = Signal(int, str)
    finished_signal = Signal(object, object, object)
    error_signal = Signal(str)
    
    def __init__(self, input_path, model_type='gan', region='head', evaluate=False, reference_path=None):
        """
        Initialize conversion thread.
        
        Args:
            input_path: Path to input MRI image
            model_type: Type of conversion model to use
            region: Anatomical region
            evaluate: Whether to evaluate the result
            reference_path: Path to reference CT image for evaluation
        """
        super().__init__()
        self.input_path = input_path
        self.model_type = model_type
        self.region = region
        self.evaluate = evaluate
        self.reference_path = reference_path
    
    def run(self):
        """Run the conversion process."""
        try:
            # Step 1: Load MRI image
            self.progress_signal.emit(10, "Loading MRI image...")
            original_mri = load_medical_image(self.input_path)
            
            # Step 2: Preprocess MRI
            self.progress_signal.emit(20, "Preprocessing MRI...")
            preprocessed_mri = preprocess_mri(self.input_path)
            
            # Step 3: Segment tissues
            self.progress_signal.emit(40, "Segmenting tissues...")
            segmented_tissues = segment_tissues(preprocessed_mri, region=self.region)
            
            # Step 4: Convert MRI to CT
            self.progress_signal.emit(60, f"Converting MRI to CT using {self.model_type} model...")
            synthetic_ct = convert_mri_to_ct(
                preprocessed_mri, 
                segmented_tissues, 
                model_type=self.model_type, 
                region=self.region
            )
            
            # Step 5: Evaluate if requested
            evaluation_results = None
            if self.evaluate and self.reference_path:
                self.progress_signal.emit(80, "Evaluating synthetic CT...")
                reference_ct = load_medical_image(self.reference_path)
                evaluation_results = evaluate_synthetic_ct(synthetic_ct, reference_ct, segmented_tissues)
            
            # Signal completion
            self.progress_signal.emit(100, "Conversion complete!")
            self.finished_signal.emit(preprocessed_mri, segmented_tissues, synthetic_ct)
            
        except Exception as e:
            logging.error(f"Error in conversion process: {str(e)}")
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    """Main window for the synthetic CT application."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        self.mri_path = None
        self.reference_ct_path = None
        self.output_dir = None
        self.config = load_config()
        self.preprocessed_mri = None
        self.segmented_tissues = None
        self.synthetic_ct = None
        self.initUI()
    
    def initUI(self):
        """Initialize the UI components."""
        # Set window properties
        self.setWindowTitle("Synthetic CT from MRI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # Add control panel
        self.createControlPanel(left_layout)
        
        # Add left panel to main layout
        main_layout.addWidget(left_panel, 1)
        
        # Create right panel for image viewers
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add image viewers
        self.createViewerTabs(right_layout)
        
        # Add right panel to main layout
        main_layout.addWidget(right_panel, 3)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def createControlPanel(self, parent_layout):
        """
        Create the control panel.
        
        Args:
            parent_layout: Parent layout to add controls to
        """
        # Input group
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        # MRI input
        mri_layout = QHBoxLayout()
        mri_layout.addWidget(QLabel("MRI Image:"))
        self.mri_path_label = QLabel("No file selected")
        self.mri_path_label.setWordWrap(True)
        mri_layout.addWidget(self.mri_path_label, 1)
        self.mri_browse_btn = QPushButton("Browse...")
        self.mri_browse_btn.clicked.connect(self.selectMRI)
        mri_layout.addWidget(self.mri_browse_btn)
        input_layout.addLayout(mri_layout)
        
        # Reference CT input (optional)
        ref_ct_layout = QHBoxLayout()
        ref_ct_layout.addWidget(QLabel("Reference CT:"))
        self.ref_ct_path_label = QLabel("No file selected (optional)")
        self.ref_ct_path_label.setWordWrap(True)
        ref_ct_layout.addWidget(self.ref_ct_path_label, 1)
        self.ref_ct_browse_btn = QPushButton("Browse...")
        self.ref_ct_browse_btn.clicked.connect(self.selectReferenceCT)
        ref_ct_layout.addWidget(self.ref_ct_browse_btn)
        input_layout.addLayout(ref_ct_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_label = QLabel("Default")
        self.output_dir_label.setWordWrap(True)
        output_layout.addWidget(self.output_dir_label, 1)
        self.output_dir_browse_btn = QPushButton("Browse...")
        self.output_dir_browse_btn.clicked.connect(self.selectOutputDir)
        output_layout.addWidget(self.output_dir_browse_btn)
        input_layout.addLayout(output_layout)
        
        input_group.setLayout(input_layout)
        parent_layout.addWidget(input_group)
        
        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        # Anatomical region
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.addItems(["head", "pelvis", "thorax"])
        region_layout.addWidget(self.region_combo)
        config_layout.addLayout(region_layout)
        
        # Conversion model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["atlas", "cnn", "gan"])
        self.model_combo.setCurrentText("gan")  # Default to GAN
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)
        
        # Processing options
        self.preproc_check = QCheckBox("Preprocessing")
        self.preproc_check.setChecked(True)
        config_layout.addWidget(self.preproc_check)
        
        self.segment_check = QCheckBox("Segmentation")
        self.segment_check.setChecked(True)
        config_layout.addWidget(self.segment_check)
        
        self.evaluate_check = QCheckBox("Evaluate results")
        self.evaluate_check.setChecked(False)
        config_layout.addWidget(self.evaluate_check)
        
        config_group.setLayout(config_layout)
        parent_layout.addWidget(config_group)
        
        # Advanced options (collapsed by default)
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        # Preprocessing options
        preproc_options = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        
        self.bias_field_check = QCheckBox("Bias Field Correction")
        self.bias_field_check.setChecked(True)
        preproc_layout.addWidget(self.bias_field_check)
        
        self.denoise_check = QCheckBox("Denoising")
        self.denoise_check.setChecked(True)
        preproc_layout.addWidget(self.denoise_check)
        
        self.normalize_check = QCheckBox("Intensity Normalization")
        self.normalize_check.setChecked(True)
        preproc_layout.addWidget(self.normalize_check)
        
        preproc_options.setLayout(preproc_layout)
        advanced_layout.addWidget(preproc_options)
        
        # Advanced slider for neural network parameters
        if hasattr(self.config, 'get') and self.config.get('conversion', {}).get('gan', {}).get('head', {}).get('batch_size'):
            batch_size = self.config.get('conversion', {}).get('gan', {}).get('head', {}).get('batch_size')
        else:
            batch_size = 1
        
        self.batch_size_layout = QHBoxLayout()
        self.batch_size_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 8)
        self.batch_size_spin.setValue(batch_size)
        self.batch_size_layout.addWidget(self.batch_size_spin)
        advanced_layout.addLayout(self.batch_size_layout)
        
        advanced_group.setLayout(advanced_layout)
        parent_layout.addWidget(advanced_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        # Run button
        self.run_btn = QPushButton("Run Conversion")
        self.run_btn.setEnabled(False)  # Disabled until MRI is selected
        self.run_btn.clicked.connect(self.runConversion)
        actions_layout.addWidget(self.run_btn)
        
        # Save outputs
        self.save_outputs_btn = QPushButton("Save Results")
        self.save_outputs_btn.setEnabled(False)  # Disabled until conversion is done
        self.save_outputs_btn.clicked.connect(self.saveResults)
        actions_layout.addWidget(self.save_outputs_btn)
        
        actions_group.setLayout(actions_layout)
        parent_layout.addWidget(actions_group)
        
        # Add spacer to push everything up
        parent_layout.addStretch()
    
    def createViewerTabs(self, parent_layout):
        """
        Create tabs for image viewers.
        
        Args:
            parent_layout: Parent layout to add tabs to
        """
        self.viewer_tabs = QTabWidget()
        
        # MRI viewer tab
        self.mri_tab = QWidget()
        mri_layout = QVBoxLayout(self.mri_tab)
        self.mri_viewer = SliceViewer()
        mri_layout.addWidget(self.mri_viewer)
        self.viewer_tabs.addTab(self.mri_tab, "MRI")
        
        # Segmentation viewer tab
        self.seg_tab = QWidget()
        seg_layout = QVBoxLayout(self.seg_tab)
        self.seg_viewer = SliceViewer()
        seg_layout.addWidget(self.seg_viewer)
        self.viewer_tabs.addTab(self.seg_tab, "Segmentation")
        
        # Synthetic CT viewer tab
        self.ct_tab = QWidget()
        ct_layout = QVBoxLayout(self.ct_tab)
        self.ct_viewer = SliceViewer()
        ct_layout.addWidget(self.ct_viewer)
        self.viewer_tabs.addTab(self.ct_tab, "Synthetic CT")
        
        # Comparison tab (shown when reference CT is available)
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)
        
        # Create side-by-side viewers
        comparison_splitter = QSplitter(Qt.Horizontal)
        
        # Reference CT viewer
        ref_panel = QWidget()
        ref_layout = QVBoxLayout(ref_panel)
        ref_layout.addWidget(QLabel("Reference CT"))
        self.ref_ct_viewer = SliceViewer()
        ref_layout.addWidget(self.ref_ct_viewer)
        comparison_splitter.addWidget(ref_panel)
        
        # Synthetic CT viewer for comparison
        synth_panel = QWidget()
        synth_layout = QVBoxLayout(synth_panel)
        synth_layout.addWidget(QLabel("Synthetic CT"))
        self.comparison_viewer = SliceViewer()
        synth_layout.addWidget(self.comparison_viewer)
        comparison_splitter.addWidget(synth_panel)
        
        comparison_layout.addWidget(comparison_splitter)
        
        # Add evaluation metrics panel
        metrics_group = QGroupBox("Evaluation Metrics")
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Metric"), 0, 0)
        metrics_layout.addWidget(QLabel("Overall"), 0, 1)
        metrics_layout.addWidget(QLabel("Bone"), 0, 2)
        metrics_layout.addWidget(QLabel("Soft Tissue"), 0, 3)
        metrics_layout.addWidget(QLabel("Air"), 0, 4)
        
        metrics_layout.addWidget(QLabel("MAE (HU)"), 1, 0)
        metrics_layout.addWidget(QLabel("MSE (HU²)"), 2, 0)
        metrics_layout.addWidget(QLabel("PSNR (dB)"), 3, 0)
        metrics_layout.addWidget(QLabel("SSIM"), 4, 0)
        
        # Create labels for metric values
        self.metric_labels = {}
        for i, metric in enumerate(["mae", "mse", "psnr", "ssim"]):
            row = i + 1
            self.metric_labels[f"{metric}_overall"] = QLabel("-")
            metrics_layout.addWidget(self.metric_labels[f"{metric}_overall"], row, 1)
            
            for j, tissue in enumerate(["bone", "soft_tissue", "air"]):
                col = j + 2
                self.metric_labels[f"{metric}_{tissue}"] = QLabel("-")
                metrics_layout.addWidget(self.metric_labels[f"{metric}_{tissue}"], row, col)
        
        metrics_group.setLayout(metrics_layout)
        comparison_layout.addWidget(metrics_group)
        
        self.viewer_tabs.addTab(self.comparison_tab, "Comparison")
        
        # Initially hide comparison tab
        self.viewer_tabs.setTabVisible(3, False)
        
        parent_layout.addWidget(self.viewer_tabs)
    
    def selectMRI(self):
        """Open file dialog to select input MRI file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select MRI Image",
            "",
            "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*)"
        )
        
        if file_path:
            self.mri_path = file_path
            self.mri_path_label.setText(file_path)
            self.updateRunButton()
            
            # Load and display MRI preview
            try:
                mri_image = load_medical_image(file_path)
                self.mri_viewer.set_image(mri_image)
                self.viewer_tabs.setCurrentIndex(0)  # Switch to MRI tab
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load MRI image: {str(e)}")
    
    def selectReferenceCT(self):
        """Open file dialog to select reference CT file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Reference CT Image",
            "",
            "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*)"
        )
        
        if file_path:
            self.reference_ct_path = file_path
            self.ref_ct_path_label.setText(file_path)
            
            # Enable evaluation checkbox
            self.evaluate_check.setEnabled(True)
            self.evaluate_check.setChecked(True)
            
            try:
                # Load and display reference CT
                ref_ct_image = load_medical_image(file_path)
                self.ref_ct_viewer.set_image(ref_ct_image)
                
                # Make comparison tab visible
                self.viewer_tabs.setTabVisible(3, True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load reference CT image: {str(e)}")
    
    def selectOutputDir(self):
        """Open directory dialog to select output directory."""
        dir_dialog = QFileDialog()
        dir_path = dir_dialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
    
    def updateRunButton(self):
        """Update the state of the Run button based on input validation."""
        self.run_btn.setEnabled(self.mri_path is not None)
    
    def runConversion(self):
        """Run the MRI to CT conversion process."""
        if not self.mri_path:
            QMessageBox.warning(self, "Warning", "Please select an MRI image first.")
            return
        
        # Disable Run button during processing
        self.run_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Create and start conversion thread
        self.conversion_thread = ConversionThread(
            input_path=self.mri_path,
            model_type=self.model_combo.currentText(),
            region=self.region_combo.currentText(),
            evaluate=self.evaluate_check.isChecked() and self.reference_ct_path is not None,
            reference_path=self.reference_ct_path
        )
        
        # Connect signals
        self.conversion_thread.progress_signal.connect(self.updateProgress)
        self.conversion_thread.finished_signal.connect(self.conversionFinished)
        self.conversion_thread.error_signal.connect(self.conversionError)
        
        # Start thread
        self.conversion_thread.start()
        
        # Update status
        self.statusBar().showMessage("Processing...")
    
    def saveResults(self):
        """Save the conversion results."""
        if not self.synthetic_ct:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return
        
        if not self.output_dir:
            # If no output directory is selected, ask for one
            dir_dialog = QFileDialog()
            dir_path = dir_dialog.getExistingDirectory(
                self,
                "Select Output Directory",
                ""
            )
            
            if dir_path:
                self.output_dir = dir_path
                self.output_dir_label.setText(dir_path)
            else:
                return
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Save synthetic CT
            ct_path = os.path.join(self.output_dir, "synthetic_ct.nii.gz")
            self.synthetic_ct.save(ct_path)
            
            # Save preprocessed MRI and segmentation if available
            if self.preprocessed_mri:
                mri_path = os.path.join(self.output_dir, "preprocessed_mri.nii.gz")
                save_medical_image(self.preprocessed_mri, mri_path)
            
            if self.segmented_tissues:
                seg_path = os.path.join(self.output_dir, "segmentation.nii.gz")
                save_medical_image(self.segmented_tissues, seg_path)
            
            QMessageBox.information(self, "Success", f"Results saved to {self.output_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
    
    @Slot(int, str)
    def updateProgress(self, value, message):
        """
        Update progress bar and status message.
        
        Args:
            value: Progress percentage (0-100)
            message: Status message
        """
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    @Slot(object, object, object)
    def conversionFinished(self, preprocessed_mri, segmented_tissues, synthetic_ct):
        """
        Handle conversion completion.
        
        Args:
            preprocessed_mri: Preprocessed MRI image
            segmented_tissues: Segmented tissues image
            synthetic_ct: Synthetic CT result
        """
        # Store results
        self.preprocessed_mri = preprocessed_mri
        self.segmented_tissues = segmented_tissues
        self.synthetic_ct = synthetic_ct
        
        # Update viewers
        self.mri_viewer.set_image(preprocessed_mri)
        self.seg_viewer.set_image(segmented_tissues)
        self.ct_viewer.set_image(synthetic_ct)
        
        if self.reference_ct_path and self.evaluate_check.isChecked():
            # Update comparison viewer
            self.comparison_viewer.set_image(synthetic_ct)
            
            # Display evaluation metrics if available
            try:
                reference_ct = load_medical_image(self.reference_ct_path)
                metrics = evaluate_synthetic_ct(synthetic_ct, reference_ct, segmented_tissues)
                
                # Update metric labels
                for metric in ["mae", "mse", "psnr", "ssim"]:
                    if metric in metrics["overall"]:
                        self.metric_labels[f"{metric}_overall"].setText(
                            f"{metrics['overall'][metric]:.2f}"
                        )
                    
                    if "by_tissue" in metrics:
                        for tissue in ["bone", "soft_tissue", "air"]:
                            if tissue in metrics["by_tissue"] and metric in metrics["by_tissue"][tissue]:
                                self.metric_labels[f"{metric}_{tissue}"].setText(
                                    f"{metrics['by_tissue'][tissue][metric]:.2f}"
                                )
                
                # Switch to comparison tab
                self.viewer_tabs.setCurrentIndex(3)
                
            except Exception as e:
                logging.error(f"Error evaluating results: {str(e)}")
        else:
            # Switch to synthetic CT tab
            self.viewer_tabs.setCurrentIndex(2)
        
        # Re-enable buttons
        self.run_btn.setEnabled(True)
        self.save_outputs_btn.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update status
        self.statusBar().showMessage("Conversion complete!")
    
    @Slot(str)
    def conversionError(self, error_message):
        """
        Handle conversion error.
        
        Args:
            error_message: Error message
        """
        # Re-enable UI
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Conversion failed!")
        
        # Show error message
        QMessageBox.critical(self, "Error", f"Conversion failed: {error_message}")


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())