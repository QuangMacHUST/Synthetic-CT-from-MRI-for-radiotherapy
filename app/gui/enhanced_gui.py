#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced GUI for Synthetic CT application with advanced options
"""

import os
import sys
import logging
import numpy as np
import SimpleITK as sitk
try:
    import cv2 
except ImportError:
    logging.warning("OpenCV (cv2) not available. Using fallback image resizing.")
    cv2 = None
from pathlib import Path

# Import Qt modules with proper paths
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

# Try importing VTK related modules, but continue if not available
try:
    import vtk
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    
# Try importing scikit-image for 3D visualization fallback
try:
    from skimage import measure
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

from app.utils import load_medical_image, SyntheticCT, setup_logging, get_config
try:
    from app.core.preprocessing.preprocess_mri import preprocess_mri
except ImportError:
    logging.warning("Could not import preprocess_mri function")
    preprocess_mri = None
    
try:
    from app.core.segmentation.segment_tissues import segment_tissues
except ImportError:
    logging.warning("Could not import segment_tissues function")
    segment_tissues = None
    
try:
    from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct
except ImportError:
    logging.warning("Could not import convert_mri_to_ct function")
    convert_mri_to_ct = None

# Import the integrated pipeline
try:
    from app.core.conversion.mri_to_ct_pipeline import MRItoCTPipeline, run_pipeline
except ImportError:
    logging.warning("Could not import MRItoCTPipeline")
    MRItoCTPipeline = None
    run_pipeline = None

class ProcessingThread(QThread):
    """Thread for processing operations"""
    
    finished = Signal(object)  # Signal for successful completion
    error = Signal(str)        # Signal for error
    progress = Signal(int)     # Signal for progress updates
    status = Signal(str)       # Signal for status messages
    
    def __init__(self, operation, *args, **kwargs):
        """Initialize with operation and arguments"""
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        
        # Add progress callback to kwargs if applicable
        if 'progress_callback' not in self.kwargs:
            self.kwargs['progress_callback'] = self.report_progress
        
        self._is_interrupted = False
        
    def run(self):
        """Run the operation"""
        try:
            result = self.operation(*self.args, **self.kwargs)
            if not self._is_interrupted:
                self.finished.emit(result)
        except Exception as e:
            if not self._is_interrupted:
                self.error.emit(str(e))
                
    def report_progress(self, value, message=None):
        """Report progress and optional status message"""
        self.progress.emit(value)
        if message:
            self.status.emit(message)
            
    def requestInterruption(self):
        """Request interruption of the thread"""
        super().requestInterruption()
        self._is_interrupted = True
        # If operation supports interruption
        if 'interrupt_flag' in self.kwargs:
            self.kwargs['interrupt_flag'] = True

# Define a custom NoWheelLabel that completely blocks wheel events
class NoWheelLabel(QLabel):
    """Custom QLabel that completely blocks wheel events"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setProperty("wheelScrollLines", 0)
        # Ensure mouse events aren't propagated up the widget hierarchy
        self.setAttribute(Qt.WA_NoMousePropagation, True)
    
    def wheelEvent(self, event):
        # Always accept wheel events and don't propagate them
        event.accept()
        return

class SimpleImageViewer(QWidget):
    """Simple viewer for displaying 3D medical images with axial, coronal, and sagittal views"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Create view widgets
        self.create_view_widgets()
        main_layout.addWidget(self.view_splitter)
        
        # Create control panel
        self.create_control_panel()
        main_layout.addWidget(self.control_panel)
        
        # Add position label
        self.position_label = QLabel("Position: X: 0, Y: 0, Z: 0")
        main_layout.addWidget(self.position_label)
        
        # Create info text area
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(80)
        main_layout.addWidget(self.info_text)
        
        # Add 3D view button
        self.view_3d_btn = QPushButton("Show 3D View")
        self.view_3d_btn.setEnabled(False)  # Enable only when image is loaded
        self.view_3d_btn.clicked.connect(self.show_3d_view)
        main_layout.addWidget(self.view_3d_btn)
        
        # Initialize properties
        self.image = None
        self.current_slices = {'axial': 0, 'coronal': 0, 'sagittal': 0}
        self.active_view = 'axial'
        self.window = 2000
        self.level = 0
        
        # Disable wheeling/zooming on the entire widget and all children
        self.setAttribute(Qt.WA_NoMousePropagation, True)
        self.setProperty("wheelScrollLines", 0)
        
        # IMPORTANT: Override wheel event for the main widget to absolutely prevent zooming
        self.wheelEvent = self.blockWheelEvent
        
        # Global event filter that will catch all wheel events in the application
        self.global_filter = WheelEventFilter()
        QCoreApplication.instance().installEventFilter(self.global_filter)
        
        self.setMinimumSize(800, 600)
    
    def create_view_widgets(self):
        """Create view widgets for axial, coronal, and sagittal views"""
        # Views layout
        views_layout = QHBoxLayout()
        
        # Create each view (axial, coronal, sagittal)
        self.views = {}
        view_names = ['axial', 'coronal', 'sagittal']
        
        for view_name in view_names:
            # Create frame for view
            view_frame = QFrame()
            view_frame.setFrameShape(QFrame.Box)
            view_frame.setLineWidth(1)
            view_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            # IMPORTANT: Block wheel events in the frame to prevent zooming
            view_frame.wheelEvent = lambda event: event.accept()
            view_frame.setAttribute(Qt.WA_NoMousePropagation, True)
            
            # Create layout for view
            frame_layout = QVBoxLayout(view_frame)
            frame_layout.setContentsMargins(2, 2, 2, 2)
            
            # Title label for view
            title_label = QLabel(view_name.capitalize())
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("font-weight: bold")
            
            # Image label for view - using our custom NoWheelLabel
            image_label = NoWheelLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            image_label.setMinimumSize(200, 200)
            
            # Critical: Disable wheelScrollLines completely across all widgets
            image_label.setProperty("wheelScrollLines", 0)
            
            # Add slice navigation controls
            slice_layout = QHBoxLayout()
            slice_layout.setContentsMargins(0, 0, 0, 0)
            
            # Slice spinner
            slice_spin = QSpinBox()
            slice_spin.setRange(1, 1)
            slice_spin.setEnabled(False)
            slice_spin.setMinimumWidth(50)
            
            # Slice slider
            slice_slider = QSlider(Qt.Horizontal)
            slice_slider.setRange(1, 1)
            slice_slider.setEnabled(False)
            
            # Disable wheel events on spinbox and slider
            slice_spin.wheelEvent = lambda event: event.accept()
            slice_slider.wheelEvent = lambda event: event.accept()
            
            slice_layout.addWidget(QLabel("Slice:"))
            slice_layout.addWidget(slice_spin)
            slice_layout.addWidget(slice_slider)
            
            # Connect slice navigation controls
            slice_spin.valueChanged.connect(
                lambda value, name=view_name: self.set_slice(name, value)
            )
            slice_slider.valueChanged.connect(
                lambda value, name=view_name: self.set_slice(name, value)
            )
            
            # Store references to view components
            self.views[view_name] = {
                'frame': view_frame,
                'title': title_label,
                'label': image_label,
                'spinner': slice_spin,
                'slider': slice_slider
            }
            
            # Add components to frame layout
            frame_layout.addWidget(title_label)
            frame_layout.addWidget(image_label)
            frame_layout.addLayout(slice_layout)
            
            # Add frame to views layout
            views_layout.addWidget(view_frame)
        
        # Create a container widget to hold the layout
        views_container = QWidget()
        views_container.setLayout(views_layout)
        
        # Block wheel events on the container
        views_container.wheelEvent = lambda event: event.accept()
        views_container.setAttribute(Qt.WA_NoMousePropagation, True)
        
        # Create splitter for views
        self.view_splitter = QSplitter()
        self.view_splitter.addWidget(views_container)
        
        # Block wheel events on the splitter to prevent zooming
        self.view_splitter.wheelEvent = lambda event: event.accept()
        self.view_splitter.setAttribute(Qt.WA_NoMousePropagation, True)
        
        self.view_splitter.setStretchFactor(0, 1)
        self.view_splitter.setStretchFactor(1, 1)
        self.view_splitter.setStretchFactor(2, 1)
    
    def eventFilter(self, obj, event):
        """Handle mouse events for labels"""
        if isinstance(obj, QLabel) and obj in [self.views[view]['label'] for view in self.views]:
            # Find current view
            current_view = None
            for view_name, view_data in self.views.items():
                if view_data['label'] == obj:
                    current_view = view_name
                    break
            
            if current_view:
                # Handle wheel events to scroll through slices without zooming
                if event.type() == QEvent.Type.Wheel:
                    # Completely capture and block the wheel event to prevent zooming
                    event.accept()
                    
                    # Update active view
                    self.active_view = current_view
                    self.view_combo.setCurrentText(current_view)
                    self.update_active_view_highlight()
                    
                    # Get scroll direction and change slice
                    delta = event.angleDelta().y()
                    
                    # More responsive scrolling - use a larger step if shift is pressed
                    step = 3 if event.modifiers() & Qt.ShiftModifier else 1
                    
                    if delta > 0:
                        # Scroll up: previous slice
                        self.change_slice(-step)
                    else:
                        # Scroll down: next slice
                        self.change_slice(step)
                    
                    # CRITICAL: Return True to completely block event from propagating further
                    return True
                
                # Handle mouse clicks to select the view
                elif event.type() == QEvent.Type.MouseButtonPress:
                    self.active_view = current_view
                    self.view_combo.setCurrentText(current_view)
                    self.update_active_view_highlight()
                    # Don't return True here to allow other mouse events to work
        
        # Pass event to parent class for default handling
        return super().eventFilter(obj, event)

    def set_image(self, image):
        """Set the image for display"""
        try:
            self.image = image
            self.image_info = {}
            
            # Extract image info if available (for SimpleITK images)
            if hasattr(image, 'GetMetaDataKeys'):
                try:
                    keys = image.GetMetaDataKeys()
                    info_text = "Image Information:\n"
                    
                    # Extract common DICOM tags
                    important_tags = {
                        '0010|0010': 'Patient Name',
                        '0010|0020': 'Patient ID',
                        '0008|0060': 'Modality',
                        '0008|0020': 'Study Date',
                        '0018|0050': 'Slice Thickness'
                    }
                    
                    for tag in important_tags:
                        if tag in keys:
                            value = image.GetMetaData(tag)
                            info_text += f"{important_tags[tag]}: {value}\n"
                    
                    # Add size and spacing info
                    size = image.GetSize()
                    spacing = image.GetSpacing()
                    info_text += f"Dimensions: {size[0]}x{size[1]}x{size[2]}\n"
                    info_text += f"Spacing: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm\n"
                    
                    self.info_text.setText(info_text)
                except Exception as e:
                    logging.warning(f"Error extracting image metadata: {str(e)}")
            
            # Get dimensions from image
            if hasattr(image, 'GetPixelIDValue'):
                # Get array from SimpleITK image
                arr = sitk.GetArrayFromImage(image)
                self.image_array = arr
                
                # Numpy array from SimpleITK has dimensions [depth, height, width]
                if arr.ndim >= 3:
                    depth, height, width = arr.shape
                elif arr.ndim == 2:
                    # Handle 2D images as single slice
                    height, width = arr.shape
                    depth = 1
                    # Reshape to 3D for consistency
                    arr = arr.reshape(1, height, width)
                    self.image_array = arr
                
                # Configure slice navigation for each view
                slices = {'axial': depth, 'coronal': height, 'sagittal': width}
                for view_name, num_slices in slices.items():
                    view = self.views[view_name]
                    view['max_slice'] = num_slices - 1
                    
                    # Update slice navigation controls
                    view['spinner'].setRange(1, num_slices)
                    view['spinner'].setEnabled(True)
                    view['slider'].setRange(1, num_slices)
                    view['slider'].setEnabled(True)
                    
                    # Set middle slice initially
                    middle_slice = max(0, num_slices // 2 - 1)
                    self.current_slices[view_name] = middle_slice
                    
                    # Enable 3D view
                    self.view_3d_btn.setEnabled(True)
                
                # Update all views
                self.update_display()
                
                # Install event filter on image labels to capture mouse events
                for view_name, view_data in self.views.items():
                    label = view_data['label']
                    
                    # Set minimum size and alignment
                    label.setMinimumSize(200, 200)
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    
                    # Install event filter for mouse events
                    label.installEventFilter(self)
                    
                    # Set strong focus to capture key events
                    label.setFocusPolicy(Qt.StrongFocus)
            
            # Update display for all views
            self.update_display()
            
            return True
        
        except Exception as e:
            logging.error(f"Error setting image: {str(e)}")
            QMessageBox.critical(None, "Error", f"Error setting image: {str(e)}")
            return False
    
    def change_slice(self, delta):
        """Change the slice in the active view by the given delta"""
        if self.active_view in self.current_slices:
            current_slice = self.current_slices[self.active_view]
            max_slice = self.views[self.active_view]['max_slice']
            
            # Compute new slice
            new_slice = current_slice + delta
            
            # Ensure within bounds
            if new_slice < 0:
                new_slice = 0
            elif new_slice > max_slice:
                new_slice = max_slice
                
            # Update only if changed
            if new_slice != current_slice:
                # Set internally (0-based)
                self.update_slice_view(self.active_view, new_slice)
                
                # Update UI controls (1-based)
                view_data = self.views[self.active_view]
                view_data['spinner'].setValue(new_slice + 1)
                view_data['slider'].setValue(new_slice + 1)
                
                # Update current slice
                self.current_slices[self.active_view] = new_slice
                
                # Update position information
                self.update_position_info()
    
    def blockWheelEvent(self, event):
        # Just accept the event and do nothing, preventing propagation
        event.accept()
    
    def create_control_panel(self):
        """Create control panel for window/level and view selection"""
        # Add controls
        controls_layout = QHBoxLayout()
        
        # Window/Level controls
        window_level_group = QGroupBox("Window/Level")
        window_level_layout = QHBoxLayout(window_level_group)
        
        window_level_layout.addWidget(QLabel("Window:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 4000)
        self.window_spin.setValue(2000)
        self.window_spin.valueChanged.connect(self.update_window_level)
        self.window_spin.setToolTip("Adjust window width (contrast)")
        window_level_layout.addWidget(self.window_spin)
        
        window_level_layout.addWidget(QLabel("Level:"))
        self.level_spin = QSpinBox()
        self.level_spin.setRange(-1000, 3000)
        self.level_spin.setValue(0)
        self.level_spin.valueChanged.connect(self.update_window_level)
        self.level_spin.setToolTip("Adjust window level (brightness)")
        window_level_layout.addWidget(self.level_spin)
        
        # Add presets for window/level
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(['Custom', 'Brain', 'Bone', 'Lung', 'Soft Tissue'])
        self.preset_combo.currentTextChanged.connect(self.apply_window_level_preset)
        self.preset_combo.setToolTip("Select a preset window/level for different tissues")
        window_level_layout.addWidget(QLabel("Preset:"))
        window_level_layout.addWidget(self.preset_combo)
        
        controls_layout.addWidget(window_level_group)
        
        # View selection
        view_group = QGroupBox("Navigation")
        view_layout = QHBoxLayout(view_group)
        view_layout.addWidget(QLabel("Active View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(['axial', 'coronal', 'sagittal'])
        self.view_combo.currentTextChanged.connect(self.set_active_view)
        self.view_combo.setToolTip("Select the active view for keyboard and mouse wheel navigation")
        view_layout.addWidget(self.view_combo)
        controls_layout.addWidget(view_group)
        
        self.control_panel = QWidget()
        self.control_panel.setLayout(controls_layout)
    
    def update_window_level(self):
        """Update window and level values for image display"""
        if hasattr(self, 'image') and self.image is not None:
            self.window = self.window_spin.value()
            self.level = self.level_spin.value()
            self.update_display()
    
    def update_display(self):
        """Update the display of all views with current window/level and slices"""
        if hasattr(self, 'image') and self.image is not None:
            for view_name in self.views:
                slice_idx = self.current_slices[view_name]
                self.update_slice_view(view_name, slice_idx)
    
    def update_slice_view(self, view_name, slice_idx=None):
        """Update an individual slice view with the current window/level settings"""
        if hasattr(self, 'image') and self.image is not None and view_name in self.views:
            # Get slice index - either passed directly or from the current UI controls
            if slice_idx is not None:
                # Ensure slice index is within bounds
                view_data = self.views[view_name]
                max_slice = view_data['max_slice']
                if slice_idx < 0:
                    slice_idx = 0
                elif slice_idx > max_slice:
                    slice_idx = max_slice
                
                # Update current slice index
                self.current_slices[view_name] = slice_idx
                
                # Update slice indicators (1-based for UI)
                view_data['slider'].setValue(slice_idx + 1)
                view_data['spinner'].setValue(slice_idx + 1)
                
                current_slice = slice_idx
            else:
                # Get current slice from spinner
                slice_spinbox = getattr(self, f"{view_name}_slice_spinbox", None)
                if slice_spinbox:
                    current_slice = slice_spinbox.value() - 1  # Convert from 1-based UI to 0-based index
                else:
                    current_slice = self.current_slices.get(view_name, 0)
            
            # Extract the slice based on view orientation
            if view_name == 'axial':
                slice_array = self.image_array[current_slice, :, :]
                total_slices = self.image_array.shape[0]
            elif view_name == 'coronal':
                slice_array = self.image_array[:, current_slice, :]
                total_slices = self.image_array.shape[1]
            elif view_name == 'sagittal':
                slice_array = self.image_array[:, :, current_slice]
                total_slices = self.image_array.shape[2]
            else:
                return
            
            # Apply window/level
            min_val = self.level - (self.window / 2)
            max_val = self.level + (self.window / 2)
            
            # Normalize to 0-255 for display
            normalized = np.clip((slice_array - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
            
            # Create QImage
            height, width = normalized.shape
            bytes_per_line = width
            q_img = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(q_img)
            
            # Draw slice number on the pixmap
            painter = QPainter(pixmap)
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QPen(Qt.green))
            painter.drawText(10, 20, f"Slice: {current_slice + 1}/{total_slices}")
            painter.end()
            
            # Update label
            label = view_data['label']
            label.setPixmap(pixmap)
            
            # Update position information if method exists
            if hasattr(self, 'update_position_info'):
                self.update_position_info()
    
    def apply_window_level_preset(self, preset_name):
        """Apply predefined window/level presets"""
        presets = {
            'Brain': {'window': 80, 'level': 40},
            'Bone': {'window': 2000, 'level': 500},
            'Lung': {'window': 1500, 'level': -500},
            'Soft Tissue': {'window': 350, 'level': 50}
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.window_spin.setValue(preset['window'])
            self.level_spin.setValue(preset['level'])
        self.update_window_level()
        
    def set_active_view(self, view_name):
        """Set the active view for keyboard navigation"""
        if view_name in self.views:
            self.active_view = view_name
            self.update_active_view_highlight()
            
    def update_active_view_highlight(self):
        """Highlight the active view"""
        for name, view_data in self.views.items():
            frame = view_data['frame']
            if name == self.active_view:
                frame.setFrameShape(QFrame.Box)
                frame.setLineWidth(3)
                frame.setStyleSheet("QFrame { border: 3px solid #3498db; }")
            else:
                frame.setFrameShape(QFrame.Box)
                frame.setLineWidth(1)
                frame.setStyleSheet("QFrame { border: 1px solid #999999; }")
                
    def show_3d_view(self):
        """Show 3D visualization of the volume"""
        if not hasattr(self, 'image') or self.image is None:
            QMessageBox.warning(None, "Warning", "No image loaded. Please load an image first.")
            return
            
        # Create a new dialog for 3D view
        dialog = QDialog(self)
        dialog.setWindowTitle("3D Visualization")
        dialog.setMinimumSize(800, 600)
        dialog_layout = QVBoxLayout(dialog)
        
        try:
            # Check if VTK is available
            if 'VTK_AVAILABLE' in globals() and VTK_AVAILABLE:
                # Create VTK renderer and interactor
                vtk_widget = QVTKRenderWindowInteractor(dialog)
                dialog_layout.addWidget(vtk_widget)
                
                # Get image array
                image_array = self.image_array
                
                # Create renderer
                renderer = vtk.vtkRenderer()
                vtk_widget.GetRenderWindow().AddRenderer(renderer)
                interactor = vtk_widget.GetRenderWindow().GetInteractor()
                
                # Create volume from image data
                volume = self.create_vtk_volume(image_array)
                if volume:
                    renderer.AddVolume(volume)
                    
                    # Set background and camera
                    renderer.SetBackground(0.1, 0.1, 0.1)
                    renderer.ResetCamera()
                    
                    # Add axes
                    axes = vtk.vtkAxesActor()
                    axes.SetTotalLength(50, 50, 50)
                    axes.SetShaftTypeToCylinder()
                    axes.SetCylinderRadius(0.02)
                    axes.GetXAxisCaptionActor2D().SetCaption("X")
                    axes.GetYAxisCaptionActor2D().SetCaption("Y")
                    axes.GetZAxisCaptionActor2D().SetCaption("Z")
                    
                    # Create orientation marker
                    marker_widget = vtk.vtkOrientationMarkerWidget()
                    marker_widget.SetOrientationMarker(axes)
                    marker_widget.SetInteractor(interactor)
                    marker_widget.SetViewport(0.0, 0.0, 0.3, 0.3)
                    marker_widget.SetEnabled(1)
                    marker_widget.InteractiveOff()
                    
                    # Start interaction
                    interactor.Initialize()
                    interactor.Start()
                    dialog.exec()
                else:
                    QMessageBox.warning(dialog, "Warning", "Failed to create volume for 3D visualization.")
            elif 'SKIMAGE_AVAILABLE' in globals() and SKIMAGE_AVAILABLE:
                # Fallback to matplotlib for 3D visualization
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.figure import Figure
                import matplotlib.pyplot as plt
                
                # Create empty widget
                mpl_widget = QWidget()
                mpl_layout = QVBoxLayout(mpl_widget)
                dialog_layout.addWidget(mpl_widget)
                
                # Create figure and canvas
                fig = Figure(figsize=(8, 6))
                canvas = FigureCanvas(fig)
                mpl_layout.addWidget(canvas)
                
                # Create 3D axes
                ax = fig.add_subplot(111, projection='3d')
                
                # Get image array for visualization
                # Extract isosurface from array
                try:
                    # Downsample array for better performance
                    downsampled = self.image_array[::2, ::2, ::2]
                    threshold = np.percentile(downsampled, 75)  # Use 75th percentile as threshold
                    
                    # Create isosurface
                    verts, faces, _, _ = measure.marching_cubes(downsampled, threshold)
                    
                    # Create mesh
                    mesh = Poly3DCollection(verts[faces], alpha=0.7)
                    face_color = [0.5, 0.5, 1.0]
                    mesh.set_facecolor(face_color)
                    mesh.set_edgecolor('k')
                    ax.add_collection3d(mesh)
                    
                    # Set limits
                    ax.set_xlim(0, downsampled.shape[0])
                    ax.set_ylim(0, downsampled.shape[1])
                    ax.set_zlim(0, downsampled.shape[2])
                    
                    # Set labels
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    
                    # Update canvas
                    canvas.draw()
                    dialog.exec()
                except Exception as e:
                    QMessageBox.warning(dialog, "Error", f"Error creating 3D surface: {str(e)}")
            else:
                QMessageBox.warning(dialog, "Warning", "3D visualization requires VTK or scikit-image, which are not available.")
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Error creating 3D view: {str(e)}")
            dialog.close()

    def create_vtk_volume(self, image_array):
        """Create VTK volume from array data"""
        try:
            # Create VTK data
            data_importer = vtk.vtkImageImport()
            
            # Convert to float and scale to 0-255 for better visualization
            data = image_array.astype(np.float32)
            min_val = np.min(data)
            max_val = np.max(data)
            data = ((data - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            data_string = data.tostring()
            
            # Set VTK data properties
            data_importer.CopyImportVoidPointer(data_string, len(data_string))
            data_importer.SetDataScalarTypeToUnsignedChar()
            data_importer.SetNumberOfScalarComponents(1)
            
            # Set dimensions
            dims = data.shape
            data_importer.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            data_importer.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
            
            # Create transfer function for opacity
            opacity_tf = vtk.vtkPiecewiseFunction()
            opacity_tf.AddPoint(0, 0.0)
            opacity_tf.AddPoint(50, 0.0)
            opacity_tf.AddPoint(100, 0.1)
            opacity_tf.AddPoint(150, 0.3)
            opacity_tf.AddPoint(200, 0.5)
            opacity_tf.AddPoint(255, 0.8)
            
            # Create transfer function for color
            color_tf = vtk.vtkColorTransferFunction()
            color_tf.AddRGBPoint(0, 0.0, 0.0, 0.0)
            color_tf.AddRGBPoint(64, 0.5, 0.0, 0.0)
            color_tf.AddRGBPoint(128, 0.9, 0.5, 0.0)
            color_tf.AddRGBPoint(192, 1.0, 0.9, 0.5)
            color_tf.AddRGBPoint(255, 1.0, 1.0, 1.0)
            
            # Create volume properties
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(color_tf)
            volume_property.SetScalarOpacity(opacity_tf)
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
            
            # Create mapper
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputConnection(data_importer.GetOutputPort())
            
            # Create volume
            volume = vtk.vtkVolume()
            volume.SetMapper(mapper)
            volume.SetProperty(volume_property)
            
            return volume
        except Exception as e:
            logging.error(f"Error creating VTK volume: {str(e)}")
            return None
    
    def set_slice(self, view_name, slice_index):
        """Set the current slice for a specific view"""
        if view_name in self.views and hasattr(self, 'image') and self.image is not None:
            # Adjust for UI displaying 1-based index
            if isinstance(slice_index, int) and slice_index > 0:
                slice_index -= 1
                
            # Update the slice view
            self.update_slice_view(view_name, slice_index)

    def update_position_info(self, x=None, y=None):
        """Update position information display"""
        if not hasattr(self, 'image') or self.image is None:
            return
        
        # Get current slices
        axial_slice = self.current_slices.get('axial', 0)
        coronal_slice = self.current_slices.get('coronal', 0)
        sagittal_slice = self.current_slices.get('sagittal', 0)
        
        # Get image dimensions
        if hasattr(self.image, 'GetSize'):
            img_size = self.image.GetSize()
            dimensions = f"Dimensions: {img_size[0]}x{img_size[1]}x{img_size[2]}"
        else:
            dimensions = ""
        
        # Update position label
        position_text = f"Position: X: {sagittal_slice+1}, Y: {coronal_slice+1}, Z: {axial_slice+1} | {dimensions}"
        self.position_label.setText(position_text)
        
        # Highlight current slice info
        if self.active_view == 'axial':
            self.views['axial']['title'].setText(f"Axial (Z: {axial_slice+1})")
        elif self.active_view == 'coronal':
            self.views['coronal']['title'].setText(f"Coronal (Y: {coronal_slice+1})")
        elif self.active_view == 'sagittal':
            self.views['sagittal']['title'].setText(f"Sagittal (X: {sagittal_slice+1})")
    
    def change_slice(self, delta):
        """Change the slice in the active view by the given delta"""
        if self.active_view in self.current_slices:
            current_slice = self.current_slices[self.active_view]
            max_slice = self.views[self.active_view]['max_slice']
            
            # Compute new slice
            new_slice = current_slice + delta
            
            # Ensure within bounds
            if new_slice < 0:
                new_slice = 0
            elif new_slice > max_slice:
                new_slice = max_slice
                
            # Update only if changed
            if new_slice != current_slice:
                # Set internally (0-based)
                self.update_slice_view(self.active_view, new_slice)
                
                # Update UI controls (1-based)
                view_data = self.views[self.active_view]
                view_data['spinner'].setValue(new_slice + 1)
                view_data['slider'].setValue(new_slice + 1)
                
                # Update current slice
                self.current_slices[self.active_view] = new_slice
                
                # Update position information
                self.update_position_info()

    def update_slice_view(self, view_name):
        """Update the slice view for the specified view"""
        if not hasattr(self, 'image') or self.image is None:
            return

        if view_name not in self.views:
            return

        slice_spinbox = getattr(self, f"{view_name}_slice_spinbox")
        current_slice = slice_spinbox.value() - 1  # Convert from 1-based UI to 0-based index
        view = self.views[view_name]

        # Get the appropriate slice based on view name
        if view_name == 'axial':
            slice_array = self.image_array[current_slice, :, :]
            total_slices = self.image_array.shape[0]
        elif view_name == 'coronal':
            slice_array = self.image_array[:, current_slice, :]
            total_slices = self.image_array.shape[1]
        elif view_name == 'sagittal':
            slice_array = self.image_array[:, :, current_slice]
            total_slices = self.image_array.shape[2]
        else:
            return

        # Apply window/level if we have values
        if hasattr(self, 'window') and hasattr(self, 'level'):
            slice_array = self.apply_window_level(slice_array, self.window, self.level)

        # Create QImage from numpy array
        if slice_array.dtype != np.uint8:
            # Normalize to 0-255 range for display
            slice_array = ((slice_array - slice_array.min()) / (slice_array.max() - slice_array.min() + 1e-10) * 255).astype(np.uint8)

        # Create QImage from the array
        height, width = slice_array.shape
        bytes_per_line = width
        qimage = QImage(slice_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # Create a QPixmap from the QImage for display
        pixmap = QPixmap.fromImage(qimage)
        
        # Draw slice number on the pixmap
        painter = QPainter(pixmap)
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(Qt.green))
        painter.drawText(10, 20, f"Slice: {current_slice + 1}/{total_slices}")
        painter.end()
        
        # Set the pixmap for the view
        view.setPixmap(pixmap)
        
        # Update position information
        self.update_position_info()

class EnhancedMainWindow(QMainWindow):
    """Enhanced main window with advanced options for MRI to CT conversion"""
    
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("Synthetic CT Generator - Enhanced")
        self.setMinimumSize(1200, 800)
        
        # Initialize variables
        self.mri_image = None
        self.preprocessed_mri = None
        self.segmented_image = None
        self.synthetic_ct = None
        self.ref_ct_image = None
        
        # Set up logging
        setup_logging()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Set central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(450)
        
        # Add controls to left panel
        self.createControlPanel(left_layout)
        
        # Create right panel (viewers)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create tab widget for main operations
        self.operation_tabs = QTabWidget()
        right_layout.addWidget(self.operation_tabs)
        
        # Add conversion tab
        conversion_tab = QWidget()
        conversion_layout = QVBoxLayout(conversion_tab)
        self.createViewerTabs(conversion_layout)
        self.operation_tabs.addTab(conversion_tab, "Conversion")
        
        # Add training tab
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        self.createTrainingPanel(training_layout)
        self.operation_tabs.addTab(training_tab, "Train Model")
        
        # Add evaluation tab
        evaluation_tab = QWidget()
        evaluation_layout = QVBoxLayout(evaluation_tab)
        self.createEvaluationPanel(evaluation_layout)
        self.operation_tabs.addTab(evaluation_tab, "Evaluation")
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # Add stretch factor
        
    def createControlPanel(self, parent_layout):
        """Create the enhanced control panel with advanced options"""
        
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
        
        # Preprocessing group
        preproc_group = QGroupBox("Preprocessing Options")
        preproc_layout = QVBoxLayout()
        
        # Bias field correction
        bias_group = QGroupBox("Bias Field Correction")
        bias_layout = QFormLayout()
        self.bias_enable = QCheckBox("Enable")
        self.bias_enable.setChecked(True)
        bias_layout.addRow("Enable:", self.bias_enable)
        
        self.bias_shrink = QSpinBox()
        self.bias_shrink.setRange(1, 8)
        self.bias_shrink.setValue(4)
        bias_layout.addRow("Shrink Factor:", self.bias_shrink)
        
        self.bias_iterations = QSpinBox()
        self.bias_iterations.setRange(10, 100)
        self.bias_iterations.setValue(50)
        bias_layout.addRow("Iterations:", self.bias_iterations)
        
        bias_group.setLayout(bias_layout)
        preproc_layout.addWidget(bias_group)
        
        # Denoising
        denoise_group = QGroupBox("Denoising")
        denoise_layout = QFormLayout()
        self.denoise_enable = QCheckBox("Enable")
        self.denoise_enable.setChecked(True)
        denoise_layout.addRow("Enable:", self.denoise_enable)
        
        self.denoise_method = QComboBox()
        self.denoise_method.addItems(["gaussian", "bilateral", "nlm"])
        denoise_layout.addRow("Method:", self.denoise_method)
        
        self.denoise_sigma = QDoubleSpinBox()
        self.denoise_sigma.setRange(0.1, 5.0)
        self.denoise_sigma.setValue(0.5)
        self.denoise_sigma.setSingleStep(0.1)
        denoise_layout.addRow("Sigma:", self.denoise_sigma)
        
        denoise_group.setLayout(denoise_layout)
        preproc_layout.addWidget(denoise_group)
        
        # Normalization
        norm_group = QGroupBox("Normalization")
        norm_layout = QFormLayout()
        self.norm_enable = QCheckBox("Enable")
        self.norm_enable.setChecked(True)
        norm_layout.addRow("Enable:", self.norm_enable)
        
        self.norm_method = QComboBox()
        self.norm_method.addItems(["minmax", "z-score", "histogram"])
        norm_layout.addRow("Method:", self.norm_method)
        
        self.norm_min = QDoubleSpinBox()
        self.norm_min.setRange(-1000, 1000)
        self.norm_min.setValue(0)
        norm_layout.addRow("Min Value:", self.norm_min)
        
        self.norm_max = QDoubleSpinBox()
        self.norm_max.setRange(-1000, 1000)
        self.norm_max.setValue(1000)
        norm_layout.addRow("Max Value:", self.norm_max)
        
        norm_group.setLayout(norm_layout)
        preproc_layout.addWidget(norm_group)
        
        preproc_group.setLayout(preproc_layout)
        parent_layout.addWidget(preproc_group)
        
        # Segmentation group
        seg_group = QGroupBox("Segmentation Options")
        seg_layout = QVBoxLayout()
        
        # Segmentation method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.seg_method = QComboBox()
        self.seg_method.addItems(["auto", "deep_learning", "atlas"])
        method_layout.addWidget(self.seg_method)
        seg_layout.addLayout(method_layout)
        
        # Tissue selection
        tissue_group = QGroupBox("Tissue Classes")
        tissue_layout = QGridLayout()
        
        self.tissue_checks = {}
        tissues = ["background", "air", "soft_tissue", "bone", "fat", "csf"]
        for i, tissue in enumerate(tissues):
            check = QCheckBox(tissue)
            check.setChecked(True)
            self.tissue_checks[tissue] = check
            tissue_layout.addWidget(check, i//2, i%2)
            
        tissue_group.setLayout(tissue_layout)
        seg_layout.addWidget(tissue_group)
        
        seg_group.setLayout(seg_layout)
        parent_layout.addWidget(seg_group)
        
        # Conversion group
        conv_group = QGroupBox("Conversion Options")
        conv_layout = QVBoxLayout()
        
        # Region selection
        region_layout = QHBoxLayout()
        region_layout.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.addItems(["head", "pelvis", "thorax"])
        region_layout.addWidget(self.region_combo)
        conv_layout.addLayout(region_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["atlas", "cnn", "gan"])
        model_layout.addWidget(self.model_combo)
        conv_layout.addLayout(model_layout)
        
        # Add the collapsible advanced options
        self.createAdvancedOptions(conv_layout)
        
        conv_group.setLayout(conv_layout)
        parent_layout.addWidget(conv_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        # Preprocess button
        self.preprocess_btn = QPushButton("Preprocess MRI")
        self.preprocess_btn.setEnabled(False)
        self.preprocess_btn.clicked.connect(self.preprocessMRI)
        actions_layout.addWidget(self.preprocess_btn)
        
        # Segment button
        self.segment_btn = QPushButton("Segment Tissues")
        self.segment_btn.setEnabled(False)
        self.segment_btn.clicked.connect(self.segmentTissues)
        actions_layout.addWidget(self.segment_btn)
        
        # Convert button
        self.convert_btn = QPushButton("Generate Synthetic CT")
        self.convert_btn.setEnabled(False)
        self.convert_btn.clicked.connect(self.generateSyntheticCT)
        actions_layout.addWidget(self.convert_btn)
        
        # Save button
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.saveResults)
        actions_layout.addWidget(self.save_btn)
        
        actions_group.setLayout(actions_layout)
        parent_layout.addWidget(actions_group)
        
        # Add spacer to push everything up
        parent_layout.addStretch()
        
    def createViewerTabs(self, parent_layout):
        """Create tabs for image viewers"""
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        parent_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.mri_viewer = SimpleImageViewer()
        self.preproc_viewer = SimpleImageViewer()
        self.seg_viewer = SimpleImageViewer()
        self.ct_viewer = SimpleImageViewer()
        
        # Add tooltip thông báo cách sử dụng
        viewer_tooltip = "Scroll chuột trên view để thay đổi lát cắt\nClick vào view để chọn view đó làm view hoạt động\nSử dụng combobox để chuyển đổi giữa các view"
        self.mri_viewer.setToolTip(viewer_tooltip)
        self.preproc_viewer.setToolTip(viewer_tooltip)
        self.seg_viewer.setToolTip(viewer_tooltip)
        self.ct_viewer.setToolTip(viewer_tooltip)
        
        # Add tabs to widget
        self.tab_widget.addTab(self.mri_viewer, "MRI")
        self.tab_widget.addTab(self.preproc_viewer, "Preprocessed")
        self.tab_widget.addTab(self.seg_viewer, "Segmentation")
        self.tab_widget.addTab(self.ct_viewer, "Synthetic CT")
        
    def selectMRI(self):
        """Select MRI image file or DICOM series directory"""
        # Ask user if they want to load a single file or a DICOM series
        options = ["Single File", "DICOM Series (Directory)"]
        selected_option, ok = QInputDialog.getItem(
            self, "Select MRI Source", "Choose input type:", options, 0, False
        )
        
        if not ok:
            return
        
        if selected_option == "Single File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select MRI Image",
                "",
                "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*.*)"
            )
            
            if not file_path:
                return
        else:
            # Select directory for DICOM series
            file_path = QFileDialog.getExistingDirectory(
                self,
                "Select DICOM Series Directory for MRI",
                ""
            )
            
            if not file_path:
                return
        
        self.mri_path_label.setText(file_path)
        logging.info(f"Attempting to load MRI from: {file_path}")
        
        # Hiển thị dialog tiến trình
        progress_dialog = QProgressDialog("Loading MRI image, please wait...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Loading MRI")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(200)  # Hiển thị ngay nếu việc tải kéo dài hơn 200ms
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Hiển thị thông báo chờ đợi nếu tải file lớn
            self.statusBar().showMessage("Loading MRI image, please wait...")
            QApplication.processEvents()  # Đảm bảo UI được cập nhật
            
            # Kiểm tra xem path có phải là thư mục không
            if os.path.isdir(file_path):
                logging.info(f"Loading DICOM series from directory: {file_path}")
                self.statusBar().showMessage("Scanning DICOM series, this may take a while...")
                QApplication.processEvents()
                
                # Đếm số file trong thư mục trước khi tải
                file_count = 0
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.lower().endswith('.dcm') or not os.path.splitext(file)[1]:
                            file_count += 1
                
                if file_count > 0:
                    self.statusBar().showMessage(f"Found {file_count} potential DICOM files, loading...")
                    QApplication.processEvents()
            
            self.mri_image = load_medical_image(file_path)
            progress_dialog.close()
            
            if self.mri_image is None:
                raise ValueError("Failed to load MRI image: Image is None")
            
            # Hiển thị thông tin về kích thước ảnh đã tải
            try:
                image_size = self.mri_image.GetSize()
                spacing = self.mri_image.GetSpacing()
                dimensions = f"{image_size[0]}x{image_size[1]}x{image_size[2]}"
                self.statusBar().showMessage(f"MRI loaded: {dimensions} voxels, spacing: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}mm")
                logging.info(f"Loaded MRI with dimensions: {dimensions}")
            except:
                pass
                
            self.mri_viewer.set_image(self.mri_image)
            self.preprocess_btn.setEnabled(True)
            logging.info("MRI loaded successfully")
            self.tab_widget.setCurrentIndex(0)  # Switch to MRI tab
        except Exception as e:
            progress_dialog.close()
            logging.error(f"Error loading MRI: {str(e)}")
            # Hiển thị thông báo lỗi chi tiết
            error_msg = f"Failed to load MRI: {str(e)}\n\nPlease check file path and format."
            QMessageBox.critical(self, "Error Loading MRI", error_msg)
            self.statusBar().showMessage("Failed to load MRI")
            
    def selectReferenceCT(self):
        """Select reference CT image file or DICOM series directory"""
        # Ask user if they want to load a single file or a DICOM series
        options = ["Single File", "DICOM Series (Directory)"]
        selected_option, ok = QInputDialog.getItem(
            self, "Select CT Source", "Choose input type:", options, 0, False
        )
        
        if not ok:
            return
        
        if selected_option == "Single File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Reference CT",
                "",
                "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*.*)"
            )
            
            if not file_path:
                return
        else:
            # Select directory for DICOM series
            file_path = QFileDialog.getExistingDirectory(
                self,
                "Select DICOM Series Directory for CT",
                ""
            )
            
            if not file_path:
                return
        
        self.ref_ct_path_label.setText(file_path)
        logging.info(f"Attempting to load reference CT from: {file_path}")
        
        # Hiển thị dialog tiến trình
        progress_dialog = QProgressDialog("Loading reference CT, please wait...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Loading CT")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(200)  # Hiển thị ngay nếu việc tải kéo dài hơn 200ms
        progress_dialog.setValue(0)
        progress_dialog.show()
        QApplication.processEvents()
        
        try:
            # Hiển thị thông báo chờ đợi nếu tải file lớn
            self.statusBar().showMessage("Loading reference CT image, please wait...")
            QApplication.processEvents()  # Đảm bảo UI được cập nhật
            
            # Kiểm tra xem path có phải là thư mục không
            if os.path.isdir(file_path):
                logging.info(f"Loading DICOM series from directory: {file_path}")
                self.statusBar().showMessage("Scanning DICOM series, this may take a while...")
                QApplication.processEvents()
                
                # Đếm số file trong thư mục trước khi tải
                file_count = 0
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.lower().endswith('.dcm') or not os.path.splitext(file)[1]:
                            file_count += 1
                
                if file_count > 0:
                    self.statusBar().showMessage(f"Found {file_count} potential DICOM files, loading...")
                    QApplication.processEvents()
            
            self.ref_ct_image = load_medical_image(file_path)
            progress_dialog.close()
            
            if self.ref_ct_image is None:
                raise ValueError("Failed to load reference CT: Image is None")
            
            # Hiển thị thông tin về kích thước ảnh đã tải
            try:
                image_size = self.ref_ct_image.GetSize()
                spacing = self.ref_ct_image.GetSpacing()
                dimensions = f"{image_size[0]}x{image_size[1]}x{image_size[2]}"
                self.statusBar().showMessage(f"CT loaded: {dimensions} voxels, spacing: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f}mm")
                logging.info(f"Loaded CT with dimensions: {dimensions}")
            except:
                pass
            
            # If we have a reference CT, add it to the viewers
            ref_ct_tab_index = -1
            for i in range(self.tab_widget.count()):
                if self.tab_widget.tabText(i) == "Reference CT":
                    ref_ct_tab_index = i
                    break
                    
            if ref_ct_tab_index == -1:
                # Thêm tab mới với SimpleImageViewer
                ref_ct_viewer = SimpleImageViewer()
                # Thêm tooltip hướng dẫn
                ref_ct_viewer.setToolTip("Scroll chuột trên view để thay đổi lát cắt\nClick vào view để chọn view đó làm view hoạt động\nSử dụng combobox để chuyển đổi giữa các view")
                self.tab_widget.addTab(ref_ct_viewer, "Reference CT")
                ref_ct_tab_index = self.tab_widget.count() - 1
            
            # Cập nhật viewer
            self.tab_widget.widget(ref_ct_tab_index).set_image(self.ref_ct_image)
            logging.info("Reference CT loaded successfully")
            self.tab_widget.setCurrentIndex(ref_ct_tab_index)  # Switch to Reference CT tab
        except Exception as e:
            progress_dialog.close()
            logging.error(f"Error loading reference CT: {str(e)}")
            # Hiển thị thông báo lỗi chi tiết
            error_msg = f"Failed to load reference CT: {str(e)}\n\nPlease check file path and format."
            QMessageBox.critical(self, "Error Loading Reference CT", error_msg)
            self.statusBar().showMessage("Failed to load reference CT")
            
    def selectOutputDir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if dir_path:
            self.output_dir_label.setText(dir_path)
            
    def preprocessMRI(self):
        """Preprocess MRI image"""
        if self.mri_image is None:
            return
            
        # Get preprocessing parameters
        preproc_params = {
            'bias_field': {
                'enable': self.bias_enable.isChecked(),
                'shrink_factor': self.bias_shrink.value(),
                'iterations': self.bias_iterations.value()
            },
            'denoising': {
                'enable': self.denoise_enable.isChecked(),
                'method': self.denoise_method.currentText(),
                'sigma': self.denoise_sigma.value()
            },
            'normalization': {
                'enable': self.norm_enable.isChecked(),
                'method': self.norm_method.currentText(),
                'min': self.norm_min.value(),
                'max': self.norm_max.value()
            }
        }
        
        # Create processing thread
        self.processing_thread = ProcessingThread(
            preprocess_mri,
            self.mri_image,
            preproc_params
        )
        
        # Connect signals
        self.processing_thread.finished.connect(self.preprocessingFinished)
        self.processing_thread.error.connect(self.processingError)
        self.processing_thread.progress.connect(self.updateProgress)
        
        # Start processing
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.processing_thread.start()
        
    def preprocessingFinished(self, result):
        """Handle preprocessing completion"""
        self.preprocessed_mri = result
        self.preproc_viewer.set_image(result)
        self.segment_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def segmentTissues(self):
        """Segment tissues in preprocessed MRI"""
        if self.preprocessed_mri is None:
            return
            
        # Get segmentation parameters
        seg_params = {
            'method': self.seg_method.currentText(),
            'tissues': [t for t, check in self.tissue_checks.items() if check.isChecked()]
        }
        
        # Create processing thread
        self.processing_thread = ProcessingThread(
            segment_tissues,
            self.preprocessed_mri,
            **seg_params
        )
        
        # Connect signals
        self.processing_thread.finished.connect(self.segmentationFinished)
        self.processing_thread.error.connect(self.processingError)
        self.processing_thread.progress.connect(self.updateProgress)
        
        # Start processing
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.processing_thread.start()
        
    def segmentationFinished(self, result):
        """Handle segmentation completion"""
        self.segmented_image = result
        self.seg_viewer.set_image(result)
        self.convert_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def generateSyntheticCT(self):
        """Generate synthetic CT from MRI using the selected model"""
        if self.mri_path is None:
            QMessageBox.warning(self, "Error", "Please select an MRI image first")
            return
            
        # Get selected options
        model_type = self.model_combo.currentText()
        region = self.region_combo.currentText()
        
        # Get preprocessing options
        apply_bias = self.bias_field_check.isChecked()
        apply_denoise = self.denoise_check.isChecked()
        apply_normalize = self.normalize_check.isChecked()
        
        # Check if we're using pipeline
        use_pipeline = self.pipeline_check.isChecked() 
        
        # Create output directory if it doesn't exist
        if not self.output_dir:
            # Default output directory alongside the MRI file
            self.output_dir = str(Path(self.mri_path).parent / "synthetic_ct_results")
            os.makedirs(self.output_dir, exist_ok=True)
            self.output_dir_label.setText(self.output_dir)
            
        # Show progress
        self.statusBar().showMessage("Generating synthetic CT...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Update UI
        self.tabWidget.setCurrentWidget(self.conversion_tab)
        
        # Use integrated pipeline for end-to-end processing
        if use_pipeline:
            # Import the complete pipeline function
            try:
                from app.core.conversion.mri_to_ct_pipeline import run_complete_pipeline_with_evaluation
                
                # Create and start processing thread
                self.processing_thread = ProcessingThread(
                    run_complete_pipeline_with_evaluation,
                    mri_path=self.mri_path,
                    output_dir=self.output_dir,
                    reference_ct_path=self.reference_ct_path if self.evaluate_check.isChecked() else None,
                    model_type=model_type,
                    region=region,
                    apply_bias_field_correction=apply_bias,
                    apply_denoising=apply_denoise,
                    apply_normalization=apply_normalize,
                    progress_callback=lambda val, msg: self.updateProgress(val) 
                )
                
                self.processing_thread.finished.connect(self.pipelineFinished)
                self.processing_thread.error.connect(self.processingError)
                self.processing_thread.progress.connect(self.updateProgress)
                self.processing_thread.start()
                
            except ImportError as e:
                QMessageBox.critical(self, "Error", f"Could not import pipeline module: {str(e)}")
                self.progress_bar.setVisible(False)
                self.statusBar().showMessage("Error: Pipeline module not available")
                return
                
        else:
            # Individual steps approach (same as before)
            # First preprocess
            if not hasattr(self, 'preprocessed_mri') or self.preprocessed_mri is None:
                self.preprocessMRI()
            else:
                # Skip to segmentation
                self.segmentTissues()
        
    def pipelineFinished(self, result):
        """Handle completion of the integrated pipeline"""
        # Update UI
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Synthetic CT generation completed successfully")
        
        # Check for errors
        if isinstance(result, dict) and 'error' in result:
            QMessageBox.warning(self, "Pipeline Error", 
                               f"Pipeline completed with errors: {result['error']}")
            return
            
        # Store results
        if 'preprocessed_mri' in result:
            self.preprocessed_mri = result['preprocessed_mri']
            
        if 'segmentations' in result:
            self.segmentations = result['segmentations']
            
        if 'synthetic_ct' in result:
            self.synthetic_ct = result['synthetic_ct']
            
        # Update viewers
        if hasattr(self, 'preprocessed_mri_viewer') and self.preprocessed_mri:
            self.preprocessed_mri_viewer.set_image(self.preprocessed_mri)
            
        if hasattr(self, 'segmentation_viewer') and self.segmentations:
            # Use bone segmentation for display if available
            if 'bone' in self.segmentations:
                self.segmentation_viewer.set_image(self.segmentations['bone'])
                
        if hasattr(self, 'synthetic_ct_viewer') and self.synthetic_ct:
            self.synthetic_ct_viewer.set_image(self.synthetic_ct)
            
        # Update comparison view if reference CT is available
        if hasattr(self, 'comparison_viewer') and self.reference_ct and self.synthetic_ct:
            self.comparison_viewer.set_images(self.reference_ct, self.synthetic_ct)
            
        # Display evaluation results if available
        if 'evaluation_results' in result:
            eval_results = result['evaluation_results']
            self.displayEvaluationResults(eval_results)
            
        # Enable save button
        self.save_outputs_btn.setEnabled(True)
        
        # Show success message
        QMessageBox.information(self, "Success", "Synthetic CT generation completed successfully!")
        
    def displayEvaluationResults(self, eval_results):
        """Display evaluation results in the GUI"""
        if not hasattr(self, 'evaluation_tab'):
            # Create evaluation tab if it doesn't exist
            self.evaluation_tab = QWidget()
            eval_layout = QVBoxLayout(self.evaluation_tab)
            
            # Create results text area
            self.eval_results_text = QTextEdit()
            self.eval_results_text.setReadOnly(True)
            eval_layout.addWidget(self.eval_results_text)
            
            # Add to tab widget
            self.tabWidget.addTab(self.evaluation_tab, "Evaluation")
            
        # Display results
        results_text = "## Evaluation Results\n\n"
        
        if isinstance(eval_results, dict):
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    results_text += f"**{metric}**: {value:.4f}\n"
                else:
                    results_text += f"**{metric}**: {value}\n"
        else:
            # If it's an EvaluationResult object
            if hasattr(eval_results, 'metrics'):
                for metric, value in eval_results.metrics.items():
                    results_text += f"**{metric}**: {value:.4f}\n"
            
        # Add path to report if available
        if hasattr(eval_results, 'report_path') and eval_results.report_path:
            results_text += f"\n**Detailed Report**: {eval_results.report_path}\n"
            
        # Add paths to images if available
        if hasattr(eval_results, 'image_paths') and eval_results.image_paths:
            results_text += "\n**Visualizations**:\n"
            for image_path in eval_results.image_paths:
                results_text += f"- {image_path}\n"
                
        # Set text
        self.eval_results_text.setMarkdown(results_text)
        
        # Switch to evaluation tab
        self.tabWidget.setCurrentWidget(self.evaluation_tab)
        
    def saveResults(self):
        """Save all results"""
        if not self.output_dir_label.text() or self.output_dir_label.text() == "Default":
            QMessageBox.warning(self, "Warning", "Please select an output directory first")
            return
            
        output_dir = self.output_dir_label.text()
        os.makedirs(output_dir, exist_ok=True)
        
        # Get output format and compression settings
        output_format = self.output_format.currentText().lower() if hasattr(self, 'output_format') else 'nifti'
        use_compression = self.use_compression.isChecked() if hasattr(self, 'use_compression') else True
        save_intermediates = self.save_intermediates.isChecked() if hasattr(self, 'save_intermediates') else False
        
        # Determine file extension
        extension = '.nii.gz' if output_format == 'nifti' and use_compression else '.nii'
        if output_format == 'dicom':
            extension = '.dcm'
        
        # Save synthetic CT (always save this)
        if self.synthetic_ct is not None:
            ct_path = os.path.join(output_dir, f"synthetic_ct{extension}")
            try:
                sitk.WriteImage(self.synthetic_ct, ct_path)
                logging.info(f"Saved synthetic CT to {ct_path}")
            except Exception as e:
                logging.error(f"Error saving synthetic CT: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error saving synthetic CT: {str(e)}")
        
        # Save intermediate results if requested
        if save_intermediates:
            # Save preprocessed MRI
            if self.preprocessed_mri is not None:
                try:
                    preproc_path = os.path.join(output_dir, f"preprocessed_mri{extension}")
                    sitk.WriteImage(self.preprocessed_mri, preproc_path)
                    logging.info(f"Saved preprocessed MRI to {preproc_path}")
                except Exception as e:
                    logging.error(f"Error saving preprocessed MRI: {str(e)}")
            
            # Save segmentation
            if self.segmented_image is not None:
                try:
                    seg_path = os.path.join(output_dir, f"segmentation{extension}")
                    sitk.WriteImage(self.segmented_image, seg_path)
                    logging.info(f"Saved segmentation to {seg_path}")
                except Exception as e:
                    logging.error(f"Error saving segmentation: {str(e)}")
        
        # Show success message
        QMessageBox.information(
            self, 
            "Save Complete", 
            f"Results saved to {output_dir}\n"
            f"Format: {output_format.upper()}" +
            (f", Compressed" if use_compression and output_format == 'nifti' else "") +
            (f"\nIntermediate results were also saved." if save_intermediates else "")
        )
        
    def processingError(self, error_msg):
        """Handle processing error"""
        QMessageBox.critical(self, "Error", f"Processing error: {error_msg}")
        self.progress_bar.setVisible(False)
        
    def updateProgress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def createTrainingPanel(self, parent_layout):
        """Create training panel for model training"""
        # Data paths section
        data_group = QGroupBox("Training Data")
        data_layout = QFormLayout()
        
        # MRI Directory
        mri_dir_layout = QHBoxLayout()
        self.mri_dir_label = QLabel("Select Directory...")
        mri_dir_button = QPushButton("Browse...")
        mri_dir_button.clicked.connect(self.selectMRITrainingDir)
        mri_dir_layout.addWidget(self.mri_dir_label, 1)
        mri_dir_layout.addWidget(mri_dir_button)
        data_layout.addRow("MRI Directory:", mri_dir_layout)
        
        # CT Directory
        ct_dir_layout = QHBoxLayout()
        self.ct_dir_label = QLabel("Select Directory...")
        ct_dir_button = QPushButton("Browse...")
        ct_dir_button.clicked.connect(self.selectCTTrainingDir)
        ct_dir_layout.addWidget(self.ct_dir_label, 1)
        ct_dir_layout.addWidget(ct_dir_button)
        data_layout.addRow("CT Directory:", ct_dir_layout)
        
        # Output Directory
        output_dir_layout = QHBoxLayout()
        self.model_output_dir_label = QLabel("Select Directory...")
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(self.selectModelOutputDir)
        output_dir_layout.addWidget(self.model_output_dir_label, 1)
        output_dir_layout.addWidget(output_dir_button)
        data_layout.addRow("Output Directory:", output_dir_layout)
        
        data_group.setLayout(data_layout)
        parent_layout.addWidget(data_group)
        
        # Training parameters
        train_params_group = QGroupBox("Training Parameters")
        train_params_layout = QFormLayout()
        
        # Model type
        self.train_model_type = QComboBox()
        self.train_model_type.addItems(["cnn", "gan"])
        train_params_layout.addRow("Model Type:", self.train_model_type)
        
        # Region
        self.train_region = QComboBox()
        self.train_region.addItems(["head", "head_neck", "pelvis", "abdomen", "thorax"])
        train_params_layout.addRow("Anatomical Region:", self.train_region)
        
        # Batch size
        self.train_batch_size = QSpinBox()
        self.train_batch_size.setRange(1, 32)
        self.train_batch_size.setValue(4)
        train_params_layout.addRow("Batch Size:", self.train_batch_size)
        
        # Epochs
        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 1000)
        self.train_epochs.setValue(100)
        train_params_layout.addRow("Epochs:", self.train_epochs)
        
        # Learning Rate
        self.train_lr = QDoubleSpinBox()
        self.train_lr.setRange(0.0001, 0.1)
        self.train_lr.setValue(0.001)
        self.train_lr.setSingleStep(0.0001)
        self.train_lr.setDecimals(6)
        train_params_layout.addRow("Learning Rate:", self.train_lr)
        
        # Use GPU checkbox
        self.train_use_gpu = QCheckBox("Use GPU if available")
        self.train_use_gpu.setChecked(True)
        train_params_layout.addRow("", self.train_use_gpu)
        
        # Model options specific to GAN
        self.gan_options_group = QGroupBox("GAN Options")
        gan_options_layout = QFormLayout()
        
        self.gan_lambda_l1 = QSpinBox()
        self.gan_lambda_l1.setRange(10, 1000)
        self.gan_lambda_l1.setValue(100)
        gan_options_layout.addRow("Lambda L1:", self.gan_lambda_l1)
        
        self.gan_options_group.setLayout(gan_options_layout)
        self.gan_options_group.setVisible(self.train_model_type.currentText() == "gan")
        
        # Connect model type change to show/hide GAN options
        self.train_model_type.currentTextChanged.connect(
            lambda text: self.gan_options_group.setVisible(text == "gan")
        )
        
        train_params_group.setLayout(train_params_layout)
        parent_layout.addWidget(train_params_group)
        parent_layout.addWidget(self.gan_options_group)
        
        # Training actions
        train_actions_group = QGroupBox("Actions")
        train_actions_layout = QVBoxLayout()
        
        # Train model button
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.trainModel)
        train_actions_layout.addWidget(self.train_model_btn)
        
        # Stop training button
        self.stop_training_btn = QPushButton("Stop Training")
        self.stop_training_btn.clicked.connect(self.stopTraining)
        self.stop_training_btn.setEnabled(False)
        train_actions_layout.addWidget(self.stop_training_btn)
        
        train_actions_group.setLayout(train_actions_layout)
        parent_layout.addWidget(train_actions_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        log_layout.addWidget(self.training_log)
        
        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)
        
        # Add stretch to push everything up
        parent_layout.addStretch()

    def createEvaluationPanel(self, parent_layout):
        """Create evaluation panel for synthetic CT evaluation"""
        # Input files section
        input_group = QGroupBox("Evaluation Input")
        input_layout = QFormLayout()
        
        # Synthetic CT
        synth_ct_layout = QHBoxLayout()
        self.eval_synth_ct_label = QLabel("Select File or Directory...")
        synth_ct_button = QPushButton("Browse...")
        synth_ct_button.clicked.connect(self.selectSynthCTForEval)
        synth_ct_layout.addWidget(self.eval_synth_ct_label, 1)
        synth_ct_layout.addWidget(synth_ct_button)
        input_layout.addRow("Synthetic CT:", synth_ct_layout)
        
        # Reference CT
        ref_ct_layout = QHBoxLayout()
        self.eval_ref_ct_label = QLabel("Select File or Directory...")
        ref_ct_button = QPushButton("Browse...")
        ref_ct_button.clicked.connect(self.selectRefCTForEval)
        ref_ct_layout.addWidget(self.eval_ref_ct_label, 1)
        ref_ct_layout.addWidget(ref_ct_button)
        input_layout.addRow("Reference CT:", ref_ct_layout)
        
        # Output Directory
        eval_output_layout = QHBoxLayout()
        self.eval_output_dir_label = QLabel("Select Directory...")
        eval_output_button = QPushButton("Browse...")
        eval_output_button.clicked.connect(self.selectEvalOutputDir)
        eval_output_layout.addWidget(self.eval_output_dir_label, 1)
        eval_output_layout.addWidget(eval_output_button)
        input_layout.addRow("Output Directory:", eval_output_layout)
        
        input_group.setLayout(input_layout)
        parent_layout.addWidget(input_group)
        
        # Evaluation parameters
        eval_params_group = QGroupBox("Evaluation Parameters")
        eval_params_layout = QFormLayout()
        
        # Metrics
        metrics_layout = QVBoxLayout()
        
        self.metric_mae = QCheckBox("Mean Absolute Error (MAE)")
        self.metric_mae.setChecked(True)
        metrics_layout.addWidget(self.metric_mae)
        
        self.metric_mse = QCheckBox("Mean Squared Error (MSE)")
        self.metric_mse.setChecked(True)
        metrics_layout.addWidget(self.metric_mse)
        
        self.metric_psnr = QCheckBox("Peak Signal-to-Noise Ratio (PSNR)")
        self.metric_psnr.setChecked(True)
        metrics_layout.addWidget(self.metric_psnr)
        
        self.metric_ssim = QCheckBox("Structural Similarity Index (SSIM)")
        self.metric_ssim.setChecked(True)
        metrics_layout.addWidget(self.metric_ssim)
        
        eval_params_layout.addRow("Metrics:", metrics_layout)
        
        # Region of analysis
        regions_layout = QVBoxLayout()
        
        self.region_all = QCheckBox("All")
        self.region_all.setChecked(True)
        regions_layout.addWidget(self.region_all)
        
        self.region_bone = QCheckBox("Bone")
        self.region_bone.setChecked(True)
        regions_layout.addWidget(self.region_bone)
        
        self.region_soft_tissue = QCheckBox("Soft Tissue")
        self.region_soft_tissue.setChecked(True)
        regions_layout.addWidget(self.region_soft_tissue)
        
        self.region_air = QCheckBox("Air")
        self.region_air.setChecked(True)
        regions_layout.addWidget(self.region_air)
        
        eval_params_layout.addRow("Regions:", regions_layout)
        
        # Generate report checkbox
        self.generate_report = QCheckBox("Generate PDF Report")
        self.generate_report.setChecked(True)
        eval_params_layout.addRow("", self.generate_report)
        
        eval_params_group.setLayout(eval_params_layout)
        parent_layout.addWidget(eval_params_group)
        
        # Evaluation actions
        eval_actions_group = QGroupBox("Actions")
        eval_actions_layout = QVBoxLayout()
        
        # Evaluate button
        self.evaluate_btn = QPushButton("Evaluate Synthetic CT")
        self.evaluate_btn.clicked.connect(self.evaluateSyntheticCT)
        eval_actions_layout.addWidget(self.evaluate_btn)
        
        eval_actions_group.setLayout(eval_actions_layout)
        parent_layout.addWidget(eval_actions_group)
        
        # Results viewer
        results_group = QGroupBox("Evaluation Results")
        results_layout = QVBoxLayout()
        
        self.results_tabs = QTabWidget()
        
        # Metrics tab
        metrics_tab = QWidget()
        metrics_tab_layout = QVBoxLayout(metrics_tab)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        metrics_tab_layout.addWidget(self.metrics_text)
        self.results_tabs.addTab(metrics_tab, "Metrics")
        
        # Difference view tab
        diff_tab = QWidget()
        diff_tab_layout = QVBoxLayout(diff_tab)
        self.diff_viewer = SimpleImageViewer()
        diff_tab_layout.addWidget(self.diff_viewer)
        self.results_tabs.addTab(diff_tab, "Difference View")
        
        results_layout.addWidget(self.results_tabs)
        results_group.setLayout(results_layout)
        parent_layout.addWidget(results_group)
        
        # Add stretch to push everything up
        parent_layout.addStretch()

    def selectMRITrainingDir(self):
        """Select directory containing MRI training data"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select MRI Training Data Directory",
            ""
        )
        
        if dir_path:
            self.mri_dir_label.setText(dir_path)
        
    def selectCTTrainingDir(self):
        """Select directory containing CT training data"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select CT Training Data Directory",
            ""
        )
        
        if dir_path:
            self.ct_dir_label.setText(dir_path)
        
    def selectModelOutputDir(self):
        """Select output directory for model"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Model Output Directory",
            ""
        )
        
        if dir_path:
            self.model_output_dir_label.setText(dir_path)
        
    def trainModel(self):
        """Train a model using the selected parameters"""
        # Verify input
        mri_dir = self.mri_dir_label.text()
        ct_dir = self.ct_dir_label.text()
        output_dir = self.model_output_dir_label.text()
        
        if mri_dir == "Select Directory..." or ct_dir == "Select Directory..." or output_dir == "Select Directory...":
            QMessageBox.warning(self, "Warning", "Please select all directories (MRI, CT, Output)")
            return
        
        # Get training parameters
        params = {
            'model_type': self.train_model_type.currentText(),
            'region': self.train_region.currentText(),
            'batch_size': self.train_batch_size.value(),
            'epochs': self.train_epochs.value(),
            'learning_rate': self.train_lr.value(),
            'use_gpu': self.train_use_gpu.isChecked()
        }
        
        # Add GAN-specific parameters if needed
        if params['model_type'] == 'gan':
            params['lambda_l1'] = self.gan_lambda_l1.value()
        
        # Log start of training
        self.training_log.clear()
        self.training_log.append(f"=== Starting {params['model_type'].upper()} Training ===")
        self.training_log.append(f"MRI Directory: {mri_dir}")
        self.training_log.append(f"CT Directory: {ct_dir}")
        self.training_log.append(f"Output Directory: {output_dir}")
        self.training_log.append(f"Region: {params['region']}")
        self.training_log.append(f"Batch Size: {params['batch_size']}")
        self.training_log.append(f"Epochs: {params['epochs']}")
        self.training_log.append(f"Learning Rate: {params['learning_rate']}")
        self.training_log.append(f"Use GPU: {params['use_gpu']}")
        if params['model_type'] == 'gan':
            self.training_log.append(f"Lambda L1: {params['lambda_l1']}")
        self.training_log.append("-------------------------------")
        
        # Create and configure the training thread based on model type
        if params['model_type'] == 'cnn':
            # Import here to avoid circular imports
            from app.training.train_cnn import train_cnn_model
            
            # Thêm QProgressDialog để hiển thị chi tiết quá trình đào tạo
            self.progress_dialog = QProgressDialog("Training CNN model...", "Cancel", 0, params['epochs'], self)
            self.progress_dialog.setWindowTitle("Training Progress")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.setAutoReset(False)
            self.progress_dialog.setAutoClose(False)
            
            self.processing_thread = ProcessingThread(
                train_cnn_model,
                mri_dir=mri_dir,
                ct_dir=ct_dir,
                output_dir=output_dir,
                region=params['region'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                learning_rate=params['learning_rate'],
                use_gpu=params['use_gpu']
            )
        else:  # GAN
            # Import here to avoid circular imports
            from app.training.train_gan import train_gan_model
            
            # Thêm QProgressDialog để hiển thị chi tiết quá trình đào tạo
            self.progress_dialog = QProgressDialog("Training GAN model...", "Cancel", 0, params['epochs'], self)
            self.progress_dialog.setWindowTitle("Training Progress")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.setAutoReset(False)
            self.progress_dialog.setAutoClose(False)
            
            self.processing_thread = ProcessingThread(
                train_gan_model,
                mri_dir=mri_dir,
                ct_dir=ct_dir,
                output_dir=output_dir,
                region=params['region'],
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                learning_rate=params['learning_rate'],
                use_gpu=params['use_gpu'],
                lambda_l1=params['lambda_l1']
            )
        
        # Connect signals for training updates
        self.processing_thread.progress.connect(self.updateTrainingProgress)
        self.processing_thread.error.connect(self.trainingError)
        self.processing_thread.finished.connect(self.trainingFinished)
        self.processing_thread.status.connect(self.updateTrainingStatus)
        
        # Connect progress dialog canceled signal to stop training
        self.progress_dialog.canceled.connect(self.stopTraining)
        
        # Update UI
        self.train_model_btn.setEnabled(False)
        self.stop_training_btn.setEnabled(True)
        
        # Start training
        self.processing_thread.start()
        
    def updateTrainingProgress(self, progress):
        """Update progress for training"""
        # Update both normal progress bar and progress dialog
        self.progress_bar.setValue(progress)
        
        if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
            self.progress_dialog.setValue(progress)
        
        # Log progress to training log
        if progress % 5 == 0:  # Log every 5 epochs to avoid too much output
            self.training_log.append(f"Epoch {progress}/{self.progress_dialog.maximum()}")
            # Ensure view scrolls to bottom to show latest log
            self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
        
    def trainingError(self, error_msg):
        """Handle training errors"""
        self.training_log.append(f"ERROR: {error_msg}")
        self.train_model_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Training Error", f"Error during training: {error_msg}")
        
    def trainingFinished(self, result):
        """Handle completion of model training"""
        # Update UI
        self.train_model_btn.setEnabled(True)
        self.stop_training_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Close progress dialog if it exists
        if hasattr(self, 'progress_dialog') and self.progress_dialog.isVisible():
            self.progress_dialog.close()
        
        # Log completion
        self.training_log.append("=== Training Complete ===")
        self.training_log.append(f"Model saved to: {result.get('model_path', result.get('generator_path', 'Unknown'))}")
        
        # Log different metrics based on model type
        if 'metrics' in result:
            self.training_log.append("Final metrics:")
            for metric, value in result['metrics'].items():
                self.training_log.append(f"  {metric}: {value:.4f}")
        
        # Include specific metrics for GAN if available
        if 'val_d_loss' in result:
            self.training_log.append(f"  Discriminator Loss: {result['val_d_loss']:.4f}")
            self.training_log.append(f"  Generator Loss: {result['val_g_loss']:.4f}")
        
        if 'training_time' in result:
            self.training_log.append(f"Training time: {result['training_time']:.2f} seconds")
        
        self.training_log.append("-------------------------------")
        
        # Scroll to bottom
        self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())
        
        # Show message
        model_type = "GAN" if 'generator_path' in result else "CNN"
        model_path = result.get('model_path', result.get('generator_path', 'Unknown'))
        QMessageBox.information(self, "Training Complete", 
                               f"{model_type} model training completed successfully.\nModel saved to: {model_path}")
        
    def stopTraining(self):
        """Stop the current training process"""
        if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
            # Set a flag to stop on next epoch
            # This assumes the training function checks for a stop flag
            # You'll need to implement this in your training code
            self.processing_thread.requestInterruption()
            self.training_log.append("Stopping training after current epoch...")
        
    def selectSynthCTForEval(self):
        """Select synthetic CT for evaluation"""
        # Ask user if they want to select a file or a directory
        options = ["Single File", "DICOM Series (Directory)"]
        selected_option, ok = QInputDialog.getItem(
            self, "Select Synthetic CT Source", "Choose input type:", options, 0, False
        )
        
        if not ok:
            return
        
        if selected_option == "Single File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Synthetic CT",
                "",
                "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*.*)"
            )
            
            if not file_path:
                return
        else:
            # Select directory for DICOM series
            file_path = QFileDialog.getExistingDirectory(
                self,
                "Select DICOM Series Directory for Synthetic CT",
                ""
            )
            
            if not file_path:
                return
            
        self.eval_synth_ct_label.setText(file_path)
        
    def selectRefCTForEval(self):
        """Select reference CT for evaluation"""
        # Ask user if they want to select a file or a directory
        options = ["Single File", "DICOM Series (Directory)"]
        selected_option, ok = QInputDialog.getItem(
            self, "Select Reference CT Source", "Choose input type:", options, 0, False
        )
        
        if not ok:
            return
        
        if selected_option == "Single File":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Reference CT",
                "",
                "Medical Images (*.nii *.nii.gz *.dcm);;All Files (*.*)"
            )
            
            if not file_path:
                return
        else:
            # Select directory for DICOM series
            file_path = QFileDialog.getExistingDirectory(
                self,
                "Select DICOM Series Directory for Reference CT",
                ""
            )
            
            if not file_path:
                return
            
        self.eval_ref_ct_label.setText(file_path)
        
    def selectEvalOutputDir(self):
        """Select evaluation output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Evaluation Output Directory",
            ""
        )
        
        if dir_path:
            self.eval_output_dir_label.setText(dir_path)
        
    def evaluateSyntheticCT(self):
        """Evaluate synthetic CT against reference CT"""
        # Verify input
        synth_ct_path = self.eval_synth_ct_label.text()
        ref_ct_path = self.eval_ref_ct_label.text()
        output_dir = self.eval_output_dir_label.text()
        
        if (synth_ct_path == "Select File or Directory..." or 
            ref_ct_path == "Select File or Directory..." or 
            output_dir == "Select Directory..."):
            QMessageBox.warning(self, "Warning", "Please select synthetic CT, reference CT, and output directory")
            return
        
        # Get metrics to evaluate
        metrics = []
        if self.metric_mae.isChecked():
            metrics.append('mae')
        if self.metric_mse.isChecked():
            metrics.append('mse')
        if self.metric_psnr.isChecked():
            metrics.append('psnr')
        if self.metric_ssim.isChecked():
            metrics.append('ssim')
        
        if not metrics:
            QMessageBox.warning(self, "Warning", "Please select at least one metric")
            return
        
        # Get regions to evaluate
        regions = []
        if self.region_all.isChecked():
            regions.append('all')
        if self.region_bone.isChecked():
            regions.append('bone')
        if self.region_soft_tissue.isChecked():
            regions.append('soft_tissue')
        if self.region_air.isChecked():
            regions.append('air')
        
        if not regions:
            QMessageBox.warning(self, "Warning", "Please select at least one region")
            return
        
        try:
            # Load images
            synth_ct = load_medical_image(synth_ct_path)
            ref_ct = load_medical_image(ref_ct_path)
            
            # Create processing thread
            from app.core.evaluation.evaluate_synthetic_ct import evaluate_synthetic_ct
            
            self.processing_thread = ProcessingThread(
                evaluate_synthetic_ct,
                synth_ct,
                ref_ct,
                metrics=metrics,
                regions=regions,
                output_dir=output_dir,
                generate_report=self.generate_report.isChecked()
            )
            
            # Connect signals
            self.processing_thread.finished.connect(self.evaluationFinished)
            self.processing_thread.error.connect(self.processingError)
            self.processing_thread.progress.connect(self.updateProgress)
            
            # Update UI
            self.progress_bar.setVisible(True)
        except Exception as e:
            logging.error(f"Error evaluating synthetic CT: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error evaluating synthetic CT: {str(e)}")

    def evaluationFinished(self, result):
        """Handle completion of evaluation"""
        self.progress_bar.setVisible(False)
        
        # Display results in the metrics text area
        self.metrics_text.clear()
        
        if isinstance(result, dict):
            # Display overall metrics
            self.metrics_text.append("=== Overall Metrics ===")
            for metric, value in result.items():
                if isinstance(value, dict):
                    continue  # Skip nested dictionaries for now
                self.metrics_text.append(f"{metric}: {value:.3f}")
                
            # Display metrics by region if available
            if 'by_region' in result and isinstance(result['by_region'], dict):
                self.metrics_text.append("\n=== Metrics by Region ===")
                for region, metrics in result['by_region'].items():
                    self.metrics_text.append(f"\nRegion: {region}")
                    for metric, value in metrics.items():
                        self.metrics_text.append(f"  {metric}: {value:.3f}")
                        
            # Create and show difference visualization if available
            if 'difference_image' in result:
                self.diff_viewer.set_image(result['difference_image'])
                self.results_tabs.setCurrentIndex(1)  # Switch to difference view tab
                
            # Show results path if available
            if 'report_path' in result and result['report_path']:
                QMessageBox.information(self, "Evaluation Complete", 
                                      f"Evaluation completed successfully.\nResults saved to: {result['report_path']}")
            else:
                QMessageBox.information(self, "Evaluation Complete", "Evaluation completed successfully.")
        else:
            self.metrics_text.append("Error: Unexpected result format")
            QMessageBox.warning(self, "Warning", "Evaluation completed, but with unexpected result format.")

    def createAdvancedOptions(self, parent_layout):
        """Create collapsible advanced options section"""
        # Create a group box for advanced options
        advanced_box = QGroupBox("Advanced Options")
        advanced_box.setCheckable(True)
        advanced_box.setChecked(False)
        parent_layout.addWidget(advanced_box)
        
        # Create layout for advanced options
        advanced_layout = QVBoxLayout(advanced_box)
        
        # Patch Size Group
        patch_group = QGroupBox("Patch Settings")
        patch_layout = QFormLayout(patch_group)
        
        # Patch size
        self.patch_size = QSpinBox()
        self.patch_size.setRange(16, 256)
        self.patch_size.setValue(64)
        self.patch_size.setSingleStep(16)
        patch_layout.addRow("Patch Size:", self.patch_size)
        
        # Batch size 
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 16)
        self.batch_size.setValue(4)
        patch_layout.addRow("Batch Size:", self.batch_size)
        
        # Use 3D checkbox
        self.use_3d = QCheckBox("Use 3D patches")
        self.use_3d.setChecked(False)
        patch_layout.addRow("", self.use_3d)
        
        # Add patch group to advanced layout
        advanced_layout.addWidget(patch_group)
        
        # Performance Group
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)
        
        # Number of threads
        self.num_threads = QSpinBox()
        self.num_threads.setRange(1, 16)
        self.num_threads.setValue(4)
        perf_layout.addRow("Threads:", self.num_threads)
        
        # GPU memory limit
        self.gpu_memory = QSpinBox()
        self.gpu_memory.setRange(1, 32)
        self.gpu_memory.setValue(4)
        self.gpu_memory.setSuffix(" GB")
        perf_layout.addRow("GPU Memory:", self.gpu_memory)
        
        # Add performance group to advanced layout
        advanced_layout.addWidget(perf_group)
        
        # Output Options Group
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)
        
        # Format
        self.output_format = QComboBox()
        self.output_format.addItems(["NIFTI", "DICOM"])
        output_layout.addRow("Format:", self.output_format)
        
        # Compression
        self.use_compression = QCheckBox("Use compression")
        self.use_compression.setChecked(True)
        output_layout.addRow("", self.use_compression)
        
        # Save intermediates
        self.save_intermediates = QCheckBox("Save intermediate results")
        self.save_intermediates.setChecked(False)
        output_layout.addRow("", self.save_intermediates)
        
        # Add output group to advanced layout
        advanced_layout.addWidget(output_group)
        
        return advanced_box

    def updateTrainingStatus(self, message):
        """Update status message for training"""
        self.training_log.append(message)
        # Ensure view scrolls to bottom to show latest log entry
        self.training_log.verticalScrollBar().setValue(self.training_log.verticalScrollBar().maximum())

# Class to capture all wheel events at the application level
class WheelEventFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            # Completely block all wheel events from propagating up the hierarchy
            return True
        return False
        
class CollapsibleBox(QWidget):
    """Collapsible box that can be expanded/collapsed"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        
        # Set layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create toggle button
        self.toggle_button = QToolButton()
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        
        # Create content area
        self.content_area = QScrollArea()
        self.content_area.setStyleSheet("QScrollArea { border: none; }")
        self.content_area.setWidgetResizable(True)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        
        # Create widget to hold the content
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_area.setWidget(self.content)
        
        # Add toggle and content area to layout
        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.content_area)
        
        # Connect toggle signal
        self.toggle_button.toggled.connect(self.on_toggle)
        
    def on_toggle(self, checked):
        """Toggle visibility of the content area when button is clicked"""
        # Use arrow to show toggle state
        arrow_type = Qt.DownArrow if checked else Qt.RightArrow
        self.toggle_button.setArrowType(arrow_type)
        
        # Set maximum height to the desired maximum if checked
        # or zero if not, to show/hide content
        content_height = self.content_layout.sizeHint().height()
        
        # Use animation for smooth toggling
        animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        animation.setDuration(200)
        animation.setStartValue(0 if checked else content_height)
        animation.setEndValue(content_height if checked else 0)
        animation.start()
        
    def addWidget(self, widget):
        """Add a widget to the collapsible box's content area"""
        self.content_layout.addWidget(widget)
        

class ProcessingThread(QThread):
    """Thread for processing operations"""
    
    finished = Signal(object)  # Signal for successful completion
    error = Signal(str)        # Signal for error
    progress = Signal(int)     # Signal for progress updates
    status = Signal(str)       # Signal for status messages
    
    def __init__(self, operation, *args, **kwargs):
        """Initialize with operation and arguments"""
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        
        # Add progress callback to kwargs if applicable
        if 'progress_callback' not in self.kwargs:
            self.kwargs['progress_callback'] = self.report_progress
        
        self._is_interrupted = False
        
    def run(self):
        """Run the operation"""
        try:
            result = self.operation(*self.args, **self.kwargs)
            if not self._is_interrupted:
                self.finished.emit(result)
        except Exception as e:
            if not self._is_interrupted:
                self.error.emit(str(e))
                
    def report_progress(self, value, message=None):
        """Report progress and optional status message"""
        self.progress.emit(value)
        if message:
            self.status.emit(message)
            
    def requestInterruption(self):
        """Request interruption of the thread"""
        super().requestInterruption()
        self._is_interrupted = True
        # If operation supports interruption
        if 'interrupt_flag' in self.kwargs:
            self.kwargs['interrupt_flag'] = True