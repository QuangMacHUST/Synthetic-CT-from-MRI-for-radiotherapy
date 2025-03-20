#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the Synthetic CT Generator GUI application
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import PySide6
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
except ImportError:
    print("Error: PySide6 is not installed.")
    print("Please install it using: pip install PySide6")
    sys.exit(1)

# Import application modules
try:
    from app.utils.io_utils import setup_logging
    from app.utils.config_utils import get_config
    
    # Try to import EnhancedMainWindow first
    try:
        from app.gui.enhanced_gui import EnhancedMainWindow
    except ImportError:
        # Fallback to MainWindow if EnhancedMainWindow is not available
        from app.gui.main_window import MainWindow as EnhancedMainWindow
        logging.warning("EnhancedMainWindow not found, using MainWindow instead")
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please make sure you are running this script from the correct directory.")
    sys.exit(1)


def main(theme='system'):
    """
    Run the Synthetic CT application with GUI.
    
    Args:
        theme: GUI theme (light, dark, system)
    """
    # Set up logging
    setup_logging()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Synthetic CT Generator")
    
    # Get configuration for GUI
    config = get_config()
    gui_config = config.get_gui_params()
    
    # Use theme from config if not specified
    if theme == 'system' and 'theme' in gui_config:
        theme = gui_config.get('theme', 'system')
    
    # Set theme
    if theme == 'dark':
        app.setStyle('Fusion')
        app.setPalette(create_dark_palette())
    elif theme == 'light':
        app.setStyle('Fusion')
        app.setPalette(app.style().standardPalette())
    # System theme is the default
    
    # Create main window
    main_window = EnhancedMainWindow()
    main_window.show()
    
    # Run application
    return app.exec()


def create_dark_palette():
    """Create a dark palette for the application."""
    from PySide6.QtGui import QPalette, QColor
    from PySide6.QtCore import Qt
    
    dark_palette = QPalette()
    
    # Set colors
    dark_color = QColor(45, 45, 45)
    disabled_color = QColor(127, 127, 127)
    text_color = QColor(255, 255, 255)
    
    # Set color groups
    dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
    dark_palette.setColor(QPalette.ColorRole.WindowText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, text_color)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, text_color)
    dark_palette.setColor(QPalette.ColorRole.Text, text_color)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
    dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
    dark_palette.setColor(QPalette.ColorRole.ButtonText, text_color)
    dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    
    return dark_palette


if __name__ == "__main__":
    # Parse command line arguments (if needed)
    import argparse
    parser = argparse.ArgumentParser(description='Synthetic CT Generator GUI')
    parser.add_argument('--theme', choices=['light', 'dark', 'system'], default='system',
                        help='GUI theme (light, dark, system)')
    args = parser.parse_args()
    
    # Run application
    sys.exit(main(theme=args.theme)) 