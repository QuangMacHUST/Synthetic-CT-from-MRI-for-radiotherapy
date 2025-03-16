#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for the synthetic CT application GUI
"""

import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.gui.main_window import MainWindow, run_gui
from app.utils.io_utils import setup_logging


def main():
    """
    Main entry point for the GUI application.
    """
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Run the GUI
    run_gui()


if __name__ == "__main__":
    main() 