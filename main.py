#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for testing the application.
"""

import sys
import os
import traceback

# Add the project root to Python path to ensure all imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function for testing the application."""
    print("Starting the application test...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Try to import app module
    try:
        import app
        print(f"Successfully imported app module (version: {app.__version__})")
    except Exception as e:
        print(f"Error importing app module: {str(e)}")
        traceback.print_exc()
        return 1
    
    # Try to import core modules
    try:
        from app import core
        print("Successfully imported core module")
        
        # Try to import core submodules
        try:
            from app.core import preprocessing, segmentation, conversion, evaluation
            print("Successfully imported core submodules")
        except Exception as e:
            print(f"Error importing core submodules: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error importing core module: {str(e)}")
        traceback.print_exc()
    
    # Try to import utils modules
    try:
        from app import utils
        print("Successfully imported utils module")
        
        # Try to import utils submodules
        try:
            from app.utils import config_utils, io_utils, logging_utils, visualization
            print("Successfully imported utils submodules")
        except Exception as e:
            print(f"Error importing utils submodules: {str(e)}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error importing utils module: {str(e)}")
        traceback.print_exc()
    
    # Try to import visualization module
    try:
        from app import visualization
        print("Successfully imported visualization module")
    except Exception as e:
        print(f"Error importing visualization module: {str(e)}")
        traceback.print_exc()
    
    # Try to import deployment module
    try:
        from app import deployment
        print("Successfully imported deployment module")
    except Exception as e:
        print(f"Error importing deployment module: {str(e)}")
        traceback.print_exc()
    
    print("Application test completed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 