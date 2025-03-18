#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Startup script for testing and running the application.
This file helps diagnose import and encoding issues.
"""

import os
import sys
import traceback
import codecs
import re

def fix_encoding(filepath):
    """Fix file encoding issues by rewriting the file."""
    print(f"Fixing encoding for: {filepath}")
    try:
        # Read content as binary to avoid encoding errors
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Remove null bytes and BOM
        content = content.replace(b'\x00', b'')
        if content.startswith(b'\xef\xbb\xbf'):  # UTF-8 BOM
            content = content[3:]
        if content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):  # UTF-16 BOM
            content = content[2:]
        
        # Force create a new clean file
        with open(filepath, 'wb') as f:
            f.write(b'#!/usr/bin/env python\n')
            f.write(b'# -*- coding: utf-8 -*-\n\n')
            
            # Extract original content without encoding headers
            lines = content.split(b'\n')
            clean_lines = []
            for line in lines:
                if (line.startswith(b'#!') or 
                    line.startswith(b'# -*-') or 
                    line.startswith(b'# coding') or
                    not line):
                    continue
                clean_lines.append(line)
            
            f.write(b'\n'.join(clean_lines))
        
        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {str(e)}")
        traceback.print_exc()
        return False

def fix_all_python_files(root_dir):
    """Fix encoding for all Python files in the project."""
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                if fix_encoding(filepath):
                    count += 1
    print(f"Fixed encoding for {count} files")

def run_test_imports():
    """Test importing project modules."""
    print("\nTesting imports:")
    
    try:
        import app
        print("✓ Successfully imported app")
        
        try:
            print(f"  App version: {app.__version__}")
        except AttributeError:
            print("  No version information available")
        
        try:
            from app import core
            print("✓ Successfully imported app.core")
            
            from app.core import preprocessing, segmentation, conversion, evaluation
            print("✓ Successfully imported core submodules")
        except ImportError as e:
            print(f"× Error importing core: {str(e)}")
        
        try:
            from app import utils
            print("✓ Successfully imported app.utils")
        except ImportError as e:
            print(f"× Error importing utils: {str(e)}")
        
        try:
            from app import visualization
            print("✓ Successfully imported app.visualization")
        except ImportError as e:
            print(f"× Error importing visualization: {str(e)}")
        
    except ImportError as e:
        print(f"× Error importing app: {str(e)}")
        return False
    
    return True

def main():
    """Main function."""
    print("Synthetic CT Utilities")
    print("=====================")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Make sure the current directory is in the path
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    
    # Fix encoding issues in Python files
    print("\nChecking for encoding issues...")
    fix_all_python_files("app")
    
    # Test imports
    success = run_test_imports()
    
    if success:
        print("\nAll imports successful!")
        
        # Try running the CLI with --help
        print("\nTrying to run the CLI:")
        try:
            from app.cli import main as cli_main
            print("CLI module imported successfully")
            
            # Redirect stdout to capture output
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                try:
                    cli_main()
                except SystemExit:
                    # CLI calls sys.exit(), which is expected
                    pass
            
            output = f.getvalue()
            if output:
                print("CLI executed successfully")
                print("Preview of CLI output:")
                lines = output.split('\n')[:10]  # First 10 lines
                for line in lines:
                    print(f"  {line}")
                if len(lines) > 10:
                    print("  ...")
            else:
                print("CLI executed but produced no output")
            
        except Exception as e:
            print(f"Error running CLI: {str(e)}")
            traceback.print_exc()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 