#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for packaging the MRI to CT conversion project for distribution.
This script creates a compressed package containing all required files.
"""

import os
import sys
import argparse
import logging
import shutil
import zipfile
import json
import datetime
import platform
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Optional

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Files and directories to include by default
DEFAULT_INCLUDE = [
    "app",
    "configs",
    "scripts",
    "setup.py",
    "requirements.txt",
    "README.md",
    "LICENSE",
]

# Files and directories to exclude by default
DEFAULT_EXCLUDE = [
    "*/__pycache__/*",
    "*.pyc",
    "*/.DS_Store",
    "*/.git/*",
    "*/.idea/*",
    "*/.vscode/*",
    "*.log",
    "*.zip",
    "data/*",
    "models/*",
    "logs/*",
    "*.ipynb_checkpoints*",
]

def should_include_path(path: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    """
    Check if a path should be included based on include and exclude patterns.
    
    Args:
        path: Path to check
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        True if path should be included, False otherwise
    """
    from fnmatch import fnmatch
    
    # Convert path to use forward slashes for consistent pattern matching
    path = path.replace("\\", "/")
    
    # Check exclude patterns first
    for pattern in exclude_patterns:
        if fnmatch(path, pattern):
            return False
    
    # If include patterns are specified, path must match at least one
    if include_patterns:
        for pattern in include_patterns:
            if fnmatch(path, pattern):
                return True
        return False
    
    # If no include patterns specified, include by default
    return True

def get_file_list(
    base_dir: str, 
    include_patterns: List[str] = None, 
    exclude_patterns: List[str] = None
) -> List[str]:
    """
    Get a list of files to include in the package.
    
    Args:
        base_dir: Base directory of the project
        include_patterns: List of glob patterns to include
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        List of file paths to include
    """
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE
    
    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE
    
    included_files = []
    
    for root, dirs, files in os.walk(base_dir):
        # Skip directories that match exclude patterns
        dirs[:] = [d for d in dirs if should_include_path(
            os.path.join(root, d).replace(base_dir + os.sep, ""), 
            include_patterns, 
            exclude_patterns
        )]
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = file_path.replace(base_dir + os.sep, "")
            
            if should_include_path(rel_path, include_patterns, exclude_patterns):
                included_files.append(file_path)
    
    return included_files

def create_zip_package(
    base_dir: str, 
    output_path: str, 
    file_list: List[str],
    include_timestamp: bool = True
) -> str:
    """
    Create a ZIP package of the project files.
    
    Args:
        base_dir: Base directory of the project
        output_path: Path to save the ZIP package
        file_list: List of files to include
        include_timestamp: Whether to include a timestamp in the filename
        
    Returns:
        Path to the created ZIP package
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add timestamp to filename if requested
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(output_path)
        output_path = f"{filename}_{timestamp}{ext}"
    
    logger.info(f"Creating ZIP package at {output_path}")
    
    # Create ZIP file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in file_list:
            rel_path = os.path.relpath(file, base_dir)
            logger.debug(f"Adding file: {rel_path}")
            zipf.write(file, rel_path)
    
    logger.info(f"Created ZIP package with {len(file_list)} files")
    return output_path

def create_manifest(file_list: List[str], base_dir: str, output_path: str) -> str:
    """
    Create a manifest file listing all files in the package.
    
    Args:
        file_list: List of files included in the package
        base_dir: Base directory of the project
        output_path: Path to save the manifest file
        
    Returns:
        Path to the created manifest file
    """
    manifest = {
        "package_info": {
            "name": "MRI to CT Conversion",
            "version": "1.0.0",
            "created_at": datetime.datetime.now().isoformat(),
            "created_by": f"{platform.node()}",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "files": []
    }
    
    # Add files to manifest
    for file in file_list:
        rel_path = os.path.relpath(file, base_dir)
        file_stat = os.stat(file)
        
        manifest["files"].append({
            "path": rel_path,
            "size_bytes": file_stat.st_size,
            "last_modified": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        })
    
    # Save manifest to file
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created manifest at {output_path}")
    return output_path

def get_git_info(base_dir: str) -> Dict:
    """
    Get Git repository information if available.
    
    Args:
        base_dir: Base directory of the project
        
    Returns:
        Dictionary containing Git information or empty dict if not a Git repository
    """
    git_info = {}
    
    try:
        # Check if directory is a Git repository
        if not os.path.isdir(os.path.join(base_dir, '.git')):
            return git_info
        
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        git_info['branch'] = result.stdout.strip()
        
        # Get latest commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        git_info['commit'] = result.stdout.strip()
        
        # Get latest commit date
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cd', '--date=iso'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        git_info['date'] = result.stdout.strip()
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        git_info['remote'] = result.stdout.strip()
        
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.warning(f"Failed to get Git information: {str(e)}")
    
    return git_info

def create_version_info(base_dir: str, output_path: str) -> str:
    """
    Create a version info file for the package.
    
    Args:
        base_dir: Base directory of the project
        output_path: Path to save the version info file
        
    Returns:
        Path to the created version info file
    """
    version_info = {
        "name": "MRI to CT Conversion",
        "version": "1.0.0",
        "created_at": datetime.datetime.now().isoformat(),
        "created_by": f"{platform.node()}",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    
    # Add Git information if available
    git_info = get_git_info(base_dir)
    if git_info:
        version_info["git"] = git_info
    
    # Save version info to file
    with open(output_path, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    logger.info(f"Created version info at {output_path}")
    return output_path

def copy_documentation(base_dir: str, output_dir: str) -> None:
    """
    Copy and organize documentation files.
    
    Args:
        base_dir: Base directory of the project
        output_dir: Directory to save documentation files
    """
    logger.info("Copying documentation files")
    
    # Create documentation directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to copy
    doc_files = {
        "README.md": "README.md",
        "LICENSE": "LICENSE.txt",
        "configs/default_config.yaml": "configs/default_config.yaml",
    }
    
    # Copy files
    for src_rel, dst_rel in doc_files.items():
        src_path = os.path.join(base_dir, src_rel)
        dst_path = os.path.join(output_dir, dst_rel)
        
        if os.path.exists(src_path):
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {src_rel} to {dst_path}")
        else:
            logger.warning(f"Documentation file not found: {src_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Package MRI to CT conversion project")
    
    parser.add_argument("--base_dir", default=".", 
                       help="Base directory of the project")
    parser.add_argument("--output", default="dist/mri_to_ct_conversion.zip", 
                       help="Path to save the ZIP package")
    parser.add_argument("--include", nargs="+", 
                       help="Additional patterns to include")
    parser.add_argument("--exclude", nargs="+", 
                       help="Additional patterns to exclude")
    parser.add_argument("--no-timestamp", action="store_true", 
                       help="Don't include timestamp in filename")
    parser.add_argument("--manifest", action="store_true", 
                       help="Create a manifest file")
    parser.add_argument("--version-info", action="store_true", 
                       help="Create a version info file")
    parser.add_argument("--docs", action="store_true", 
                       help="Copy documentation files")
    parser.add_argument("--all", action="store_true", 
                       help="Include all optional files (manifest, version info, docs)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logger level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Resolve base directory
    base_dir = os.path.abspath(args.base_dir)
    logger.info(f"Using base directory: {base_dir}")
    
    # Combine include and exclude patterns
    include_patterns = DEFAULT_INCLUDE.copy()
    exclude_patterns = DEFAULT_EXCLUDE.copy()
    
    if args.include:
        include_patterns.extend(args.include)
    
    if args.exclude:
        exclude_patterns.extend(args.exclude)
    
    # Get list of files to include
    file_list = get_file_list(base_dir, include_patterns, exclude_patterns)
    logger.info(f"Found {len(file_list)} files to include")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create manifest if requested
    if args.manifest or args.all:
        manifest_path = os.path.join(base_dir, "dist", "manifest.json")
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        create_manifest(file_list, base_dir, manifest_path)
        
        # Add manifest to file list
        if os.path.exists(manifest_path) and manifest_path not in file_list:
            file_list.append(manifest_path)
    
    # Create version info if requested
    if args.version_info or args.all:
        version_path = os.path.join(base_dir, "dist", "version.json")
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        create_version_info(base_dir, version_path)
        
        # Add version info to file list
        if os.path.exists(version_path) and version_path not in file_list:
            file_list.append(version_path)
    
    # Copy documentation if requested
    if args.docs or args.all:
        docs_dir = os.path.join(base_dir, "dist", "docs")
        copy_documentation(base_dir, docs_dir)
        
        # Add documentation files to file list (if they aren't already included)
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path not in file_list:
                    file_list.append(file_path)
    
    # Create ZIP package
    zip_path = create_zip_package(
        base_dir, 
        args.output, 
        file_list, 
        not args.no_timestamp
    )
    
    logger.info(f"Package created successfully: {zip_path}")
    logger.info(f"Package size: {os.path.getsize(zip_path) / (1024*1024):.2f} MB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 