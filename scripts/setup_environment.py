#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for setting up and verifying the environment for MRI to CT conversion.
This script installs dependencies, checks environment, and prepares directories.
"""

import os
import sys
import argparse
import logging
import subprocess
import platform
import importlib
import shutil
from pathlib import Path
import pkg_resources
from typing import List, Dict, Tuple, Optional, Union
import json

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Required packages with minimum versions
REQUIRED_PACKAGES = {
    "numpy": "1.20.0",
    "SimpleITK": "2.0.0",
    "torch": "1.9.0",
    "pandas": "1.3.0",
    "matplotlib": "3.4.0",
    "scikit-image": "0.18.0",
    "scikit-learn": "0.24.0",
    "nibabel": "3.2.0",
    "pydicom": "2.2.0",
    "tqdm": "4.61.0",
    "pyyaml": "5.4.0",
    "tensorboard": "2.6.0",
    "opencv-python": "4.5.0",
    "pillow": "8.2.0",
    "reportlab": "3.5.0",
}

# Optional packages
OPTIONAL_PACKAGES = {
    "pytorch-lightning": "1.4.0",  # For advanced training
    "plotly": "5.1.0",             # For interactive visualizations
    "ipywidgets": "7.6.0",         # For interactive notebooks
    "jupyterlab": "3.0.0",         # For development notebooks
    "albumentations": "1.0.0",     # For advanced data augmentation
}

def check_system_requirements() -> Tuple[bool, Dict[str, str]]:
    """
    Check system requirements including OS, CPU, and GPU.
    
    Returns:
        Tuple containing a boolean (requirements met) and a dict of system info
    """
    logger.info("Checking system requirements")
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "gpu": "Unknown",
        "ram_gb": "Unknown"
    }
    
    # Check for GPU (CUDA) support
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch
            system_info["gpu"] = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "None"
            system_info["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        logger.warning(f"Failed to detect GPU: {str(e)}")
    
    # Check RAM
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong)
                ]
            memory_status = MEMORYSTATUS()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
            system_info["ram_gb"] = round(memory_status.dwTotalPhys / (1024**3), 2)
        elif platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            total_memory = [line for line in meminfo.split('\n') if 'MemTotal' in line][0]
            total_kb = int(total_memory.split(':')[1].strip().split(' ')[0])
            system_info["ram_gb"] = round(total_kb / (1024**2), 2)
        elif platform.system() == "Darwin":  # macOS
            output = subprocess.check_output(['sysctl', 'hw.memsize']).decode('utf-8')
            total_bytes = int(output.split(':')[1].strip())
            system_info["ram_gb"] = round(total_bytes / (1024**3), 2)
    except Exception as e:
        logger.warning(f"Failed to detect RAM: {str(e)}")
    
    # Check if system meets minimum requirements
    requirements_met = True
    
    # Minimum RAM requirement: 8GB
    if system_info["ram_gb"] != "Unknown" and float(system_info["ram_gb"]) < 8:
        logger.warning(f"Low RAM detected: {system_info['ram_gb']}GB (min recommended: 8GB)")
        requirements_met = False
    
    # For deep learning, recommend GPU
    if system_info["gpu"] == "None" or system_info["gpu"] == "Unknown":
        logger.warning("No GPU detected. Deep learning tasks will be slow.")
        # Don't fail for this, but warn
    
    # Python version check
    if sys.version_info < (3, 7):
        logger.warning(f"Python version {platform.python_version()} is below minimum required (3.7)")
        requirements_met = False
    
    return requirements_met, system_info

def check_packages(required_only: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    Check if required packages are installed with correct versions.
    
    Args:
        required_only: If True, only check required packages, not optional ones
        
    Returns:
        Tuple containing a boolean (all packages OK), list of missing packages, 
        and list of packages with version mismatches
    """
    logger.info("Checking package requirements")
    
    packages_to_check = REQUIRED_PACKAGES.copy()
    if not required_only:
        packages_to_check.update(OPTIONAL_PACKAGES)
    
    missing_packages = []
    version_mismatches = []
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package, min_version in packages_to_check.items():
        package_key = package.lower().replace('-', '_')
        
        if package_key not in installed_packages:
            missing_packages.append(package)
            continue
        
        installed_version = installed_packages[package_key]
        
        # Compare versions
        try:
            if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(min_version):
                version_mismatches.append(f"{package}: installed={installed_version}, required>={min_version}")
        except Exception as e:
            logger.warning(f"Error comparing versions for {package}: {str(e)}")
    
    # Special case for torch with CUDA
    if 'torch' not in missing_packages and 'torch' in packages_to_check:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("PyTorch installed but CUDA is not available")
        except ImportError:
            # If we get here, torch should already be in missing_packages
            pass
    
    all_ok = len(missing_packages) == 0 and len(version_mismatches) == 0
    
    return all_ok, missing_packages, version_mismatches

def install_requirements(req_file: str = None, missing_packages: List[str] = None) -> bool:
    """
    Install requirements from a file or a list of packages.
    
    Args:
        req_file: Path to requirements.txt file
        missing_packages: List of package names to install
        
    Returns:
        True if installation was successful, False otherwise
    """
    if req_file is not None and os.path.exists(req_file):
        logger.info(f"Installing requirements from {req_file}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {str(e)}")
            return False
    
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                if package in REQUIRED_PACKAGES:
                    version_spec = f"{package}>={REQUIRED_PACKAGES[package]}"
                    subprocess.check_call([sys.executable, "-m", "pip", "install", version_spec])
                elif package in OPTIONAL_PACKAGES:
                    version_spec = f"{package}>={OPTIONAL_PACKAGES[package]}"
                    subprocess.check_call([sys.executable, "-m", "pip", "install", version_spec])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {str(e)}")
            return False
    
    return True

def create_directory_structure(base_dir: str = None) -> bool:
    """
    Create the directory structure for the project.
    
    Args:
        base_dir: Base directory for the project (default: current directory)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating directory structure")
    
    # Use current directory if not specified
    if base_dir is None:
        base_dir = os.getcwd()
    
    base_path = Path(base_dir)
    
    # Define directory structure
    directories = [
        base_path / "data" / "input" / "mri",
        base_path / "data" / "input" / "ct",
        base_path / "data" / "output" / "preprocessed",
        base_path / "data" / "output" / "segmented",
        base_path / "data" / "output" / "synthetic_ct",
        base_path / "data" / "output" / "evaluation",
        base_path / "models" / "cnn",
        base_path / "models" / "gan",
        base_path / "models" / "atlas",
        base_path / "logs",
        base_path / "configs"
    ]
    
    # Create directories
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            return False
    
    return True

def create_default_config(config_dir: str) -> str:
    """
    Create default configuration file.
    
    Args:
        config_dir: Directory to save configuration file
        
    Returns:
        Path to created configuration file
    """
    logger.info("Creating default configuration file")
    
    config_path = Path(config_dir) / "default_config.yaml"
    
    # Default configuration
    config = {
        "general": {
            "project_name": "MRI-to-CT Conversion",
            "data_dir": "data",
            "models_dir": "models",
            "logs_dir": "logs",
            "random_seed": 42
        },
        "preprocessing": {
            "apply_bias_field_correction": True,
            "apply_denoising": True,
            "normalize": True,
            "resample": True,
            "target_spacing": [1.0, 1.0, 1.0]
        },
        "segmentation": {
            "methods": {
                "head": "atlas",
                "pelvis": "threshold",
                "thorax": "threshold"
            },
            "threshold_values": {
                "air": [-1000, -800],
                "lung": [-800, -500],
                "fat": [-100, -50],
                "soft_tissue": [-50, 100],
                "bone": [300, 3000]
            }
        },
        "conversion": {
            "default_method": "cnn",
            "save_intermediate_results": True,
            "cnn": {
                "model_path": "models/cnn/unet3d_latest.pth",
                "batch_size": 1,
                "patch_size": [64, 64, 64],
                "use_gpu": True
            },
            "gan": {
                "model_path": "models/gan/cyclegan_latest.pth",
                "batch_size": 1,
                "patch_size": [64, 64, 64],
                "use_gpu": True
            },
            "atlas": {
                "atlas_dir": "models/atlas",
                "registration_method": "rigid+affine+deformable",
                "num_atlas": 5
            }
        },
        "evaluation": {
            "metrics": ["MAE", "MSE", "PSNR", "SSIM"],
            "generate_reports": True,
            "dvh_metrics": {
                "dose_levels": [20, 30, 40],
                "volume_levels": [5, 50, 95]
            }
        },
        "training": {
            "cnn": {
                "batch_size": 8,
                "patch_size": [64, 64, 64],
                "epochs": 100,
                "learning_rate": 1e-4,
                "val_ratio": 0.2,
                "test_ratio": 0.1,
                "loss_function": "L1",
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "augmentation": True
            },
            "gan": {
                "batch_size": 4,
                "patch_size": [64, 64, 64],
                "epochs": 200,
                "learning_rate_g": 1e-4,
                "learning_rate_d": 1e-4,
                "val_ratio": 0.2,
                "test_ratio": 0.1,
                "lambda_cycle": 10.0,
                "lambda_identity": 5.0,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau",
                "augmentation": True
            }
        },
        "gui": {
            "theme": "dark",
            "window_size": [1280, 720],
            "default_visualization": "overlay"
        }
    }
    
    # Save configuration to YAML file
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created default configuration at {config_path}")
        return str(config_path)
    except Exception as e:
        logger.error(f"Failed to create default configuration: {str(e)}")
        return ""

def check_for_cuda() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    logger.info("Checking for CUDA availability")
    
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                
                logger.info(f"CUDA is available: version {cuda_version}")
                logger.info(f"Found {device_count} GPU(s): {device_name}")
            else:
                logger.warning("CUDA is not available. GPU acceleration will not be used.")
            
            return cuda_available
    except Exception as e:
        logger.warning(f"Error checking CUDA: {str(e)}")
    
    return False

def save_system_info(system_info: Dict[str, str], output_file: str) -> None:
    """
    Save system information to a file.
    
    Args:
        system_info: Dictionary of system information
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(system_info, f, indent=4)
        logger.info(f"System information saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save system information: {str(e)}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Set up environment for MRI to CT conversion")
    
    parser.add_argument("--base_dir", help="Base directory for the project")
    parser.add_argument("--install_requirements", action="store_true", 
                       help="Install required packages")
    parser.add_argument("--check_only", action="store_true", 
                       help="Only check requirements without installing")
    parser.add_argument("--include_optional", action="store_true", 
                       help="Include optional packages")
    parser.add_argument("--create_directories", action="store_true", 
                       help="Create directory structure")
    parser.add_argument("--create_config", action="store_true", 
                       help="Create default configuration file")
    parser.add_argument("--all", action="store_true", 
                       help="Perform all setup tasks")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0
    
    # Set base directory
    base_dir = args.base_dir or os.getcwd()
    logger.info(f"Using base directory: {base_dir}")
    
    # Check system requirements
    requirements_met, system_info = check_system_requirements()
    
    if not requirements_met:
        logger.warning("System does not meet minimum requirements.")
        if not args.check_only:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    # Save system info
    if args.all or args.check_only:
        system_info_path = os.path.join(base_dir, "system_info.json")
        save_system_info(system_info, system_info_path)
    
    # Check packages
    packages_ok, missing_packages, version_mismatches = check_packages(not args.include_optional)
    
    if not packages_ok:
        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        if version_mismatches:
            logger.warning(f"Version mismatches: {', '.join(version_mismatches)}")
        
        if args.install_requirements or args.all:
            # Install missing packages
            if missing_packages:
                logger.info("Installing missing packages...")
                install_requirements(missing_packages=missing_packages)
            
            # Check again after installation
            packages_ok, still_missing, still_mismatched = check_packages(not args.include_optional)
            
            if not packages_ok:
                logger.error("Failed to install all required packages.")
                if still_missing:
                    logger.error(f"Still missing: {', '.join(still_missing)}")
                if still_mismatched:
                    logger.error(f"Still mismatched: {', '.join(still_mismatched)}")
                return 1
        elif args.check_only:
            logger.warning("Package requirements not met. Use --install_requirements to install.")
        else:
            logger.error("Package requirements not met. Setup aborted.")
            return 1
    
    # Create directory structure
    if args.create_directories or args.all:
        if not create_directory_structure(base_dir):
            logger.error("Failed to create directory structure.")
            return 1
    
    # Create default configuration
    if args.create_config or args.all:
        config_dir = os.path.join(base_dir, "configs")
        os.makedirs(config_dir, exist_ok=True)
        config_path = create_default_config(config_dir)
        if not config_path:
            logger.error("Failed to create default configuration file.")
            return 1
    
    # Check for CUDA (only informational)
    check_for_cuda()
    
    logger.info("Environment setup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 