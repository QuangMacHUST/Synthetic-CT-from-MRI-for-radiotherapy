#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Configuration utilities for MRI to CT conversion.
Provides tools to load, save, and manage configuration.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Import default region parameters from separate file
try:
    from app.utils.default_region_params import DEFAULT_REGION_PARAMS
except ImportError:
    # Define DEFAULT_REGION_PARAMS here if import fails
    from copy import deepcopy
    DEFAULT_REGION_PARAMS = {}
    logging.warning("Could not import DEFAULT_REGION_PARAMS from default_region_params.py")

# Set up logger
logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "default_config.yaml"
)

# Default region-specific parameters
DEFAULT_REGION_PARAMS = {
    "brain": {
        "window_width": 80,
        "window_level": 40,
        "tissue_classes": ["csf", "gray_matter", "white_matter", "bone", "air"],
        "hu_ranges": {
            "csf": [-10, 15],
            "gray_matter": [20, 40],
            "white_matter": [30, 45],
            "bone": [400, 1000],
            "air": [-1000, -800]
        },
        "segmentation_method": "threshold",
        "registration_params": {
            "transform_type": "rigid",
            "metric": "mutual_information",
            "optimizer": "gradient_descent",
            "sampling_percentage": 0.1
        }
    },
    "head_neck": {
        "window_width": 350,
        "window_level": 40,
        "tissue_classes": ["soft_tissue", "bone", "air", "fat"],
        "hu_ranges": {
            "soft_tissue": [-100, 100],
            "bone": [300, 1500],
            "air": [-1000, -800],
            "fat": [-120, -80]
        },
        "segmentation_method": "atlas",
        "registration_params": {
            "transform_type": "affine",
            "metric": "mutual_information",
            "optimizer": "gradient_descent",
            "sampling_percentage": 0.1
        }
    },
    "pelvis": {
        "window_width": 400,
        "window_level": 40,
        "tissue_classes": ["soft_tissue", "bone", "air", "fat"],
        "hu_ranges": {
            "soft_tissue": [-100, 100],
            "bone": [200, 1200],
            "air": [-1000, -800],
            "fat": [-120, -80]
        },
        "segmentation_method": "deep_learning",
        "registration_params": {
            "transform_type": "bspline",
            "metric": "mutual_information",
            "optimizer": "gradient_descent",
            "sampling_percentage": 0.1
        }
    },
    "abdomen": {
        "window_width": 400,
        "window_level": 40,
        "tissue_classes": ["soft_tissue", "bone", "air", "fat", "liver", "kidney"],
        "hu_ranges": {
            "soft_tissue": [-100, 100],
            "bone": [200, 1200],
            "air": [-1000, -800],
            "fat": [-120, -80],
            "liver": [40, 60],
            "kidney": [20, 40]
        },
        "segmentation_method": "atlas",
        "registration_params": {
            "transform_type": "affine",
            "metric": "mutual_information",
            "optimizer": "gradient_descent",
            "sampling_percentage": 0.1
        }
    },
    "thorax": {
        "window_width": 1500,
        "window_level": -600,
        "tissue_classes": ["soft_tissue", "bone", "air", "fat", "lung"],
        "hu_ranges": {
            "soft_tissue": [-100, 100],
            "bone": [200, 1200],
            "air": [-1000, -800],
            "fat": [-120, -80],
            "lung": [-950, -750]
        },
        "segmentation_method": "atlas",
        "registration_params": {
            "transform_type": "affine",
            "metric": "mutual_information",
            "optimizer": "gradient_descent",
            "sampling_percentage": 0.1
        }
    }
}


class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (None to use default)
        """
        self.config_path = config_path if config_path else DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        
    def _load_config(self):
        """
        Load configuration from file.
        
        Returns:
            Loaded configuration
        """
        # Create default config if not exists
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            logger.info("Creating default configuration")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Create default config
            default_config = self._create_default_config()
            
            # Save default config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
                
            return default_config
            
        # Load existing config
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {str(e)}")
            logger.info("Using default configuration")
            return self._create_default_config()
    
    def _create_default_config(self):
        """
        Create default configuration.
        
        Returns:
            Default configuration
        """
        return {
            "app": {
                "name": "Synthetic CT Generator",
                "version": "1.0.0",
                "description": "Generate synthetic CT images from MRI",
                "debug": False,
                "log_level": "INFO",
                "data_dir": "data",
                "output_dir": "output",
                "models_dir": "models",
            },
            "preprocessing": {
                "bias_field": {
                    "enable": True,
                    "shrink_factor": 4,
                    "iterations": 50
                },
                "denoising": {
                    "enable": True,
                    "method": "gaussian",
                    "sigma": 0.5
                },
                "normalization": {
                    "enable": True,
                    "method": "minmax",
                    "min": 0,
                    "max": 1000
                }
            },
            "segmentation": {
                "method": "auto",
                "tissues": ["background", "air", "soft_tissue", "bone", "fat", "csf"]
            },
            "conversion": {
                "method": "atlas",
                "region": "head",
                "batch_size": 1,
                "patch_size": 64,
                "use_3d": False
            },
            "evaluation": {
                "metrics": ["mae", "mse", "psnr", "ssim"],
                "reference_required": False
            },
            "gui": {
                "window_size": [1200, 800],
                "theme": "light",
                "font_size": 10
            },
            "regions": DEFAULT_REGION_PARAMS
        }
    
    def get(self, *keys, default=None):
        """
        Get value from configuration using nested keys.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Value from configuration or default
        """
        result = self.config
        
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
                
        return result
    
    def set(self, value, *keys):
        """
        Set value in configuration using nested keys.
        
        Args:
            value: Value to set
            *keys: Nested keys to access
            
        Returns:
            None
        """
        if not keys:
            return
            
        # Navigate to the parent dictionary
        parent = self.config
        for key in keys[:-1]:
            if key not in parent or not isinstance(parent[key], dict):
                parent[key] = {}
            parent = parent[key]
            
        # Set the value
        parent[keys[-1]] = value
    
    def update(self, updates):
        """
        Update configuration with a dictionary.
        
        Args:
            updates: Dictionary with updates
            
        Returns:
            Updated configuration
        """
        self._update_dict(self.config, updates)
        return self.config
    
    def _update_dict(self, d, u):
        """
        Update a dictionary recursively.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        import collections.abc
        
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping) and k in d and isinstance(d[k], dict):
                d[k] = self._update_dict(d[k], v)
            else:
                d[k] = v
        
        return d
    
    def save_config(self, output_path):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
            
        Returns:
            Path to saved configuration file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved configuration to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving configuration to {output_path}: {str(e)}")
            return None
    
    def get_preprocessing_params(self, region=None):
        """
        Get preprocessing parameters.
        
        Args:
            region: Optional anatomical region for region-specific parameters
            
        Returns:
            Preprocessing parameters
        """
        params = self.get('preprocessing', default={})
        if region and 'regions' in params and region in params['regions']:
            # Merge region-specific params with general params
            region_params = params['regions'][region]
            general_params = {k: v for k, v in params.items() if k != 'regions'}
            return {**general_params, **region_params}
        return params
    
    def get_segmentation_params(self, region=None):
        """
        Get segmentation parameters.
        
        Args:
            region: Optional anatomical region for region-specific parameters
            
        Returns:
            Segmentation parameters
        """
        params = self.get('segmentation', default={})
        if region and region in params:
            # Return region-specific parameters
            return params[region]
        return params
    
    def get_conversion_params(self, method=None, region=None):
        """
        Get conversion parameters.
        
        Args:
            method: Optional conversion method (atlas, cnn, gan)
            region: Optional anatomical region for region-specific parameters
            
        Returns:
            Conversion parameters
        """
        params = self.get('conversion', default={})
        
        # Get method-specific parameters
        if method and method in params:
            method_params = params[method]
            
            # Get region-specific parameters within the method
            if region and region in method_params:
                return method_params[region]
            
            return method_params
            
        return params
    
    def get_evaluation_params(self):
        """Get evaluation parameters."""
        return self.get('evaluation', default={})
    
    def get_gui_params(self):
        """Get GUI parameters."""
        return self.get('gui', default={})

    def get_region_params(self, region: str) -> Dict[str, Any]:
        """
        Get region-specific parameters.
        
        Args:
            region: Anatomical region name
            
        Returns:
            Region-specific parameters dictionary
        """
        # First try to get from config
        region_params = self.get("regions", region)
        
        # Use default if not in config
        if not region_params:
            if region in DEFAULT_REGION_PARAMS:
                logger.info(f"Using default parameters for region: {region}")
                return DEFAULT_REGION_PARAMS[region]
            else:
                logger.warning(f"No parameters found for region: {region}, using brain as default")
                if "brain" in DEFAULT_REGION_PARAMS:
                    return DEFAULT_REGION_PARAMS["brain"]
                else:
                    logger.error("No default parameters available")
                    return {}
        
        return region_params


def get_config():
    """
    Get configuration manager instance.
    
    Returns:
        ConfigManager instance
    """
    # Use module-level singleton for configuration
    if not hasattr(get_config, "_instance"):
        get_config._instance = ConfigManager()
    
    return get_config._instance


def create_default_config(output_path):
    """
    Create default configuration file.
    
    Args:
        output_path: Path to save default configuration
        
    Returns:
        Path to saved configuration file
    """
    config_manager = ConfigManager()
    return config_manager.save_config(output_path)


def update_config_from_args(args):
    """
    Update configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Updated configuration
    """
    # Get current configuration
    config_manager = get_config()
    
    # Create updates from arguments
    updates = {}
    
    # Handle common arguments
    if hasattr(args, 'region') and args.region:
        updates['conversion'] = {'default_region': args.region}
    
    if hasattr(args, 'model') and args.model:
        updates['conversion'] = {'default_method': args.model}
    
    # Mode-specific updates
    if hasattr(args, 'mode'):
        if args.mode == 'preprocess':
            if hasattr(args, 'bias_correction'):
                updates['preprocessing'] = {'bias_field_correction': {'enable': args.bias_correction}}
            if hasattr(args, 'denoise'):
                updates['preprocessing'] = {'denoising': {'enable': args.denoise}}
            if hasattr(args, 'normalize'):
                updates['preprocessing'] = {'normalization': {'enable': args.normalize}}
        
        elif args.mode == 'segment':
            if hasattr(args, 'method') and args.method:
                updates['segmentation'] = {'method': args.method}
        
        elif args.mode == 'evaluate':
            if hasattr(args, 'metrics') and args.metrics:
                metrics_list = args.metrics.split(',')
                updates['evaluation'] = {'metrics': metrics_list}
    
    # Update configuration
    config_manager.update(updates)
    
    return config_manager.config 

def get_region_params(region: str) -> Dict[str, Any]:
    """
    Get region-specific parameters.
    
    Args:
        region: Anatomical region name
        
    Returns:
        Region-specific parameters dictionary
    """
    config_manager = get_config()
    return config_manager.get_region_params(region) 