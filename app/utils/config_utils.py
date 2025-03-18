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

# Set up logger
logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "default_config.yaml"
)


class ConfigManager:
    """Manage application configuration."""
    
    def __init__(self, config_path=None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file (None to use default)
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """Load configuration from file."""
        # Start with default configuration
        default_config = self._load_yaml_config(DEFAULT_CONFIG_PATH)
        
        # Override with user configuration if provided
        if self.config_path and os.path.exists(self.config_path):
            user_config = self._load_yaml_config(self.config_path)
            self.config = self._deep_update(default_config, user_config)
        else:
            self.config = default_config
            
        return self.config
    
    def _load_yaml_config(self, file_path):
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {str(e)}")
            if file_path == DEFAULT_CONFIG_PATH:
                logger.critical("Failed to load default configuration. Using empty configuration.")
                return {}
            else:
                logger.warning("Using default configuration.")
                return self._load_yaml_config(DEFAULT_CONFIG_PATH)
    
    def _deep_update(self, d1, d2):
        """
        Recursively update dictionary d1 with values from d2.
        
        Args:
            d1: Base dictionary
            d2: Dictionary with values to update
            
        Returns:
            Updated dictionary
        """
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            return d2
            
        result = d1.copy()
        for k, v in d2.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_update(result[k], v)
            else:
                result[k] = v
        
        return result
    
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
    
    def get(self, *keys, default=None):
        """
        Get value from configuration using nested keys.
        
        Args:
            *keys: Key path to access nested configuration
            default: Default value if key not found
            
        Returns:
            Configuration value or default
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
            *keys: Key path to access nested configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not keys:
            return False
            
        current = self.config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
        return True
    
    def update(self, updates):
        """
        Update configuration with dictionary of updates.
        
        Args:
            updates: Dictionary with updates
            
        Returns:
            Updated configuration
        """
        self.config = self._deep_update(self.config, updates)
        return self.config
    
    def get_conversion_params(self, method, region):
        """
        Get conversion parameters for specified method and region.
        
        Args:
            method: Conversion method ('atlas', 'cnn', 'gan')
            region: Anatomical region ('head', 'pelvis', 'thorax')
            
        Returns:
            Conversion parameters as dictionary
        """
        return self.get('conversion', method, region, default={})
    
    def get_preprocessing_params(self):
        """Get preprocessing parameters."""
        return self.get('preprocessing', default={})
    
    def get_segmentation_params(self, region):
        """
        Get segmentation parameters for specified region.
        
        Args:
            region: Anatomical region ('head', 'pelvis', 'thorax')
            
        Returns:
            Segmentation parameters as dictionary
        """
        return self.get('segmentation', region, default={})
    
    def get_evaluation_params(self):
        """Get evaluation parameters."""
        return self.get('evaluation', default={})
    
    def get_gui_params(self):
        """Get GUI parameters."""
        return self.get('gui', default={})


# Global configuration manager instance
_config_manager = None


def load_config(config_path=None):
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (None to use default)
        
    Returns:
        Loaded configuration
    """
    global _config_manager
    
    if _config_manager is None or (_config_manager.config_path != config_path and config_path is not None):
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_config():
    """
    Get current configuration.
    
    Returns:
        Current configuration
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


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