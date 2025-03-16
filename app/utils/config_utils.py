#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utilities for MRI to CT conversion
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> 'ConfigManager':
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses default config.
        
    Returns:
        ConfigManager object with loaded configuration
    """
    # If config_path is not provided, use default config
    if config_path is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        config_path = os.path.join(root_dir, "configs", "default_config.yaml")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return ConfigManager(config_data, config_path)
    
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {str(e)}")
        raise


class ConfigManager:
    """
    Manager for configuration settings.
    """
    
    def __init__(self, config_data: Dict[str, Any], config_path: str):
        """
        Initialize with configuration data.
        
        Args:
            config_data: Dictionary containing configuration settings
            config_path: Path to the configuration file
        """
        self.config = config_data
        self.config_path = config_path
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section (None for entire section)
            default: Default value if key/section doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            if key is None:
                return self.config.get(section, default)
            else:
                section_data = self.config.get(section, {})
                return section_data.get(key, default)
        
        except Exception as e:
            logger.warning(f"Error getting config value {section}.{key}: {str(e)}")
            return default
    
    def get_nested(self, path: List[str], default: Any = None) -> Any:
        """
        Get nested configuration value using path list.
        
        Args:
            path: List of keys defining path to value
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            current = self.config
            for key in path:
                if key not in current:
                    return default
                current = current[key]
            return current
        
        except Exception as e:
            logger.warning(f"Error getting nested config value {'.'.join(path)}: {str(e)}")
            return default
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key within section
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save configuration (defaults to original path)
        """
        save_path = path if path is not None else self.config_path
        
        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error saving configuration to {save_path}: {str(e)}")
            raise
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """
        Get preprocessing parameters.
        
        Returns:
            Dictionary of preprocessing parameters
        """
        return self.get('preprocessing', default={})
    
    def get_segmentation_params(self, region: str) -> Dict[str, Any]:
        """
        Get segmentation parameters for a specific region.
        
        Args:
            region: Anatomical region ('head', 'pelvis', or 'thorax')
            
        Returns:
            Dictionary of segmentation parameters for the region
        """
        seg_config = self.get('segmentation', default={})
        region_config = seg_config.get(region, {})
        
        # Merge with common settings
        result = {k: v for k, v in seg_config.items() if k != 'head' and k != 'pelvis' and k != 'thorax'}
        result.update(region_config)
        
        return result
    
    def get_conversion_params(self, method: str, region: str) -> Dict[str, Any]:
        """
        Get conversion parameters for a specific method and region.
        
        Args:
            method: Conversion method ('atlas', 'cnn', or 'gan')
            region: Anatomical region ('head', 'pelvis', or 'thorax')
            
        Returns:
            Dictionary of conversion parameters for the method and region
        """
        conversion_config = self.get('conversion', default={})
        method_config = conversion_config.get(method, {})
        region_config = method_config.get(region, {})
        
        return region_config
    
    def get_evaluation_params(self) -> Dict[str, Any]:
        """
        Get evaluation parameters.
        
        Returns:
            Dictionary of evaluation parameters
        """
        return self.get('evaluation', default={})
    
    def get_gui_params(self) -> Dict[str, Any]:
        """
        Get GUI parameters.
        
        Returns:
            Dictionary of GUI parameters
        """
        return self.get('gui', default={})
    
    def get_io_params(self) -> Dict[str, Any]:
        """
        Get IO parameters.
        
        Returns:
            Dictionary of IO parameters
        """
        return self.get('io', default={})
    
    def merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration to merge
        """
        self._merge_dicts(self.config, new_config)
    
    def _merge_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
        """
        Recursively merge dictionaries.
        
        Args:
            d1: First dictionary (modified in place)
            d2: Second dictionary (values override d1)
        """
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._merge_dicts(d1[k], v)
            else:
                d1[k] = v 