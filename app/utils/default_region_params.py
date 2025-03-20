#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Default region-specific parameters for MRI to CT conversion.
These parameters provide default values for different anatomical regions.
"""

from typing import Dict, Any

# Default region-specific parameters
DEFAULT_REGION_PARAMS: Dict[str, Dict[str, Any]] = {
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