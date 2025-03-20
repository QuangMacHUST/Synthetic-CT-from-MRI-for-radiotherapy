#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI to CT conversion module - proxy module for backward compatibility.
This module redirects imports to the appropriate submodule.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple

try:
    from app.core.conversion.convert_mri_to_ct import (
        convert_mri_to_ct,
        AtlasBasedConverter,
        CNNConverter,
        GANConverter
    )
    
    __all__ = [
        "convert_mri_to_ct",
        "AtlasBasedConverter",
        "CNNConverter",
        "GANConverter"
    ]
    
except ImportError as e:
    logging.warning(f"Could not import from conversion submodule: {e}")
    
    # Provide stub implementations for backward compatibility
    import SimpleITK as sitk
    import numpy as np
    
    def convert_mri_to_ct(mri_image, segmentation=None, model_type='gan', region='head'):
        """Stub implementation that raises NotImplementedError"""
        logging.error("convert_mri_to_ct is not properly implemented")
        raise NotImplementedError("convert_mri_to_ct function not available")
    
    class AtlasBasedConverter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AtlasBasedConverter not available")
    
    class CNNConverter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("CNNConverter not available")
    
    class GANConverter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("GANConverter not available")
            
    __all__ = [
        "convert_mri_to_ct",
        "AtlasBasedConverter",
        "CNNConverter",
        "GANConverter"
    ] 