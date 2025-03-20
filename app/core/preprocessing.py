#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MRI preprocessing module - proxy module for backward compatibility.
This module redirects imports to the appropriate submodule.
"""

import logging

try:
    from app.core.preprocessing.preprocess_mri import preprocess_mri
    
    __all__ = ["preprocess_mri"]
    
except ImportError as e:
    logging.warning(f"Could not import from preprocessing submodule: {e}")
    
    # Provide stub implementation for backward compatibility
    import SimpleITK as sitk
    
    def preprocess_mri(mri_image, bias_correction=True, denoise=True, normalize=True, **kwargs):
        """Stub implementation that raises NotImplementedError"""
        logging.error("preprocess_mri is not properly implemented")
        raise NotImplementedError("preprocess_mri function not available")
    
    __all__ = ["preprocess_mri"] 