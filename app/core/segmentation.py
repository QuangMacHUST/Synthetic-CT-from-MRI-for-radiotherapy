 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tissue segmentation module - proxy module for backward compatibility.
This module redirects imports to the appropriate submodule.
"""

import logging

try:
    from app.core.segmentation.segment_tissues import segment_tissues, TissueSegmentation
    
    __all__ = ["segment_tissues", "TissueSegmentation"]
    
except ImportError as e:
    logging.warning(f"Could not import from segmentation submodule: {e}")
    
    # Provide stub implementations for backward compatibility
    import SimpleITK as sitk
    
    def segment_tissues(mri_image, method='auto', region='head', **kwargs):
        """Stub implementation that raises NotImplementedError"""
        logging.error("segment_tissues is not properly implemented")
        raise NotImplementedError("segment_tissues function not available")
    
    class TissueSegmentation:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("TissueSegmentation not available")
    
    __all__ = ["segment_tissues", "TissueSegmentation"]