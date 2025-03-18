#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create clean __init__.py files in the project.
"""

import os

APP_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MRI to synthetic CT conversion for radiotherapy planning."""

__version__ = "1.0.0"
__all__ = ["core", "utils", "visualization"]
'''

CORE_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Core module for MRI to CT conversion."""

__all__ = [
    "preprocessing",
    "segmentation", 
    "conversion",
    "evaluation"
]
'''

PREPROCESSING_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.core.preprocessing.preprocess_mri import preprocess_mri

__all__ = ["preprocess_mri"]
'''

SEGMENTATION_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.core.segmentation.segment_tissues import segment_tissues, TissueSegmentation

__all__ = ["segment_tissues", "TissueSegmentation"]
'''

CONVERSION_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.core.conversion.convert_mri_to_ct import convert_mri_to_ct, AtlasBasedConverter, CNNConverter, GANConverter

__all__ = ["convert_mri_to_ct", "AtlasBasedConverter", "CNNConverter", "GANConverter"]
'''

EVALUATION_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.core.evaluation.evaluate_synthetic_ct import evaluate_synthetic_ct, calculate_mae, calculate_mse, calculate_psnr, calculate_ssim

__all__ = ["evaluate_synthetic_ct", "calculate_mae", "calculate_mse", "calculate_psnr", "calculate_ssim"]
'''

UTILS_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility modules for MRI to CT conversion."""

__all__ = [
    "config_utils",
    "io_utils",
    "logging_utils",
    "visualization",
    "dicom_utils"
]
'''

VISUALIZATION_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualization module for MRI to CT conversion."""

from app.utils.visualization import (
    plot_slice as plot_image_slice,
    plot_comparison,
    generate_evaluation_report as create_visual_report
)

__all__ = [
    "plot_image_slice",
    "plot_comparison",
    "create_visual_report"
]
'''

DEPLOYMENT_INIT = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deployment module for MRI to CT conversion."""

__all__ = ["api_server"]
'''

def write_file(path, content):
    """Write content to a file, ensuring directory exists."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {path}")
    except Exception as e:
        print(f"Error creating {path}: {str(e)}")

def main():
    """Main function to create all __init__.py files."""
    root = os.getcwd()
    
    # App directory
    write_file(os.path.join(root, "app", "__init__.py"), APP_INIT)
    
    # Core module
    write_file(os.path.join(root, "app", "core", "__init__.py"), CORE_INIT)
    write_file(os.path.join(root, "app", "core", "preprocessing", "__init__.py"), PREPROCESSING_INIT)
    write_file(os.path.join(root, "app", "core", "segmentation", "__init__.py"), SEGMENTATION_INIT)
    write_file(os.path.join(root, "app", "core", "conversion", "__init__.py"), CONVERSION_INIT)
    write_file(os.path.join(root, "app", "core", "evaluation", "__init__.py"), EVALUATION_INIT)
    
    # Utils module
    write_file(os.path.join(root, "app", "utils", "__init__.py"), UTILS_INIT)
    
    # Visualization module
    write_file(os.path.join(root, "app", "visualization", "__init__.py"), VISUALIZATION_INIT)
    
    # Deployment module
    write_file(os.path.join(root, "app", "deployment", "__init__.py"), DEPLOYMENT_INIT)
    
    print("All __init__.py files have been created.")

if __name__ == "__main__":
    main() 