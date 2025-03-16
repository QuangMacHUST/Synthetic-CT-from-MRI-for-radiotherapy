#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the MRI to synthetic CT conversion package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="synthetic_ct",
    version="0.1.0",
    author="Radiotherapy Team",
    author_email="your.email@example.com",
    description="A package for converting MRI images to synthetic CT images for radiotherapy planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Synthetic-CT-from-MRI-for-radiotherapy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "SimpleITK>=2.1.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "pydicom>=2.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.7.0",
        "PySide6>=6.2.0",
    ],
    entry_points={
        'console_scripts': [
            'synthetic_ct=app.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'synthetic_ct': [
            'configs/*.yaml',
            'models/*.pth',
        ],
    },
) 