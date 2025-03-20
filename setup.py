#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the MRI to synthetic CT conversion package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="synthetic-ct-from-mri",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Generate synthetic CT images from MRI for radiotherapy planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/synthetic-ct-from-mri",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "synthetic_ct=app.cli:main",
            "synthetic_ct_gui=app.gui.run:main",
        ],
    },
) 