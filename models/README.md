# Models Directory

This directory contains pre-trained models for the Synthetic CT from MRI for Radiotherapy project.

## Directory Structure

The models directory is organized by method and anatomical region:

```
models/
├── atlas/
│   ├── head_ct_atlas.h5
│   ├── head_segmentation_atlas.h5
│   ├── pelvis_ct_atlas.h5
│   ├── pelvis_segmentation_atlas.h5
│   ├── thorax_ct_atlas.h5
│   └── thorax_segmentation_atlas.h5
├── cnn/
│   ├── head_cnn_model.h5
│   ├── pelvis_cnn_model.h5
│   └── thorax_cnn_model.h5
├── gan/
│   ├── head_gan_discriminator.h5
│   ├── head_gan_generator.h5
│   ├── pelvis_gan_discriminator.h5
│   ├── pelvis_gan_generator.h5
│   ├── thorax_gan_discriminator.h5
│   └── thorax_gan_generator.h5
└── segmentation/
    ├── head_segmentation_model.h5
    ├── pelvis_segmentation_model.h5
    └── thorax_segmentation_model.h5
```

## Models Description

### Atlas-Based Models

Atlas models contain paired MRI and CT images for atlas-based registration and conversion.

- `head_ct_atlas.h5`: Atlas for head region CT conversion
- `head_segmentation_atlas.h5`: Atlas for head region segmentation
- `pelvis_ct_atlas.h5`: Atlas for pelvis region CT conversion
- `pelvis_segmentation_atlas.h5`: Atlas for pelvis region segmentation
- `thorax_ct_atlas.h5`: Atlas for thorax region CT conversion
- `thorax_segmentation_atlas.h5`: Atlas for thorax region segmentation

### CNN-Based Models

CNN models are convolutional neural networks trained to convert MRI to CT.

- `head_cnn_model.h5`: CNN model for head region conversion
- `pelvis_cnn_model.h5`: CNN model for pelvis region conversion
- `thorax_cnn_model.h5`: CNN model for thorax region conversion

### GAN-Based Models

GAN models are generative adversarial networks for realistic synthetic CT generation.

- `head_gan_generator.h5`: GAN generator for head region conversion
- `head_gan_discriminator.h5`: GAN discriminator for head region training
- `pelvis_gan_generator.h5`: GAN generator for pelvis region conversion
- `pelvis_gan_discriminator.h5`: GAN discriminator for pelvis region training
- `thorax_gan_generator.h5`: GAN generator for thorax region conversion
- `thorax_gan_discriminator.h5`: GAN discriminator for thorax region training

### Segmentation Models

Segmentation models are used to segment different tissue types from MRI.

- `head_segmentation_model.h5`: Deep learning model for head region tissue segmentation
- `pelvis_segmentation_model.h5`: Deep learning model for pelvis region tissue segmentation
- `thorax_segmentation_model.h5`: Deep learning model for thorax region tissue segmentation

## Placeholder Models

The models in this directory are placeholder models created for testing the system's functionality. To use real models, replace these files with trained models.

To create placeholder models for testing, use the script:

```bash
python scripts/create_placeholder_models.py
```

## Using the Models

The models are automatically loaded by the conversion system based on the specified method and anatomical region. The system will look for models in this directory structure by default.

Configuration for model paths can be modified in `configs/default_config.yaml`. 