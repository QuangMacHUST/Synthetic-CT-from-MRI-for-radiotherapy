"""
Training module for MRI to synthetic CT conversion models.
Includes CNN and GAN model training functionality.
"""

from app.training.train_cnn import MRICTDataset, UNet3D, train_model as train_cnn_model
from app.training.train_gan import Generator, Discriminator, train_gan as train_gan_model

__all__ = [
    'MRICTDataset',
    'UNet3D',
    'train_cnn_model',
    'Generator',
    'Discriminator',
    'train_gan_model'
] 