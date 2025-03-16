#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a GAN model for MRI to CT conversion
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils import setup_logging, load_medical_image, load_config
from app.core.preprocessing import preprocess_mri
from app.core.segmentation import segment_tissues


def create_dataset(
    mri_dirs: List[str],
    ct_dirs: List[str],
    output_dir: str,
    region: str = "head",
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    stride: Tuple[int, int, int] = (32, 32, 32),
    max_samples: Optional[int] = None
) -> str:
    """
    Create a dataset of MRI and CT patches for training.

    Args:
        mri_dirs: List of directories containing MRI images.
        ct_dirs: List of directories containing corresponding CT images.
        output_dir: Output directory to save the dataset.
        region: Anatomical region (head, pelvis, thorax).
        patch_size: Size of patches to extract.
        stride: Stride for patch extraction.
        max_samples: Maximum number of samples to extract.

    Returns:
        Path to the output dataset directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mri_patches = []
    ct_patches = []
    segmentation_patches = []
    
    logging.info("Creating dataset...")
    
    # Get config for preprocessing
    config = load_config()
    preproc_params = config.get_preprocessing_params(region)
    
    # Get list of MRI files
    mri_files = []
    for mri_dir in mri_dirs:
        for root, _, files in os.walk(mri_dir):
            for file in files:
                if file.endswith((".nii", ".nii.gz")):
                    mri_files.append(os.path.join(root, file))
    
    # Get list of CT files
    ct_files = []
    for ct_dir in ct_dirs:
        for root, _, files in os.walk(ct_dir):
            for file in files:
                if file.endswith((".nii", ".nii.gz")):
                    ct_files.append(os.path.join(root, file))
    
    # Check if MRI and CT lists have the same length
    if len(mri_files) != len(ct_files):
        raise ValueError("Number of MRI and CT files must match.")
    
    # Process each MRI-CT pair
    for mri_file, ct_file in zip(mri_files, ct_files):
        logging.info(f"Processing {os.path.basename(mri_file)} and {os.path.basename(ct_file)}")
        
        # Load and preprocess MRI
        mri = load_medical_image(mri_file)
        preprocessed_mri = preprocess_mri(mri_file)
        
        # Segment tissues from MRI
        segmented_tissues = segment_tissues(preprocessed_mri, region=region)
        
        # Load CT
        ct = load_medical_image(ct_file)
        
        # Convert to numpy arrays
        mri_array = sitk.GetArrayFromImage(preprocessed_mri)
        seg_array = sitk.GetArrayFromImage(segmented_tissues)
        ct_array = sitk.GetArrayFromImage(ct)
        
        # Extract patches
        for z in range(0, mri_array.shape[0] - patch_size[0] + 1, stride[0]):
            for y in range(0, mri_array.shape[1] - patch_size[1] + 1, stride[1]):
                for x in range(0, mri_array.shape[2] - patch_size[2] + 1, stride[2]):
                    # Extract patches
                    mri_patch = mri_array[
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]
                    
                    seg_patch = seg_array[
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]
                    
                    ct_patch = ct_array[
                        z:z+patch_size[0],
                        y:y+patch_size[1],
                        x:x+patch_size[2]
                    ]
                    
                    # Skip patches with little information
                    if np.std(mri_patch) < 10 or np.std(ct_patch) < 10:
                        continue
                    
                    # Add to lists
                    mri_patches.append(mri_patch)
                    segmentation_patches.append(seg_patch)
                    ct_patches.append(ct_patch)
                    
                    # Check if we have enough samples
                    if max_samples is not None and len(mri_patches) >= max_samples:
                        break
                
                if max_samples is not None and len(mri_patches) >= max_samples:
                    break
            
            if max_samples is not None and len(mri_patches) >= max_samples:
                break
        
        if max_samples is not None and len(mri_patches) >= max_samples:
            break
    
    # Convert to numpy arrays
    mri_patches = np.array(mri_patches)
    segmentation_patches = np.array(segmentation_patches)
    ct_patches = np.array(ct_patches)
    
    # Normalize
    mri_patches = (mri_patches - np.min(mri_patches)) / (np.max(mri_patches) - np.min(mri_patches))
    ct_patches = (ct_patches - np.min(ct_patches)) / (np.max(ct_patches) - np.min(ct_patches))
    
    # One-hot encode segmentation
    num_classes = np.max(segmentation_patches) + 1
    segmentation_one_hot = np.zeros((segmentation_patches.shape[0], *patch_size, num_classes))
    for i in range(num_classes):
        segmentation_one_hot[..., i] = (segmentation_patches == i).astype(np.float32)
    
    # Save datasets
    np.save(os.path.join(output_dir, "mri_patches.npy"), mri_patches)
    np.save(os.path.join(output_dir, "segmentation_patches.npy"), segmentation_one_hot)
    np.save(os.path.join(output_dir, "ct_patches.npy"), ct_patches)
    
    logging.info(f"Dataset created with {len(mri_patches)} samples.")
    
    return output_dir


def build_generator(
    input_shape: Tuple[int, int, int, int],
    segmentation_shape: Tuple[int, int, int, int],
    output_channels: int = 1
) -> Model:
    """
    Build a generator model for MRI to CT conversion.

    Args:
        input_shape: Shape of the input MRI image.
        segmentation_shape: Shape of the segmentation mask.
        output_channels: Number of output channels.

    Returns:
        Generator model.
    """
    # MRI input
    mri_input = layers.Input(shape=input_shape)
    
    # Segmentation input
    segmentation_input = layers.Input(shape=segmentation_shape)
    
    # Concatenate MRI and segmentation
    concat = layers.Concatenate()([mri_input, segmentation_input])
    
    # Encoder
    # Block 1
    conv1 = layers.Conv3D(32, 3, padding="same")(concat)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU(0.2)(conv1)
    conv1 = layers.Conv3D(32, 3, padding="same")(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU(0.2)(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Block 2
    conv2 = layers.Conv3D(64, 3, padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU(0.2)(conv2)
    conv2 = layers.Conv3D(64, 3, padding="same")(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU(0.2)(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Block 3
    conv3 = layers.Conv3D(128, 3, padding="same")(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU(0.2)(conv3)
    conv3 = layers.Conv3D(128, 3, padding="same")(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU(0.2)(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Bottom
    conv4 = layers.Conv3D(256, 3, padding="same")(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU(0.2)(conv4)
    conv4 = layers.Conv3D(256, 3, padding="same")(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU(0.2)(conv4)
    
    # Decoder
    # Block 3
    up3 = layers.UpSampling3D(size=(2, 2, 2))(conv4)
    up3 = layers.Conv3D(128, 2, padding="same")(up3)
    up3 = layers.BatchNormalization()(up3)
    up3 = layers.LeakyReLU(0.2)(up3)
    up3 = layers.Concatenate()([up3, conv3])
    
    conv5 = layers.Conv3D(128, 3, padding="same")(up3)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU(0.2)(conv5)
    conv5 = layers.Conv3D(128, 3, padding="same")(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU(0.2)(conv5)
    
    # Block 2
    up2 = layers.UpSampling3D(size=(2, 2, 2))(conv5)
    up2 = layers.Conv3D(64, 2, padding="same")(up2)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.LeakyReLU(0.2)(up2)
    up2 = layers.Concatenate()([up2, conv2])
    
    conv6 = layers.Conv3D(64, 3, padding="same")(up2)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.LeakyReLU(0.2)(conv6)
    conv6 = layers.Conv3D(64, 3, padding="same")(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.LeakyReLU(0.2)(conv6)
    
    # Block 1
    up1 = layers.UpSampling3D(size=(2, 2, 2))(conv6)
    up1 = layers.Conv3D(32, 2, padding="same")(up1)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.LeakyReLU(0.2)(up1)
    up1 = layers.Concatenate()([up1, conv1])
    
    conv7 = layers.Conv3D(32, 3, padding="same")(up1)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.LeakyReLU(0.2)(conv7)
    conv7 = layers.Conv3D(32, 3, padding="same")(conv7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.LeakyReLU(0.2)(conv7)
    
    # Output
    output = layers.Conv3D(output_channels, 1, activation="tanh", padding="same")(conv7)
    
    return Model(inputs=[mri_input, segmentation_input], outputs=output)


def build_discriminator(
    input_shape: Tuple[int, int, int, int]
) -> Model:
    """
    Build a discriminator model for GAN.

    Args:
        input_shape: Shape of the input image.

    Returns:
        Discriminator model.
    """
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1
    conv1 = layers.Conv3D(32, 4, strides=2, padding="same")(inputs)
    conv1 = layers.LeakyReLU(0.2)(conv1)
    
    # Layer 2
    conv2 = layers.Conv3D(64, 4, strides=2, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU(0.2)(conv2)
    
    # Layer 3
    conv3 = layers.Conv3D(128, 4, strides=2, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU(0.2)(conv3)
    
    # Layer 4
    conv4 = layers.Conv3D(256, 4, padding="same")(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU(0.2)(conv4)
    
    # Output
    output = layers.Conv3D(1, 4, padding="same")(conv4)
    
    return Model(inputs=inputs, outputs=output)


class GANTrainer:
    """
    Class for training a GAN model for MRI to CT conversion.
    """
    
    def __init__(
        self,
        generator: Model,
        discriminator: Model,
        batch_size: int = 4,
        lambda_l1: float = 100.0
    ):
        """
        Initialize the GAN trainer.

        Args:
            generator: Generator model.
            discriminator: Discriminator model.
            batch_size: Batch size for training.
            lambda_l1: Weight for L1 loss.
        """
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.lambda_l1 = lambda_l1
        
        # Set up optimizers
        self.generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Build combined model
        self.combined = self.build_combined_model()
    
    def build_combined_model(self) -> Model:
        """
        Build the combined GAN model.

        Returns:
            Combined model.
        """
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # MRI and segmentation inputs
        mri_input = layers.Input(shape=self.generator.input_shape[0][1:])
        segmentation_input = layers.Input(shape=self.generator.input_shape[1][1:])
        
        # Generate CT from MRI and segmentation
        generated_ct = self.generator([mri_input, segmentation_input])
        
        # Discriminator takes generated CT
        discriminator_output = self.discriminator(generated_ct)
        
        # Combined model (stacked generator and discriminator)
        combined = Model(
            inputs=[mri_input, segmentation_input],
            outputs=[discriminator_output, generated_ct]
        )
        
        # Compile combined model
        combined.compile(
            loss=["binary_crossentropy", "mae"],
            loss_weights=[1, self.lambda_l1],
            optimizer=self.generator_optimizer
        )
        
        return combined
    
    def train(
        self,
        mri_data: np.ndarray,
        segmentation_data: np.ndarray,
        ct_data: np.ndarray,
        epochs: int = 200,
        save_interval: int = 10,
        output_dir: str = "models/gan"
    ) -> Dict[str, list]:
        """
        Train the GAN model.

        Args:
            mri_data: MRI data.
            segmentation_data: Segmentation data.
            ct_data: CT data.
            epochs: Number of epochs to train.
            save_interval: Interval to save models.
            output_dir: Output directory to save models.

        Returns:
            Dictionary of training history.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create real and fake labels
        real_labels = np.ones((self.batch_size, 1, *self.discriminator.output_shape[1:]))
        fake_labels = np.zeros((self.batch_size, 1, *self.discriminator.output_shape[1:]))
        
        # Training history
        history = {
            "d_loss": [],
            "g_loss": [],
            "g_adv_loss": [],
            "g_l1_loss": []
        }
        
        # Training loop
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train discriminator
            idx = np.random.randint(0, mri_data.shape[0], self.batch_size)
            mri_batch = mri_data[idx]
            segmentation_batch = segmentation_data[idx]
            ct_batch = ct_data[idx]
            
            # Generate fake CT
            fake_ct = self.generator.predict([mri_batch, segmentation_batch])
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(ct_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_ct, fake_labels)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            # Train generator
            g_loss = self.combined.train_on_batch(
                [mri_batch, segmentation_batch],
                [real_labels, ct_batch]
            )
            
            # Save history
            history["d_loss"].append(d_loss)
            history["g_loss"].append(g_loss[0])
            history["g_adv_loss"].append(g_loss[1])
            history["g_l1_loss"].append(g_loss[2])
            
            # Print progress
            logging.info(
                f"[D loss: {d_loss:.4f}] [G loss: {g_loss[0]:.4f}, adv: {g_loss[1]:.4f}, L1: {g_loss[2]:.4f}]"
            )
            
            # Save models
            if (epoch + 1) % save_interval == 0:
                self.save_models(output_dir, epoch + 1)
        
        # Save final models
        self.save_models(output_dir)
        
        # Save training history
        self.save_history(history, output_dir)
        
        return history
    
    def save_models(self, output_dir: str, epoch: Optional[int] = None) -> None:
        """
        Save generator and discriminator models.

        Args:
            output_dir: Output directory.
            epoch: Current epoch (for filename).
        """
        if epoch is not None:
            generator_path = os.path.join(output_dir, f"generator_epoch{epoch}.h5")
            discriminator_path = os.path.join(output_dir, f"discriminator_epoch{epoch}.h5")
        else:
            generator_path = os.path.join(output_dir, "generator.h5")
            discriminator_path = os.path.join(output_dir, "discriminator.h5")
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        
        logging.info(f"Models saved to {output_dir}")
    
    def save_history(self, history: Dict[str, list], output_dir: str) -> None:
        """
        Save training history and plot.

        Args:
            history: Training history.
            output_dir: Output directory.
        """
        # Save history as numpy file
        np.save(os.path.join(output_dir, "history.npy"), history)
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(history["d_loss"], label="Discriminator loss")
        plt.plot(history["g_loss"], label="Generator loss")
        plt.legend()
        plt.title("GAN Losses")
        
        plt.subplot(2, 1, 2)
        plt.plot(history["g_adv_loss"], label="Generator adversarial loss")
        plt.plot(history["g_l1_loss"], label="Generator L1 loss")
        plt.legend()
        plt.title("Generator Component Losses")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_history.png"))
        plt.close()


def main():
    """Main function for training GAN model."""
    parser = argparse.ArgumentParser(description="Train GAN model for MRI to CT conversion")
    
    # Dataset creation
    parser.add_argument("--mri_dirs", nargs="+", help="Directories containing MRI images")
    parser.add_argument("--ct_dirs", nargs="+", help="Directories containing CT images")
    parser.add_argument("--dataset_dir", default="data/processed/gan_dataset", help="Output directory for dataset")
    parser.add_argument("--create_dataset", action="store_true", help="Create dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples")
    
    # Model parameters
    parser.add_argument("--region", default="head", choices=["head", "pelvis", "thorax"], help="Anatomical region")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64], help="Patch size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 loss")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval to save models")
    parser.add_argument("--output_dir", default="models/gan", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=logging.INFO)
    
    # Create dataset if requested
    if args.create_dataset:
        if args.mri_dirs is None or args.ct_dirs is None:
            parser.error("--mri_dirs and --ct_dirs are required to create dataset")
        
        create_dataset(
            args.mri_dirs,
            args.ct_dirs,
            args.dataset_dir,
            region=args.region,
            patch_size=tuple(args.patch_size),
            max_samples=args.max_samples
        )
    
    # Load dataset
    try:
        mri_data = np.load(os.path.join(args.dataset_dir, "mri_patches.npy"))
        segmentation_data = np.load(os.path.join(args.dataset_dir, "segmentation_patches.npy"))
        ct_data = np.load(os.path.join(args.dataset_dir, "ct_patches.npy"))
        
        logging.info(f"Loaded dataset with {len(mri_data)} samples")
        logging.info(f"MRI shape: {mri_data.shape}")
        logging.info(f"Segmentation shape: {segmentation_data.shape}")
        logging.info(f"CT shape: {ct_data.shape}")
    except FileNotFoundError:
        parser.error("Dataset not found. Use --create_dataset to create it.")
    
    # Reshape data to add channel dimension if needed
    if len(mri_data.shape) == 4:
        mri_data = np.expand_dims(mri_data, axis=-1)
    
    if len(ct_data.shape) == 4:
        ct_data = np.expand_dims(ct_data, axis=-1)
    
    # Build models
    generator = build_generator(
        mri_data.shape[1:],
        segmentation_data.shape[1:],
        output_channels=ct_data.shape[-1]
    )
    
    discriminator = build_discriminator(ct_data.shape[1:])
    
    # Compile discriminator
    discriminator.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=["accuracy"]
    )
    
    # Create GAN trainer
    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        batch_size=args.batch_size,
        lambda_l1=args.lambda_l1
    )
    
    # Train models
    trainer.train(
        mri_data=mri_data,
        segmentation_data=segmentation_data,
        ct_data=ct_data,
        epochs=args.epochs,
        save_interval=args.save_interval,
        output_dir=os.path.join(args.output_dir, args.region)
    )


if __name__ == "__main__":
    main() 