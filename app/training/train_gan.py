#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for GAN-based MRI to synthetic CT conversion model.
This script handles dataset preparation, generator and discriminator model definition,
adversarial training, and validation.
"""

import os
import time
import datetime
import logging
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import SimpleITK as sitk
from skimage.metrics import structural_similarity

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.utils.config_utils import load_config
from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT
from app.core.preprocessing import preprocess_mri
from app.training.train_cnn import MRICTDataset  # Reuse dataset class from CNN training

# Set up logger
logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """Generator model for MRI to CT conversion using a 3D U-Net architecture."""
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        """
        Initialize the Generator.
        
        Args:
            in_channels: Number of input channels (default: 1)
            out_channels: Number of output channels (default: 1)
            init_features: Number of features in the first layer (default: 32)
        """
        super(Generator, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # Output range [-1, 1]
    
    def forward(self, x):
        """Forward pass."""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.tanh(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=features),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=features),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )


class Discriminator(nn.Module):
    """Discriminator model for MRI to CT GAN."""
    
    def __init__(self, in_channels=2, init_features=32):
        """
        Initialize the Discriminator.
        
        Args:
            in_channels: Number of input channels (MRI + CT = 2)
            init_features: Number of features in the first layer
        """
        super(Discriminator, self).__init__()
        
        features = init_features
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=features, out_channels=features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=features * 2, out_channels=features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=features * 4, out_channels=features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.output = nn.Conv3d(in_channels=features * 8, out_channels=1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, mri, ct):
        """
        Forward pass.
        
        Args:
            mri: MRI image
            ct: CT image
        
        Returns:
            Discriminator output
        """
        # Concatenate MRI and CT along channel dimension
        x = torch.cat((mri, ct), dim=1)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        
        return x


def weights_init(m):
    """Initialize network weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def train_gan(args):
    """Train the GAN model."""
    # Set up logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_gan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting GAN training with arguments: {args}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets and data loaders
    train_dataset = MRICTDataset(
        args.data_dir,
        patch_size=args.patch_size,
        samples_per_volume=args.samples_per_volume,
        mode='train'
    )
    val_dataset = MRICTDataset(
        args.data_dir,
        patch_size=args.patch_size,
        samples_per_volume=args.samples_per_volume // 2,
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Create models
    generator = Generator(in_channels=1, out_channels=1, init_features=args.init_features)
    discriminator = Discriminator(in_channels=2, init_features=args.init_features)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    # Create optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate * 0.5, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Loss functions
    gan_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create directory for model checkpoints
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Labels for real and fake
    real_label = 1.0
    fake_label = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        generator.train()
        discriminator.train()
        
        train_g_loss = 0.0
        train_d_loss = 0.0
        train_mae = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})")
        for batch_idx, (mri, ct) in enumerate(train_progress):
            batch_size = mri.size(0)
            
            # Move data to device
            mri = mri.to(device)
            ct = ct.to(device)
            
            # Prepare real and fake labels
            real_labels = torch.full((batch_size, 1, 4, 4, 4), real_label, device=device)
            fake_labels = torch.full((batch_size, 1, 4, 4, 4), fake_label, device=device)
            
            #------------------------
            # Update Discriminator
            #------------------------
            
            # Train with real data
            optimizer_d.zero_grad()
            
            # Forward pass real data through discriminator
            real_output = discriminator(mri, ct)
            
            # Calculate loss on real data
            d_loss_real = gan_loss(real_output, real_labels)
            
            # Generate fake data
            fake_ct = generator(mri)
            
            # Forward pass fake data through discriminator
            fake_output = discriminator(mri, fake_ct.detach())
            
            # Calculate loss on fake data
            d_loss_fake = gan_loss(fake_output, fake_labels)
            
            # Combined discriminator loss
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            
            # Backward pass and optimize
            d_loss.backward()
            optimizer_d.step()
            
            #------------------------
            # Update Generator
            #------------------------
            
            # Only update generator every few iterations
            if batch_idx % args.n_critic == 0:
                optimizer_g.zero_grad()
                
                # Generate fake data
                fake_ct = generator(mri)
                
                # Forward pass fake data through discriminator
                fake_output = discriminator(mri, fake_ct)
                
                # Calculate adversarial loss
                g_loss_gan = gan_loss(fake_output, real_labels)
                
                # Calculate L1 loss between fake and real CT
                g_loss_l1 = l1_loss(fake_ct, ct) * args.l1_lambda
                
                # Combined generator loss
                g_loss = g_loss_gan + g_loss_l1
                
                # Backward pass and optimize
                g_loss.backward()
                optimizer_g.step()
                
                # Calculate metrics
                train_g_loss += g_loss.item()
            
            # Calculate metrics
            train_d_loss += d_loss.item()
            with torch.no_grad():
                mae = F.l1_loss(fake_ct, ct).item()
                train_mae += mae
            
            # Update progress bar
            train_progress.set_postfix({"D_loss": d_loss.item(), "G_loss": g_loss.item() if batch_idx % args.n_critic == 0 else "N/A", "MAE": mae})
        
        # Calculate average metrics
        train_d_loss /= len(train_loader)
        train_g_loss /= (len(train_loader) // args.n_critic)
        train_mae /= len(train_loader)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train_D", train_d_loss, epoch)
        writer.add_scalar("Loss/train_G", train_g_loss, epoch)
        writer.add_scalar("MAE/train", train_mae, epoch)
        
        # Validation phase
        generator.eval()
        discriminator.eval()
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Validation (Epoch {epoch+1})")
            for batch_idx, (mri, ct) in enumerate(val_progress):
                batch_size = mri.size(0)
                
                # Move data to device
                mri = mri.to(device)
                ct = ct.to(device)
                
                # Prepare real and fake labels
                real_labels = torch.full((batch_size, 1, 4, 4, 4), real_label, device=device)
                fake_labels = torch.full((batch_size, 1, 4, 4, 4), fake_label, device=device)
                
                # Generate fake data
                fake_ct = generator(mri)
                
                # Forward pass real data through discriminator
                real_output = discriminator(mri, ct)
                
                # Calculate loss on real data
                d_loss_real = gan_loss(real_output, real_labels)
                
                # Forward pass fake data through discriminator
                fake_output = discriminator(mri, fake_ct)
                
                # Calculate loss on fake data
                d_loss_fake = gan_loss(fake_output, fake_labels)
                
                # Combined discriminator loss
                d_loss = (d_loss_real + d_loss_fake) * 0.5
                
                # Calculate adversarial loss
                g_loss_gan = gan_loss(fake_output, real_labels)
                
                # Calculate L1 loss between fake and real CT
                g_loss_l1 = l1_loss(fake_ct, ct) * args.l1_lambda
                
                # Combined generator loss
                g_loss = g_loss_gan + g_loss_l1
                
                # Calculate metrics
                val_d_loss += d_loss.item()
                val_g_loss += g_loss.item()
                val_mae += F.l1_loss(fake_ct, ct).item()
                val_mse += F.mse_loss(fake_ct, ct).item()
                
                # Update progress bar
                val_progress.set_postfix({"D_loss": d_loss.item(), "G_loss": g_loss.item()})
                
                # Save a sample image every few epochs
                if batch_idx == 0 and epoch % 5 == 0:
                    # Convert to numpy and scale back (-1, 1) -> HU range
                    mri_np = mri[0, 0].cpu().numpy()
                    ct_np = ((ct[0, 0].cpu().numpy() + 1) / 2) * 3000 - 1000  # Scale from [-1, 1] to [-1000, 2000]
                    fake_ct_np = ((fake_ct[0, 0].cpu().numpy() + 1) / 2) * 3000 - 1000
                    
                    # Save middle slice
                    slice_idx = mri_np.shape[0] // 2
                    
                    # Create figure
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Plot MRI, CT, and predicted CT
                    axes[0].imshow(mri_np[slice_idx], cmap='gray')
                    axes[0].set_title('MRI')
                    axes[0].axis('off')
                    
                    axes[1].imshow(ct_np[slice_idx], cmap='gray', vmin=-1000, vmax=1000)
                    axes[1].set_title('Ground Truth CT')
                    axes[1].axis('off')
                    
                    axes[2].imshow(fake_ct_np[slice_idx], cmap='gray', vmin=-1000, vmax=1000)
                    axes[2].set_title('GAN Predicted CT')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    output_image_dir = Path(args.output_dir) / "images"
                    output_image_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_image_dir / f"gan_epoch_{epoch+1:03d}.png", dpi=150)
                    plt.close()
                    
                    # Add to TensorBoard
                    writer.add_figure(f"GAN Sample Images/epoch_{epoch+1}", fig, epoch)
        
        # Calculate average metrics
        val_d_loss /= len(val_loader)
        val_g_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        val_psnr = 20 * np.log10(2.0 / np.sqrt(val_mse)) if val_mse > 0 else 100.0
        
        # Log to TensorBoard
        writer.add_scalar("Loss/val_D", val_d_loss, epoch)
        writer.add_scalar("Loss/val_G", val_g_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("MSE/val", val_mse, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train D Loss: {train_d_loss:.4f}, Train G Loss: {train_g_loss:.4f}, Train MAE: {train_mae:.4f}, "
            f"Val D Loss: {val_d_loss:.4f}, Val G Loss: {val_g_loss:.4f}, Val MAE: {val_mae:.4f}, Val PSNR: {val_psnr:.4f}"
        )
        
        # Learning rate scheduler
        scheduler_g.step(val_g_loss)
        scheduler_d.step(val_d_loss)
        
        # Save model if validation loss improved
        if val_g_loss < best_val_loss:
            best_val_loss = val_g_loss
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint_path = checkpoint_dir / f"gan_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'val_g_loss': val_g_loss,
                'val_d_loss': val_d_loss,
                'val_mae': val_mae,
                'val_psnr': val_psnr,
            }, checkpoint_path)
            
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
            
            # Save best model
            best_model_path = Path(args.output_dir) / "best_gan_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'val_g_loss': val_g_loss,
                'val_d_loss': val_d_loss,
                'val_mae': val_mae,
                'val_psnr': val_psnr,
            }, best_model_path)
            
            logger.info(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Close TensorBoard writer
    writer.close()
    
    # Save final model
    final_model_path = Path(args.output_dir) / "final_gan_model.pth"
    torch.save({
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'val_g_loss': val_g_loss,
        'val_d_loss': val_d_loss,
        'val_mae': val_mae,
        'val_psnr': val_psnr,
    }, final_model_path)
    
    logger.info(f"Saved final model to {final_model_path}")
    logger.info("GAN training completed.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train GAN model for MRI to CT conversion")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model parameters
    parser.add_argument("--patch_size", type=int, default=64, help="Size of patches to extract")
    parser.add_argument("--samples_per_volume", type=int, default=100, help="Number of samples per volume")
    parser.add_argument("--init_features", type=int, default=32, help="Initial features in GAN")
    parser.add_argument("--l1_lambda", type=float, default=100.0, help="Weight for L1 loss")
    parser.add_argument("--n_critic", type=int, default=1, help="Number of discriminator updates per generator update")
    
    # Other parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_gan(args)


if __name__ == "__main__":
    main() 