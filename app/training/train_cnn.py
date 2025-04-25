#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Training script for CNN-based MRI to synthetic CT conversion model.
This script handles dataset preparation, model definition, training, and validation.
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

# Import từ ConfigManager thay vì load_config
from app.utils.config_utils import ConfigManager
from app.utils.io_utils import load_medical_image, save_medical_image, SyntheticCT
from app.core.preprocessing import preprocess_mri

# Set up logger
logger = logging.getLogger(__name__)


class MRICTDataset(Dataset):
    """Dataset for MRI to CT conversion."""
    
    def __init__(self, data_dir, patch_size=64, samples_per_volume=100, transform=None, mode='train'):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing paired MRI and CT data
            patch_size: Size of the patches to extract (default: 64)
            samples_per_volume: Number of samples to extract per volume (default: 100)
            transform: Transforms to apply to the data
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.transform = transform
        self.mode = mode
        
        # Load the data pairs
        self.data_pairs = []
        self._load_data_pairs()
        
        # Extract patches
        self.patches = []
        if mode == 'train' or mode == 'val':
            self._extract_patches()
    
    def _load_data_pairs(self):
        """Load paired MRI and CT data from directory."""
        logger.info(f"Loading data pairs from {self.data_dir}")
        
        # Get list of subdirectories (each containing a patient's data)
        patient_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Split into train, val, test (70%, 15%, 15%)
        random.shuffle(patient_dirs)
        n_train = int(len(patient_dirs) * 0.7)
        n_val = int(len(patient_dirs) * 0.15)
        
        if self.mode == 'train':
            patient_dirs = patient_dirs[:n_train]
        elif self.mode == 'val':
            patient_dirs = patient_dirs[n_train:n_train+n_val]
        else:  # test
            patient_dirs = patient_dirs[n_train+n_val:]
        
        for patient_dir in patient_dirs:
            # Look for MRI and CT files
            mri_file = None
            ct_file = None
            
            for file in patient_dir.iterdir():
                if file.is_file():
                    if 'mri' in file.stem.lower():
                        mri_file = file
                    elif 'ct' in file.stem.lower():
                        ct_file = file
            
            if mri_file is not None and ct_file is not None:
                self.data_pairs.append((mri_file, ct_file))
        
        logger.info(f"Loaded {len(self.data_pairs)} data pairs for {self.mode}")
    
    def _extract_patches(self):
        """Extract patches from the data pairs."""
        logger.info(f"Extracting patches for {self.mode}")
        
        for mri_file, ct_file in tqdm(self.data_pairs, desc="Extracting patches"):
            # Load images
            mri_img = load_medical_image(mri_file)
            ct_img = load_medical_image(ct_file)
            
            # Convert to numpy arrays
            mri_arr = sitk.GetArrayFromImage(mri_img)
            ct_arr = sitk.GetArrayFromImage(ct_img)
            
            # Extract random patches
            for _ in range(self.samples_per_volume):
                # Get random location for patch
                z = random.randint(0, mri_arr.shape[0] - self.patch_size)
                y = random.randint(0, mri_arr.shape[1] - self.patch_size)
                x = random.randint(0, mri_arr.shape[2] - self.patch_size)
                
                # Extract patches
                mri_patch = mri_arr[z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]
                ct_patch = ct_arr[z:z+self.patch_size, y:y+self.patch_size, x:x+self.patch_size]
                
                # Add channel dimension
                mri_patch = np.expand_dims(mri_patch, axis=0)
                ct_patch = np.expand_dims(ct_patch, axis=0)
                
                # Normalize patches
                mri_patch = (mri_patch - mri_patch.mean()) / (mri_patch.std() + 1e-8)
                ct_patch = (ct_patch - (-1000)) / 2000  # Normalize to [-0.5, 1.0]
                
                # Apply transforms if available
                if self.transform:
                    mri_patch, ct_patch = self.transform(mri_patch, ct_patch)
                
                self.patches.append((mri_patch, ct_patch))
        
        logger.info(f"Extracted {len(self.patches)} patches for {self.mode}")
    
    def __len__(self):
        """Return the number of patches or data pairs."""
        if self.mode == 'train' or self.mode == 'val':
            return len(self.patches)
        else:  # test mode
            return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """Get a patch or data pair."""
        if self.mode == 'train' or self.mode == 'val':
            mri_patch, ct_patch = self.patches[idx]
            return torch.from_numpy(mri_patch.astype(np.float32)), torch.from_numpy(ct_patch.astype(np.float32))
        else:  # test mode
            mri_file, ct_file = self.data_pairs[idx]
            mri_img = load_medical_image(mri_file)
            ct_img = load_medical_image(ct_file)
            
            # Convert to numpy arrays
            mri_arr = sitk.GetArrayFromImage(mri_img)
            ct_arr = sitk.GetArrayFromImage(ct_img)
            
            # Add channel dimension
            mri_arr = np.expand_dims(mri_arr, axis=0)
            ct_arr = np.expand_dims(ct_arr, axis=0)
            
            # Normalize
            mri_arr = (mri_arr - mri_arr.mean()) / (mri_arr.std() + 1e-8)
            ct_arr = (ct_arr - (-1000)) / 2000  # Normalize to [-0.5, 1.0]
            
            return torch.from_numpy(mri_arr.astype(np.float32)), torch.from_numpy(ct_arr.astype(np.float32)), mri_file, ct_file


class UNet3D(nn.Module):
    """3D U-Net for MRI to CT conversion."""
    
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        """
        Initialize the 3D U-Net.
        
        Args:
            in_channels: Number of input channels (default: 1)
            out_channels: Number of output channels (default: 1)
            init_features: Number of features in the first layer (default: 32)
        """
        super(UNet3D, self).__init__()
        
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        """Forward pass."""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)
    
    @staticmethod
    def _block(in_channels, features, name):
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=features),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(num_features=features),
            nn.LeakyReLU(inplace=True),
        )


def train_model(args):
    """Train the CNN model."""
    # Set up logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting training with arguments: {args}")
    
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
    
    # Create model, optimizer, and loss function
    model = UNet3D(in_channels=1, out_channels=1, init_features=args.init_features)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create directory for model checkpoints
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        train_progress = tqdm(train_loader, desc=f"Training (Epoch {epoch+1})")
        for batch_idx, (mri, ct) in enumerate(train_progress):
            # Move data to device
            mri = mri.to(device)
            ct = ct.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(mri)
            
            # Calculate loss
            loss = 0.7 * l1_loss(output, ct) + 0.3 * mse_loss(output, ct)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item()
            with torch.no_grad():
                mae = F.l1_loss(output, ct).item()
                train_mae += mae
            
            # Update progress bar
            train_progress.set_postfix({"loss": loss.item(), "mae": mae})
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("MAE/train", train_mae, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mse = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Validation (Epoch {epoch+1})")
            for batch_idx, (mri, ct) in enumerate(val_progress):
                # Move data to device
                mri = mri.to(device)
                ct = ct.to(device)
                
                # Forward pass
                output = model(mri)
                
                # Calculate loss
                loss = 0.7 * l1_loss(output, ct) + 0.3 * mse_loss(output, ct)
                
                # Calculate metrics
                val_loss += loss.item()
                val_mae += F.l1_loss(output, ct).item()
                val_mse += F.mse_loss(output, ct).item()
                
                # Update progress bar
                val_progress.set_postfix({"loss": loss.item()})
                
                # Save a sample image every few epochs
                if batch_idx == 0 and epoch % 5 == 0:
                    # Convert to numpy and scale back
                    mri_np = mri[0, 0].cpu().numpy()
                    ct_np = ct[0, 0].cpu().numpy() * 2000 - 1000
                    output_np = output[0, 0].cpu().numpy() * 2000 - 1000
                    
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
                    
                    axes[2].imshow(output_np[slice_idx], cmap='gray', vmin=-1000, vmax=1000)
                    axes[2].set_title('Predicted CT')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    output_image_dir = Path(args.output_dir) / "images"
                    output_image_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_image_dir / f"epoch_{epoch+1:03d}.png", dpi=150)
                    plt.close()
                    
                    # Add to TensorBoard
                    writer.add_figure(f"Sample Images/epoch_{epoch+1}", fig, epoch)
        
        # Calculate average metrics
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_mse /= len(val_loader)
        val_psnr = 20 * np.log10(2.0 / np.sqrt(val_mse)) if val_mse > 0 else 100.0
        
        # Log to TensorBoard
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("MSE/val", val_mse, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        
        # Log to console
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val PSNR: {val_psnr:.4f}"
        )
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_psnr': val_psnr,
            }, checkpoint_path)
            
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
            
            # Save best model
            best_model_path = Path(args.output_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
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
    final_model_path = Path(args.output_dir) / "final_model.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_psnr': val_psnr,
    }, final_model_path)
    
    logger.info(f"Saved final model to {final_model_path}")
    logger.info("Training completed.")

    return {
        'model_path': final_model_path,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_psnr': val_psnr,
        'training_time': time.time() - start_time
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train CNN model for MRI to CT conversion")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and logs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model parameters
    parser.add_argument("--patch_size", type=int, default=64, help="Size of patches to extract")
    parser.add_argument("--samples_per_volume", type=int, default=100, help="Number of samples per volume")
    parser.add_argument("--init_features", type=int, default=32, help="Initial features in U-Net")
    
    # Other parameters
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train_model(args)


def train_cnn_model(mri_dir, ct_dir, output_dir, region="head", batch_size=4, epochs=100, 
                   learning_rate=0.001, use_gpu=True, **kwargs):
    """
    Public interface for CNN model training that can be called from the GUI.
    
    Args:
        mri_dir: Directory containing MRI images
        ct_dir: Directory containing CT images
        output_dir: Directory to save model and results
        region: Anatomical region
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        use_gpu: Whether to use GPU
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with results
    """
    # Configure logging for GUI
    logging.info("=== Starting CNN Training ===")
    logging.info(f"MRI Directory: {mri_dir}")
    logging.info(f"CT Directory: {ct_dir}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Region: {region}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Use GPU: {use_gpu}")
    logging.info("-------------------------------")
    
    # Create args object for compatibility with existing train_model function
    class Args:
        pass
    
    args = Args()
    args.data_dir = Path(mri_dir).parent  # Assuming mri_dir and ct_dir are within the same parent dir
    args.output_dir = output_dir
    args.patch_size = kwargs.get('patch_size', 64)
    args.batch_size = batch_size
    args.epochs = epochs
    args.lr = learning_rate
    args.no_cuda = not use_gpu
    args.samples_per_volume = kwargs.get('samples_per_volume', 100)
    args.region = region
    args.pretrained = kwargs.get('pretrained', None)
    args.resume = kwargs.get('resume', False)
    args.seed = kwargs.get('seed', 42)
    args.val_split = kwargs.get('val_split', 0.2)
    args.test_split = kwargs.get('test_split', 0.1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup log file for this training run
    log_file = os.path.join(output_dir, f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        start_time = time.time()
        
        # Call the existing train_model function
        results = train_model(args)
        
        # Add training time to results
        training_time = time.time() - start_time
        results['training_time'] = training_time
        
        logging.info(f"Training completed in {training_time:.2f} seconds")
        logging.info(f"Model saved to: {results['model_path']}")
        
        return results
    
    except Exception as e:
        error_msg = f"Error during CNN model training: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    finally:
        # Clean up the file handler
        logger.removeHandler(file_handler)


if __name__ == "__main__":
    main() 