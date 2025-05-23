
"""
Multi-GPU Training Script for VAE using Accelerate Library

This script trains the provided VAE model on multiple GPUs using the Hugging Face Accelerate library.
It includes mixed-precision training, learning rate scheduling, and logging.

Author: [Dorin Wu]
Date: [2024/10/22]
"""

import os
import argparse
import logging
from glob import glob

from accelerate.utils import set_seed
from tqdm import tqdm
# os.environ['HF_HOME'] = '/NEW_EDS/JJ_Group/shaoyh/dorin/cache'
# if not os.path.isdir(os.environ['HF_HOME']):
#     os.makedirs(os.environ['HF_HOME'])
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
# Import the VAE model from the provided code
from myVAEdesign2 import VAE



# =============================================================================
# Dataset Definition
# =============================================================================

class VAEDataset(Dataset):
    """
    Custom Dataset for loading VAE training and evaluation data.
    Each data file is a 1D tensor of length 135659520.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory containing the data files.
        """
        self.data_files = glob(os.path.join(data_dir, 'normalized_*.pth'))
        if not self.data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")
        # logger.info(f"Found {len(self.data_files)} files in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Load and return a data sample.
        """
        try:
            data = torch.load(self.data_files[idx])
            if data.numel() != 135659520:
                raise ValueError(f"Data size mismatch in {self.data_files[idx]}")
            # Reshape data to (1, length) for 1D convolution
            data = data.unsqueeze(0)
            return data
        except Exception as e:
            # logger.error(f"Error loading {self.data_files[idx]}: {e}")
            raise e

# =============================================================================
# Training Function
# =============================================================================

def train(args):
    """
    Main training loop.
    """
    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no', log_with="all", project_dir=args.output_dir )
    device = accelerator.device
    # 设置随机种子（可选）
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(2024)

    # =============================================================================
    # Setup Logging
    # =============================================================================
    logger = get_logger(__name__, log_level="INFO")
    log_file_path = './logs/train.log'
    # logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # 开始记录日志
    logger.info("This is an info message.")


    # 使用 accelerator 提供的 logger
    accelerator.print(f"Using {accelerator.device.type} device")
    accelerator.print("Random seed set to", args.seed)
    logger.info("Starting training...")


    # Initialize the VAE model
    model = VAE(
        latent_dim=args.latent_dim,
        input_length=135659520,
        kld_weight=args.kld_weight
    ).to(device)

    # Prepare datasets and dataloaders
    train_dataset = VAEDataset(args.train_data_dir)
    val_dataset = VAEDataset(args.val_data_dir)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Define optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    if args.fp16:
        model = model.half()  # 将模型参数转换为 FP16
    # Create checkpoint directory if it doesn't exist
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from_checkpoint and args.checkpoint_dir:
        checkpoints = glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*'))
        if checkpoints:
            # 根据epoch编号排序checkpoint
            checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
            latest_checkpoint = checkpoints[-1]
            accelerator.load_state(latest_checkpoint)
            # 加载训练状态，包括epoch信息
            training_state = torch.load(os.path.join(latest_checkpoint, 'training_state.pt'))
            start_epoch = training_state['epoch']
            logger.info(f"Resumed training from checkpoint {latest_checkpoint} at epoch {start_epoch}")
        else:
            logger.info(f"No checkpoints found in {args.checkpoint_dir}, starting from scratch.")


    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0

        # Use progress bar only in the main process
        if accelerator.is_main_process:
            train_iter = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]")
        else:
            train_iter = train_dataloader

        for batch_idx, data in enumerate(train_iter):
            optimizer.zero_grad()

            # Move data to the appropriate device
            data = data.to(device, non_blocking=True)

            # Forward pass
            recon_data, mu, logvar = model(data)

            # Compute loss
            # loss, recon_loss, kld_loss = model.loss_function(recon_data, data.squeeze(1), mu, logvar)
            if hasattr(model, 'module'):
                loss, recon_loss, kld_loss = model.module.loss_function(recon_data, data.squeeze(1), mu, logvar)
            else:
                loss, recon_loss, kld_loss = model.loss_function(recon_data, data.squeeze(1), mu, logvar)

            total_loss += loss.item()

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            # Log training progress
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f},"
                    f"Learning Rate: {current_lr:.6f}"
                )
                logger.info(
                    f"Epoch [{epoch+1}/{args.num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}"
                    f"Learning Rate: {current_lr:.6f}"
                )
            metrics = {
                'train_loss': loss.item(),
                'train_recon_loss': recon_loss.item(),
                'train_kld_loss': kld_loss.item(),
                'epoch': epoch + 1,
                'batch': batch_idx + 1,
                'Learning Rate': current_lr
            }
            accelerator.log(metrics, step=epoch * len(train_dataloader) + batch_idx)

        # Step the scheduler
        scheduler.step()

        # Save checkpoint with epoch
        if (epoch + 1) % args.save_checkpoint_epochs == 0 and args.checkpoint_dir and accelerator.is_main_process:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}')
            accelerator.save_state(checkpoint_path)
            # 保存当前的epoch信息
            training_state = {'epoch': epoch + 1}
            torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))
            logger.info(f"Saved checkpoint at {checkpoint_path}")
            accelerator.print(f"Saved checkpoint at {checkpoint_path}")


        if epoch % args.validation_epochs == 0:
            accelerator.print(f"Epoch [{epoch+1}/{args.num_epochs}], Total Loss: {total_loss:.4f}")
            logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Total Loss: {total_loss:.4f}")
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_dataloader:
                    data = data.to(device, non_blocking=True)
                    recon_data, mu, logvar = model(data)
                    # loss, _, _ = model.loss_function(recon_data, data.squeeze(1), mu, logvar)
                    if hasattr(model, 'module'):
                        loss, _, _ = model.module.loss_function(recon_data, data.squeeze(1), mu, logvar)
                    else:
                        loss, _, _ = model.loss_function(recon_data, data.squeeze(1), mu, logvar)

                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_dataloader)
                if accelerator.is_main_process:
                    accelerator.print(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
                    logger.info(f"Validation Loss after Epoch {epoch+1}: {avg_val_loss:.4f}")
                    # 记录验证损失
                metrics = {
                    'validation_loss': avg_val_loss,
                    'epoch': epoch + 1
                }
                accelerator.log(metrics, step=(epoch + 1) * len(train_dataloader))

    # Save the final model
    if args.output_dir and accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, 'vae_final.pth')
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-GPU Training Script for VAE")

    # Data parameters
    parser.add_argument('--train_data_dir', type=str, required=True,
                        help="Directory containing training data files.")
    parser.add_argument('--val_data_dir', type=str, required=True,
                        help="Directory containing validation data files.")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--latent_dim', type=int, default=256,
                        help="Dimension of the latent space.")
    parser.add_argument('--kld_weight', type=float, default=0.5,
                        help="Weight for the KL divergence loss.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ))
    # Mixed precision
    parser.add_argument('--fp16', action='store_true',
                        help="Use mixed precision training.")

    # Logging and output
    parser.add_argument('--log_interval', type=int, default=2,
                        help="How often to log training progress (in batches).")
    parser.add_argument('--output_dir', type=str, default='./output2/new',
                        help="Directory to save the final model.")
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/lora_vae_checkpoints/',
                        help="Directory to save/load checkpoints.")
    parser.add_argument('--save_checkpoint_epochs', type=int, default=20,
                        help="Save checkpoints every N epochs.")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="Resume training from the latest checkpoint.")

    # Seed for reproducibility (optional)
    parser.add_argument('--seed', type=int, default=2024,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    return args

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    try:
        args = parse_args()
        train(args)
    except Exception as e:
        raise e
