
"""
Multi-GPU Training Script for VAE using Accelerate Library

This script trains the provided VAE model on multiple GPUs using the Hugging Face Accelerate library.
It includes mixed-precision training, learning rate scheduling, and logging.

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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
# Import the VAE model from the provided code
from myVAEdesign3 import VAE
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA


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
    log_file_path = './logs/train4.log'
    file_handler = logging.FileHandler(log_file_path)  # 指定日志文件名
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.logger.addHandler(file_handler)
    # 开始记录日志
    logger.info("This is an info message.")


    # 使用 accelerator 提供的 logger
    accelerator.print(f"Using {accelerator.device.type} device")
    accelerator.print("Random seed set to", args.seed)
    logger.info("Starting training...")

    # 初始化用于记录训练和验证损失的列表
    train_loss_list = []
    val_loss_list = []

    # Initialize the VAE model
    model = VAE(
        latent_dim=args.latent_dim,
        input_length=135659520,
        kld_weight=args.kld_weight,
        encoder_channel_list=args.encoder_channels,
        decoder_channel_list=args.decoder_channels
    ).to(device)

    # Prepare datasets and dataloaders
    train_dataset = VAEDataset(args.train_data_dir)
    val_dataset = VAEDataset(args.val_data_dir)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Define optimizer and scheduler
    # TODO: Adam改为AdamW
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-6)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # TODO: 改为使用 transformers 提供的支持warm up的学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # Calculate total steps for the entire training
    accelerator.print(f"Total training steps: {total_steps}")
    logger.info(f"Total training steps: {total_steps}")

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler, logger = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler, logger
    )
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

    #  检查模型参数的 requires_grad 状态
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Warning: Parameter {name} does not require gradient.")

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
                # data.squeeze(1)改为data
                loss, recon_loss, kld_loss = model.module.loss_function(recon_data, data, mu, logvar)
            else:
                loss, recon_loss, kld_loss = model.loss_function(recon_data, data, mu, logvar)

            # with torch.set_grad_enabled(True):
            # Backward pass
            accelerator.backward(loss)

            # 检查梯度是否正常计算
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Warning: No gradient for parameter {name}")
                elif torch.all(param.grad == 0):
                    print(f"Warning: Gradient for parameter {name} is all zeros.")

            optimizer.step()
            # TODO: 改为每一步后都进行学习率调度
            scheduler.step()

            total_loss += loss.item()
            with torch.no_grad():
                output_min = recon_data.min().item()
                output_max = recon_data.max().item()
                logger.info(f"Decoder output range: [{output_min}, {output_max}]")
                accelerator.print(f"\nDecoder output range: [{output_min}, {output_max}]\n")

            # Get current learning rate
            # current_lr = optimizer.param_groups[0]['lr']
            current_lr = scheduler.get_last_lr()[0]

            # Log training progress
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}, "
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
        # scheduler.step()

        # Save checkpoint with epoch
        if (epoch + 1) % args.save_checkpoint_epochs == 0 and args.checkpoint_dir and accelerator.is_main_process:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}')
            accelerator.save_state(checkpoint_path)
            # 保存当前的epoch信息
            training_state = {'epoch': epoch + 1}
            torch.save(training_state, os.path.join(checkpoint_path, 'training_state.pt'))
            logger.info(f"Saved checkpoint at {checkpoint_path}")
            accelerator.print(f"Saved checkpoint at {checkpoint_path}")

        # 记录当前epoch的平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        train_loss_list.append(avg_train_loss)

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
                val_loss_list.append(avg_val_loss)
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
        accelerator.print(f"Model saved to {model_path}")
        logger.info(f"Model saved to {model_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, args.num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve1.png'))
    plt.show()


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
    parser.add_argument('--kld_weight', type=float, default=0.005,
                        help="Weight for the KL divergence loss.")
    # encoder channels，输入一个列表
    parser.add_argument('--encoder_channels', type=int, nargs='+',default=None,
                        help="Number of channels in the encoder layers.")
    # decoder channels，输入一个列表
    parser.add_argument('--decoder_channels', type=int, nargs='+',default=None,
                        help="Number of channels in the decoder layers.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ))
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warm-up steps ratio")

    # Mixed precision
    parser.add_argument('--fp16', action='store_true',
                        help="Use mixed precision training.")



    # Logging and output
    parser.add_argument('--log_interval', type=int, default=1,
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
