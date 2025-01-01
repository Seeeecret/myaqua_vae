
"""
Multi-GPU Training Script for VAE using Accelerate Library

This script trains the provided VAE model on multiple GPUs using the Hugging Face Accelerate library.
It includes mixed-precision training, learning rate scheduling, and logging.
改自NND的develop分支的VAE

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
from myVAEdesign3_rank8 import OneDimVAE as VAE
import matplotlib.pyplot as plt
import os
import numpy as np
from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# =============================================================================
# Dataset Definition
# =============================================================================

class VAEDataset(Dataset):
    """
    Custom Dataset for loading normalized VAE training and evaluation data.
    Each data file is a dictionary containing:
      - "flattened": the normalized data as a 1D tensor.
      - "mean": the mean value of the original data.
      - "std": the standard deviation of the original data.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Directory containing the normalized data files.
        """
        self.data_files = glob(os.path.join(data_dir, 'normalized_*.pth'))
        if not self.data_files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Load and return a data sample.
        """
        try:
            data_dict = torch.load(self.data_files[idx])
            if "data" not in data_dict:
                raise ValueError(f"Missing 'data' key in {self.data_files[idx]}")

            # Load the flattened normalized data
            flattened_data = data_dict["data"]

            # Ensure data is in the expected shape for training
            if flattened_data.dim() != 1:
                raise ValueError(f"Flattened data in {self.data_files[idx]} is not 1D")

            # Reshape to (1, length) for 1D convolution and return
            return flattened_data.unsqueeze(0)
        except Exception as e:
            raise ValueError(f"Error loading {self.data_files[idx]}: {e}")


# =============================================================================
# Training Function
# =============================================================================
def print_grad(name):
    def hook(grad):
        # 打印层的名字和对应的梯度
        print(f"Layer: {name}, Gradient mean: {grad.mean()}, Gradient std: {grad.std()}")
    return hook
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
    log_file_path = args.log_dir
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
    train_recon_loss_list = []  # 新增：记录重建损失
    train_kld_loss_list = []  # 新增：记录KLD损失
    # Initialize the VAE model
    model = VAE(
        latent_dim=args.latent_dim,
        input_length=3391488,
        kld_weight=args.kld_weight,
        manual_std=args.manual_std,
    ).to(device)

    # Prepare datasets and dataloaders
    train_dataset = VAEDataset(args.train_data_dir)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    total_steps = len(train_dataloader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Define optimizer and scheduler
    # TODO: Adam改为AdamW
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-6)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # TODO: 改为使用 transformers 提供的支持warm up的学习率调度器
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # TODO: 从warmup的学习率调度器改为余弦退火调度器，仿照NND的develop分支
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.num_epochs * 2,
    )
    # Calculate total steps for the entire training
    accelerator.print(f"Total training steps: {total_steps}")
    logger.info(f"Total training steps: {total_steps}")

    # Prepare everything with accelerator
    # model, optimizer, train_dataloader, val_dataloader, scheduler, logger = accelerator.prepare(
    #     model, optimizer, train_dataloader, val_dataloader, scheduler, logger
    # )
    model, optimizer, train_dataloader, scheduler, logger = accelerator.prepare(
        model, optimizer, train_dataloader , scheduler, logger
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
        total_recon_loss = 0  # 累积重建损失
        total_kld_loss = 0  # 累积KLD损失

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
            loss, recon_loss, kld_loss = model(data)
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()


            total_loss += loss.item()
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            recon_loss = recon_loss.mean()  # 或 recon_loss.sum()
            kld_loss = kld_loss.mean()  # 或 kld_loss.sum()
            total_recon_loss += recon_loss.item()  # 累积重建损失
            total_kld_loss += kld_loss.item()  # 累积KLD损失

            # Log training progress
            if accelerator.is_main_process and batch_idx % args.log_interval == 0:
                accelerator.print(
                    f"Epoch [{epoch + 1}/{args.num_epochs}], "
                    f"Batch [{batch_idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}, "
                    f"Learning Rate: {current_lr:.10f}"
                )
                logger.info(
                    f"Epoch [{epoch+1}/{args.num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Recon Loss: {recon_loss.item():.4f}, "
                    f"KLD Loss: {kld_loss.item():.4f}"
                    f"Learning Rate: {current_lr:.10f}"
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
        avg_recon_loss = total_recon_loss / len(train_dataloader)  # 计算平均重建损失
        avg_kld_loss = total_kld_loss / len(train_dataloader)  # 计算平均KLD损失


        train_loss_list.append(avg_train_loss)
        train_recon_loss_list.append(avg_recon_loss)  # 记录平均重建损失
        train_kld_loss_list.append(avg_kld_loss)  # 记录平均KLD损失

        scheduler.step()  # 每个epoch结束后调用 scheduler.step()，更新学习率

    # Save the final model
    if args.output_dir and accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # model_path = os.path.join(args.output_dir, 'vae_final.pth')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_End')

        accelerator.save_state(checkpoint_path)
        # 下面代码启用后，进度条会卡住，最后报错
        # TODO: 猜测是wait_for_everyone()方法导致的
        # accelerator.wait_for_everyone()
        # unwrapped_model = accelerator.unwrap_model(model)
        # torch.save(unwrapped_model.state_dict(), model_path)
        accelerator.print(f"Model saved to {checkpoint_path}")
        logger.info(f"Model saved to {checkpoint_path}")
    os.makedirs(args.output_dir, exist_ok=True)
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_loss_list, label='Train Loss')
    # plt.plot(range(1, args.num_epochs + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Epochs')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss_curve1.png'))
    plt.show()

    # 绘制recon_loss单独的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_recon_loss_list, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Recon_Loss')
    plt.title('Reconstruction Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'recon_loss_curve.png'))
    plt.show()

    # 绘制kld_loss单独的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_kld_loss_list, label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('KLD_Loss')
    plt.title('KLD Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'kld_loss_curve.png'))
    plt.show()

    # 绘制包含 loss, recon_loss, kld_loss 的图像
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_epochs + 1), train_loss_list, label='Total Loss')
    plt.plot(range(1, args.num_epochs + 1), train_recon_loss_list, label='Reconstruction Loss')
    plt.plot(range(1, args.num_epochs + 1), train_kld_loss_list, label='KLD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Components over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, 'loss_components_curve.png'))  # 保存新的图像
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

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=2e-4,
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
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="Warm-up steps ratio")

    # Mixed precision
    parser.add_argument('--fp16', action='store_true',
                        help="Use mixed precision training.")



    # Logging and output
    parser.add_argument('--log_interval', type=int, default=1,
                        help="How often to log training progress (in batches).")
    # 日志输出目录
    parser.add_argument('--log_dir', type=str, default='./logs/trainlog.log')
    parser.add_argument('--output_dir', type=str, default='./output2/new',
                        help="Directory to save the final model.")
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/lora_vae_checkpoints/',
                        help="Directory to save/load checkpoints.")
    parser.add_argument('--save_checkpoint_epochs', type=int, default=20,
                        help="Save checkpoints every N epochs.")
    parser.add_argument('--resume_from_checkpoint', action='store_true',
                        help="Resume training from the latest checkpoint.")

    parser.add_argument('--manual_std', type=str, default=None)

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
