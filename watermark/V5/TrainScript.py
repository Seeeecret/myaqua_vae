import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 假设VAE_Watermark和HessianAwareTrainer类已在之前的代码中定义
from WatermarkModuleV5_2 import VAE_Watermark, HessianAwareTrainer



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
        self.data_files = glob.glob(os.path.join(data_dir, 'normalized_*.pth'))
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


class TrainingConfig:
    """ 集中管理训练参数 """

    def __init__(self):
        self.data_dir = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/juliensimon/stable-diffusion-v1-5-pokemon-lora/normalized_data"
        self.batch_size = 10
        self.num_workers = 2
        self.latent_dim = 128  # 与数据维度匹配
        self.msg_length = 32  # 32-bit水印
        self.max_epochs = 2000
        self.lr = 2e-4
        self.checkpoint_dir = "/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/checkpoints/V5test/"
        self.log_dir = "/baai-cwm-nas/algorithm/ziyang.yan/logs"
        self.log_interval = 50
        self.save_interval = 500
        self.input_length = 797184  # 输入数据长度（需与数据匹配）


class WatermarkVAETrainer:
    def __init__(self, config):
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_dir = os.path.join(config.log_dir, "V5test")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir) if self.log_dir else None
        self.global_step = 0
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = VAE_Watermark(
            input_length=config.input_length,
            latent_dim=config.latent_dim,
            msg_length=config.msg_length
        ).to(self.device)

        # 初始化训练器
        self.trainer = HessianAwareTrainer(
            latent_dim=config.latent_dim,
            input_length=config.input_length,
            msg_length=config.msg_length
        )

        # 混合精度训练
        self.scaler = GradScaler()

        # 创建日志目录
        # self.log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))


    def prepare_dataloaders(self):
        """ 准备数据加载器 """
        full_dataset = VAEDataset(self.config.data_dir)

        # 按9:1划分训练验证集
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )

    def generate_watermark(self, batch_size):
        """ 生成随机水印信息 """
        return torch.randint(0, 2, (batch_size, self.config.msg_length)).float().to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(self.train_loader):
            # 数据预处理
            # 打印形状
            # print(data.shape)
            x = data.to(self.device)  # (B, 1, L)

            # 生成水印
            m = self.generate_watermark(x.size(0))

            # 混合精度训练步骤
            with autocast():
                losses = self.trainer.train_step(x, m)

            # losses的内容为:{
            #             'loss/recon': recon_loss.item(),
            #             'loss/kl': kl_loss.item(),
            #             'loss/wm': wm_loss.item(),
            #             'loss/adv': adv_loss.item(),
            #             'loss/robust': robust_loss.item(),
            #             'hessian/mu': self.hessian_mu_ema.mean().item(),
            #             'hessian/logvar': self.hessian_logvar_ema.mean().item()
            #         }


            # 记录损失

            # 日志记录
            if batch_idx % self.config.log_interval == 0:
                self._log_training(epoch, batch_idx, losses)

            # 保存检查点
            if batch_idx % self.config.save_interval == 0:
                self._save_checkpoint(epoch, batch_idx)

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct_bits = 0
        total_bits = 0

        with torch.no_grad():
            for data in self.val_loader:
                x = data.to(self.device)
                m = self.generate_watermark(x.size(0))

                # 前向计算
                x_rec, _, _, z = self.model(x, m)
                m_pred = (self.trainer.wm_decoder(z) > 0.5).float()

                pad_x = self.model.pad_sequence(x)
                pad_x = pad_x.squeeze(1)

                # 计算指标
                val_loss += F.mse_loss(x_rec, pad_x).item()
                correct_bits += (m_pred == m).sum().item()
                total_bits += m.numel()

        return {
            'val_loss': val_loss / len(self.val_loader),
            'bit_accuracy': correct_bits / total_bits
        }

    def _log_training(self, epoch, batch_idx, losses):
        """ 记录训练日志 """
        log_str = (
            f"Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] | "
            f"Loss: {losses['total_loss']:.4f} | "
            f"Recon: {losses['loss/recon']:.4f} | "
            f"KL loss: {losses['loss/kl']:.4f} | "
            f"WM loss: {losses['loss/wm']:.4f} | "
            f"Adv loss: {losses['loss/adv']:.4f} | "
            f"Robust loss: {losses['loss/robust']:.4f}"
        )
        print(log_str)

        # 写入TensorBoard
        if self.writer:
            for key, value in losses.items():
                self.writer.add_scalar(f'train/{key}', value, global_step=self.global_step)
            self.global_step += 1

    def _save_checkpoint(self, epoch, batch_idx):
        """ 保存模型检查点 """
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.trainer.opt_vae.state_dict(),
            'trainer_state': self.trainer.state_dict()
        }

        filename = f"checkpoint_ep{epoch}_batch{batch_idx}.pth"
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, filename))

    def run_training(self):
        # 准备数据
        self.prepare_dataloaders()

        # 训练循环
        best_val_acc = 0.0
        for epoch in range(self.config.max_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            # 保存最佳模型
            if val_metrics['bit_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['bit_accuracy']
                self._save_checkpoint(epoch, 'best')

            # 打印验证结果
            print(f"\nValidation after Epoch {epoch}:")
            print(f"  Reconstruction Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Watermark Accuracy: {val_metrics['bit_accuracy']:.2%}\n")


if __name__ == "__main__":
    # 初始化配置
    config = TrainingConfig()

    # 检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 运行训练
    trainer = WatermarkVAETrainer(config)
    trainer.run_training()
