import glob
import os

import math
import torch
from torch.utils.tensorboard import SummaryWriter

import odvae

os.environ['HF_HOME'] = '/NEW_EDS/JJ_Group/shaoyh/dorin/cache'
if not os.path.isdir(os.environ['HF_HOME']):
    os.makedirs(os.environ['HF_HOME'])
from diffusers import UNet2DConditionModel
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import Dataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

import torch
import os
import myVAEdesign2
# 试着将重建的lora权重加载到odvae模型中

logger = get_logger(__name__)

class ParameterVectorDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths  # 数据文件的路径列表

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # 加载第 idx 个参数向量
        data = torch.load(self.data_paths[idx])
        return data

def main():
    accelerator = Accelerator(mixed_precision='fp16')  # 可选参数：'fp16', 'bf16', 'no'
    # 如果保存模型检查点的目录不存在，则创建
    checkpoint_dir = "./checkpoints/lora_vae_checkpoints"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 如果保存模型日志的目录不存在，则创建
    log_dir = "./logs/lora_vae_logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # 示例数据加载（请替换为您的数据加载逻辑）
    # 生成随机数据作为示例
    batch_size = 2
    dataset_path = "../checkpoints/lora_weights_dataset"

    rand_val_dataset_path = os.path.join(dataset_path, "rand_val")
    rand_test_dataset_path = os.path.join(dataset_path, "rand_test")

    # 随机测试数据文件列表
    rand_test_data_paths = glob.glob(os.path.join(rand_test_dataset_path,"rand_test_normalized_data*.pth"))
    # 创建数据集
    rand_test_data_sets = ParameterVectorDataset(rand_test_data_paths)
    # 创建数据加载器
    rand_test_data_loader = DataLoader(
        rand_test_data_sets, batch_size=batch_size, shuffle=True, num_workers=2)
    # 随机评估数据文件列表
    rand_val_data_paths = glob.glob(os.path.join(rand_val_dataset_path, "rand_val_normalized_data*.pth"))

    # 创建数据集
    rand_val_data_sets = ParameterVectorDataset(rand_val_data_paths)
    # 创建数据加载器
    rand_val_data_loader = DataLoader(rand_val_data_sets, batch_size=batch_size, shuffle=True, num_workers=2)
    # 设置模型参数
    latent_dim = 256
    kld_weight = 0.5
    in_dim = 135659520  # 请确保 in_dim 设置正确

    # 使用 Accelerator 进行多卡训练


    # 初始化模型
    # myVAE_model = odvae.large(in_dim=in_dim, latent_dim=latent_dim, kld_weight=kld_weight)
    myVAE_model = myVAEdesign2.VAE(input_length=in_dim, latent_dim=latent_dim, kld_weight=kld_weight)
    # 设置优化器
    optimizer = torch.optim.Adam(myVAE_model.parameters(), lr=1e-3, weight_decay=2e-6)

    # 初始化最佳验证损失
    best_val_loss = float('inf')



    # 定义早停参数
    early_stopping_patience = 20
    early_stopping_counter = 0

    # 定义训练参数
    num_update_steps_per_epoch = math.ceil(len(rand_test_data_loader))
    num_train_epochs = 50
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    total_batch_size = batch_size * accelerator.num_processes

    # batch_size = 4
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(rand_test_data_sets)}")
    logger.info(f"  Num batches each epoch = {len(rand_test_data_loader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size  = {batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # 定义学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                              factor=0.5, patience=10,
                                                              verbose=True, min_lr=1e-6)

    myVAE_model, optimizer, rand_test_data_loader, lr_scheduler = accelerator.prepare(
        myVAE_model, optimizer, rand_test_data_loader, lr_scheduler
    )
    # avg_loss = 0.0
    # avg_recon_loss = 0.0
    # avg_kld_loss = 0.0
    # global_step = 0
    # first_epoch = 0
    # progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar.set_description("Steps")

    # 开始训练循环
    for epoch in range(num_train_epochs):
        myVAE_model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0

        rand_test_loader_tqdm = tqdm(rand_test_data_loader, desc=f"Epoch {epoch + 1}/{num_train_epochs} - Train", leave=False)

        for batch_idx, batch in enumerate(rand_test_loader_tqdm):
            # x = batch  # x 的形状为 [batch_size, 135659520]
            x = batch.view(batch_size, 1, -1)  # 调整形状为 [batch_size, 1, 135659520]
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = myVAE_model(x)
            loss, recon_loss, kld_loss = myVAE_model.loss_function(recon_batch, batch, mu, logvar)
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()
            if batch_idx % 10 == 0:
                rand_test_loader_tqdm.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Recon Loss": f"{recon_loss.item():.4f}",
                    "KLD Loss": f"{kld_loss.item():.4f}"
                })
            # 释放未使用的显存
            # free_memory()
        avg_loss = total_loss / len(rand_test_data_loader.dataset)
        avg_recon_loss = total_recon_loss / len(rand_test_data_loader.dataset)
        avg_kld_loss = total_kld_loss / len(rand_test_data_loader.dataset)

        print(f"====> Epoch: {epoch + 1} Average loss: {avg_loss:.4f} "
              f"Recon Loss: {avg_recon_loss:.4f} KLD Loss: {avg_kld_loss:.4f}")

        # 保存模型（仅在主进程）
        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(myVAE_model)
            torch.save(unwrapped_model.state_dict(), os.path.join("./output2", f"vae_epoch_{epoch + 1}.pth"))
        # 验证阶段
        # myVAE_model.eval()
        # val_loss = 0.0
        # rand_val_loader_tqdm = tqdm(rand_val_data_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False)
        #
        # with torch.no_grad():
        #     for batch_idx, batch in enumerate(rand_val_loader_tqdm):
        #         x = batch  # x 的形状为 [batch_size, 135659520]
        #         outputs = myVAE_model(x)
        #         loss = outputs['loss']
        #         val_loss += loss.item()
        #
        # avg_val_loss = val_loss / len(rand_val_data_loader)
        # if accelerator.is_main_process:
        #     print(f'Avg Validation Loss: {avg_val_loss:.4f}')
        #     scheduler.step(avg_val_loss)
        #     writer.add_scalar('AvgLoss/Train', avg_test_loss, epoch)
        #     writer.add_scalar('AvgLoss/Validation', avg_val_loss, epoch)
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_test_loss:.4f}')
        #
        # # 记录学习率
        # current_lr = optimizer.param_groups[0]['lr']
        # if accelerator.is_main_process:
        #     writer.add_scalar('Learning_Rate', current_lr, epoch)
        #
        # # 检查验证损失是否降低
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     early_stopping_counter = 0  # 重置早停计数器
        #     # 保存最佳模型
        #     if accelerator.is_main_process:
        #         checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
        #         accelerator.save(myVAE_model.state_dict(), checkpoint_path)
        #         print(f'Best ODVAE_model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}')
        # else:
        #     early_stopping_counter += 1
        #     if early_stopping_counter >= early_stopping_patience:
        #         if accelerator.is_main_process:
        #             print(f'Early stopping at epoch {epoch + 1}')
        #         break
        #
        # # 打印当前 epoch 的训练和验证损失
        # if (epoch + 1) % 1000 == 0 and accelerator.is_main_process:
        #     checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
        #     accelerator.save(myVAE_model.state_dict(), checkpoint_path)
        #     print(f'ODVAE_model checkpoint saved at epoch {epoch + 1}')
        #
        # if accelerator.is_main_process:
        #     writer.close()


if __name__ == '__main__':
    main()
