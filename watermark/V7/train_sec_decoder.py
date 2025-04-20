import os
import json
import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.append('../')
sys.path.append('../../')

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from utils.models import SecretDecoder

logger = get_logger(__name__)


class WatermarkDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        """
        Args:
            json_path: 标签JSON文件路径
            img_dir: 图像目录路径
            transform: 应用于图像的转换
        """
        self.img_dir = Path(img_dir)
        self.transform = transform

        # 加载标签数据
        with open(json_path, 'r') as f:
            self.labels = json.load(f)

        self.image_paths = list(self.labels.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = self.img_dir / img_name

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 提取标签并转换为张量
        msg = self.labels[img_name]['msg']
        # 将01bit转换为张量
        msg = [int(bit) for bit in msg]
        msg = torch.tensor(msg)
        # label = torch.zeros(len(msg))
        # for i, bit in enumerate(msg):
        #     label[i] = int(bit)
        labels = F.one_hot(msg, num_classes=2).float()

        return image, labels, msg


def binary_cross_entropy_loss(pred, target):
    """
    计算二进制交叉熵损失
    pred: [B, bit_length, 2] 形状的预测结果
    target: [B, bit_length] 形状的目标值
    """
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    for i in range(pred.size(1)):
        total_loss += loss_fn(pred[:, i], target[:, i].long())
    return total_loss / pred.size(1)


def compute_accuracy(pred, target):
    """
    计算预测的准确率
    """
    pred_bits = torch.argmax(pred, dim=2)
    correct = (pred_bits == target).float().mean()
    return correct


def train(args):
    # 初始化Accelerator
    log_dir = args.logging_dir
    os.makedirs(log_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,  # 添加这一行
    )
    #
    # accelerator.init_trackers(
    #     project_name="watermark_decoder",
    #     config=args.__dict__,
    #     init_kwargs={"tensorboard": {"logging_dir": args.logging_dir}}
    # )

    # 设置随机种子
    if args.seed is not None:
        set_seed(args.seed)

    # 设置图像转换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集和数据加载器
    train_dataset = WatermarkDataset(
        json_path=args.label_path,
        img_dir=args.image_dir,
        transform=transform
    )

    if accelerator.is_local_main_process:
        print(f"数据集大小: {len(train_dataset)} 个样本")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 初始化模型
    bit_length = 4  # 根据 "0010" 长度设置为4
    model = SecretDecoder(output_size=bit_length)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 准备训练
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # 设置总步数和进度条
    total_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(
        range(total_steps),
        disable=not accelerator.is_local_main_process,
        desc="训练进度"
    )

    # 用于计算速度的变量
    start_time = time.time()

    # 创建控制台输出的表头
    if accelerator.is_local_main_process:
        print("\n轮次  |  批次  |   损失   |  准确率  |  每批次时间  |  剩余时间")
        print("-------------------------------------------------------------")

    # 训练循环
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        epoch_start_time = time.time()


        for step, (images, labels, msg) in enumerate(train_dataloader):
            step_start_time = time.time()


            with accelerator.accumulate(model):
                # 前向传播
                outputs = model(images)
                loss = F.binary_cross_entropy_with_logits(outputs, labels.cuda())

                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # 更新统计信息
                train_loss += loss.item()
                train_accuracy += compute_accuracy(outputs, msg).item()

            # 更新进度条
            progress_bar.update(1)

            # 计算批次时间
            step_time = time.time() - step_start_time
            steps_done = epoch * len(train_dataloader) + step + 1
            steps_remaining = total_steps - steps_done
            estimated_time_left = steps_remaining * step_time

            # 每隔指定步数在控制台输出
            if accelerator.is_local_main_process and (
                    step % args.console_log_steps == 0 or step == len(train_dataloader) - 1):
                accuracy = compute_accuracy(outputs, msg).item()
                print(
                    f"{epoch:4d}  |  {step:4d}  |  {loss.item():7.4f}  |  {accuracy:7.4f}  |  {step_time:10.2f}s  |  {str(timedelta(seconds=int(estimated_time_left)))}")

            # 每隔一定步数记录和输出
            if step % args.logging_steps == 0:
                logs = {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "accuracy": compute_accuracy(outputs, msg).item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
                accelerator.log(logs, step=step + epoch * len(train_dataloader))

                if accelerator.is_local_main_process:
                    logger.info(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, "
                                f"Accuracy: {compute_accuracy(outputs, msg).item():.4f}")

        # 每个epoch结束后记录平均损失和准确率
        avg_loss = train_loss / len(train_dataloader)
        avg_accuracy = train_accuracy / len(train_dataloader)
        epoch_time = time.time() - epoch_start_time

        if accelerator.is_local_main_process:
            print("\n" + "-" * 60)
            print(f"轮次 {epoch} 完成 - 耗时: {epoch_time:.2f}秒")
            print(f"平均损失: {avg_loss:.4f}, 平均准确率: {avg_accuracy:.4f}")
            print("-" * 60 + "\n")

        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

        # 保存模型
        if (epoch + 1) % args.save_epochs == 0 or epoch == args.num_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)

            accelerator.save(unwrapped_model.state_dict(), os.path.join(save_path, "model.pt"))
            if accelerator.is_local_main_process:
                accelerator.print(f"✅ 模型已保存至 {save_path}")
                logger.info(f"Model saved at {save_path}")

    # 训练结束
    total_time = time.time() - start_time
    if accelerator.is_local_main_process:
        print("\n" + "="*50)
        print(f"训练完成! 总用时: {str(timedelta(seconds=int(total_time)))}")
        print("="*50 + "\n")
    accelerator.end_training()


def set_seed(seed):
    """设置随机种子"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="训练水印提取模型")
    parser.add_argument("--image_dir", type=str, required=True, help="图像文件夹路径")
    parser.add_argument("--label_path", type=str, required=True, help="标签JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="每个设备的批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器的工作进程数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--console_log_steps", type=int, default=5, help="控制台输出步数")
    parser.add_argument("--save_epochs", type=int, default=25, help="保存模型的轮数间隔")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="混合精度类型")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="用于记录实验的平台")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--logging_dir", type=str,
                        default="/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/my_logs/V7",
                        help="TensorBoard日志目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
