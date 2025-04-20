import json
import os
from pathlib import Path
from torchvision.transforms import transforms as T
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class WatermarkDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        """
        :param img_dir: 图像目录
        :param label_file: 水印信息文件（JSON）
        :param transform: 数据增强变换
        """
        with open(os.path.join(img_dir, label_file)) as f:  # 自动拼接路径
            self.labels = json.load(f)

        self.img_paths = list(Path(img_dir).glob("*.png"))
        self.transform = transform or self.default_transform()

    def default_transform(self):
        # 包含攻击模拟的增强
        return T.Compose([
            T.Resize(512),
            T.RandomApply([T.RandomRotation(15)], p=0.3),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1)], p=0.3),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # 提取文件名对应的标签
        filename = Path(img_path).stem  # 自动去除扩展名
        if filename.startswith("watermarked"):
            bits = self.labels[filename]  # 直接通过文件名获取
            label = torch.tensor([int(b) for b in bits], dtype=torch.float32)
        else:
            label = torch.zeros(18, dtype=torch.float32)  # 18位全0

        return self.transform(image), label


# 数据加载示例
def get_dataloaders(data_dir, batch_size=32):
    full_dataset = WatermarkDataset(
        img_dir=data_dir,
        label_file="watermark_labels.json"
    )

    # 8:1:1划分
    train_size = int(0.8 * len(full_dataset))
    val_size = (len(full_dataset) - train_size) // 2
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )
