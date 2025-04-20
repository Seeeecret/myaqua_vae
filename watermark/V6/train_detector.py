import torch
from torch import nn, optim
from tqdm import tqdm
import logging
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log")    ]
)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3)
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_metrics(self, preds, labels):
        # 计算准确率、AUC、误码率
        preds = torch.sigmoid(preds)
        acc = ((preds > 0.5) == labels.bool()).float().mean()

        # 误码率（BER）
        ber = (preds.round() != labels).float().mean()

        return {
            'acc': acc.item(),
            'ber': ber.item()
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc="Training")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_metrics = {'acc': 0, 'ber': 0}

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                metrics = self.compute_metrics(outputs, labels)

                for k in metrics:
                    total_metrics[k] += metrics[k]

        for k in total_metrics:
            total_metrics[k] /= len(self.val_loader)

        logger.info(f"Validation Metrics - Acc: {total_metrics['acc']:.4f}, BER: {total_metrics['ber']:.4f}")
        return total_metrics

    def train(self, epochs=50, early_stop=5):
        best_ber = float('inf')
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc: {val_metrics['acc']:.4f}, BER: {val_metrics['ber']:.4f}")

            # 早停机制
            # if val_metrics['ber'] < best_ber:
            #     best_ber = val_metrics['ber']
            #     no_improve = 0
            #     torch.save(self.model.state_dict(), "best_detector.pth")
            #                 logger.info(f"New best model saved with BER: {best_ber:.4f}")
            # else:
            #     no_improve += 1

            # 每隔1000轮保存一次模型
            if (epoch + 1) % 1000 == 0:
                torch.save(self.model.state_dict(), f"detector_epoch{epoch + 1}.pth")
                logger.info(f"Model saved at epoch {epoch + 1}")


            # if no_improve >= early_stop:
            #     print(f"Early stopping at epoch {epoch + 1}")
            logger.info(f"Early stopping at epoch {epoch + 1}")

            #     break

            self.scheduler.step(val_metrics['acc'])


# 完整训练流程
if __name__ == "__main__":
    from watermark_detect import WatermarkDetector
    from dataset import get_dataloaders

    # 初始化检测器
    config = {
        "freq_bands": [[5, 5], [5, 6], [6, 5], [6, 6], [7, 5], [7, 7]],
        "dct_channels": 3,
        "num_bits": 18
    }
    detector = WatermarkDetector(config)

    # 获取数据
    train_loader, val_loader, _ = get_dataloaders("/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V6output/sampleDataset")

    # 开始训练
    trainer = Trainer(detector, train_loader, val_loader)
    trainer.train(epochs=3000)
