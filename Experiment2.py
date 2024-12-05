from core.data.lora_dataset import LoRADataset  # 确保从正确的模块导入LoRADataset
from torch.utils.data import Dataset, DataLoader
from core.model.vae import VAE
import torch
import torch.nn.functional as F
import numpy as np

# 加载数据集
dataset = LoRADataset(
    '/home/ma-user/work/ymx/Neural-Network-Parameter-Diffusion-main/florence2-lora/bus/adapter_model.safetensors')
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def recon_and_kl_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss


def train(epoch, model, data_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        recon_loss, kl_loss = recon_and_kl_loss(recon_batch, data, mu, logvar)
        loss = recon_loss + kl_loss
        loss.backward()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)}"
                  f" ({100. * batch_idx / len(data_loader):.0f}%)]\t"
                  f"Loss: {loss.item() / len(data):.6f}, "
                  f"Recon Loss: {recon_loss.item() / len(data):.6f}, "
                  f"KL Loss: {kl_loss.item() / len(data):.6f}")

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(data_loader.dataset):.6f}, "
          f"Average Recon Loss: {train_recon_loss / len(data_loader.dataset):.6f}, "
          f"Average KL Loss: {train_kl_loss / len(data_loader.dataset):.6f}")


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_size = max(w.numel() for w in dataset.weights)  # 取权重矩阵展平后的最大大小
hidden_size = 512  # 隐藏层大小
latent_size = 128  # 潜在向量大小

model = VAE(input_size, hidden_size, latent_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train
for epoch in range(1, 150):
    train(epoch, model, data_loader, optimizer, device)

with torch.no_grad():
    z = torch.randn(1, latent_size).to(device)  # 从标准正态分布中采样
    generated_matrix = model.decode(z).cpu().numpy()

    # 使用 np.array2string 打印完整的矩阵
    print("Generated LoRA matrix:")
    # print(np.array2string(generated_matrix, threshold=np.inf, floatmode='fixed', precision=8))

