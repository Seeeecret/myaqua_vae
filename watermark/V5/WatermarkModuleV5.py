import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# ===================== 核心模块定义 =====================
class WatermarkEncoder(nn.Module):
    """ 水印信息编码器 """

    def __init__(self, msg_length=64, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_length, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # 初始化敏感度系数（需后续通过Hessian分析更新）
        self.gamma_mu = nn.Parameter(torch.ones(latent_dim))
        self.gamma_sigma = nn.Parameter(torch.ones(latent_dim))

    def forward(self, m):
        encoded = torch.tanh(self.fc(m.float()))
        return encoded * self.gamma_mu, encoded * self.gamma_sigma


class VAE_Watermark(nn.Module):
    """ 带水印嵌入的VAE主模型 """

    def __init__(self, img_channels=3, latent_dim=256):
        super().__init__()

        # 原始VAE编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU())

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid())

        # 水印相关参数
        self.wm_encoder = WatermarkEncoder(latent_dim=latent_dim)
        self.W_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_sigma = nn.Linear(latent_dim, latent_dim, bias=False)

        # 权重正交初始化
        nn.init.orthogonal_(self.W_mu.weight)
        nn.init.orthogonal_(self.W_sigma.weight)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def embed_watermark(self, mu, logvar, m):
        # 生成水印扰动
        delta_mu, delta_logvar = self.wm_encoder(m)

        # 应用约束后的扰动
        new_mu = mu + self.W_mu(delta_mu)
        new_logvar = logvar + self.W_sigma(delta_logvar)
        return new_mu, new_logvar

    def forward(self, x, m=None):
        mu, logvar = self.encode(x)

        if m is not None:  # 训练时嵌入水印
            mu_wm, logvar_wm = self.embed_watermark(mu, logvar, m)
            z = self.reparameterize(mu_wm, logvar_wm)
        else:  # 测试时不嵌入
            z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar, z


class Discriminator(nn.Module):
    """ Wasserstein判别器 """

    def __init__(self, latent_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1))

    def forward(self, z):
        return self.model(z)


class WatermarkDecoder(nn.Module):
    """ 水印信息解码器 """

    def __init__(self, latent_dim=256, msg_length=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, msg_length),
            nn.Sigmoid())

    def forward(self, z):
        return self.model(z)


# ===================== 损失函数与训练逻辑 =====================
def compute_gradient_penalty(D, real_samples, fake_samples):
    """ Wasserstein GAN梯度惩罚 """
    alpha = torch.rand(real_samples.size(0), 1).to(real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    gradients = grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty.mean()


class Trainer:
    def __init__(self, latent_dim=256, msg_length=64, beta=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.vae = VAE_Watermark(latent_dim=latent_dim).to(self.device)
        self.D = Discriminator(latent_dim).to(self.device)
        self.wm_decoder = WatermarkDecoder(latent_dim, msg_length).to(self.device)

        # 优化器
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=1e-4)
        self.opt_wm = torch.optim.Adam(self.wm_decoder.parameters(), lr=2e-4)

        # 超参数
        self.beta = beta  # KL散度权重
        self.lambda_adv = 0.1  # 对抗损失权重
        self.lambda_wm = 1.0  # 水印损失权重
        self.lambda_robust = 0.5  # 鲁棒性损失权重

    def attack_simulation(self, z):
        """ 潜空间攻击模拟 """
        # 随机选择攻击类型
        if torch.rand(1) < 0.3:  # 高斯噪声
            return z + torch.randn_like(z) * 0.1
        elif torch.rand(1) < 0.3:  # 均匀量化
            return torch.round(z * 255) / 255
        else:  # 随机掩码
            mask = (torch.rand_like(z) > 0.1).float()
            return z * mask

    def train_step(self, x, m):
        # ========== 判别器训练 ==========
        self.opt_D.zero_grad()

        # 原始潜变量
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z_real = self.vae.reparameterize(mu, logvar)

        # 含水印潜变量
        x_rec, mu_wm, logvar_wm, z_fake = self.vae(x, m)

        # 计算判别器损失
        loss_D = -torch.mean(self.D(z_real)) + torch.mean(self.D(z_fake))
        gp = compute_gradient_penalty(self.D, z_real.data, z_fake.data)
        loss_D_total = loss_D + 10.0 * gp
        loss_D_total.backward()
        self.opt_D.step()

        # ========== VAE与水印联合训练 ==========
        self.opt_vae.zero_grad()
        self.opt_wm.zero_grad()

        # 重建损失
        recon_loss = F.mse_loss(x_rec, x, reduction='mean')

        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar_wm - mu_wm.pow(2) - logvar_wm.exp())
        kl_loss = kl_loss.mean()

        # 水印可译性损失
        wm_pred = self.wm_decoder(z_fake)
        wm_loss = F.binary_cross_entropy(wm_pred, m.float())

        # 对抗隐蔽性损失
        adv_loss = -torch.mean(self.D(z_fake))

        # 鲁棒性损失（模拟攻击）
        z_attacked = self.attack_simulation(z_fake.detach())
        wm_pred_robust = self.wm_decoder(z_attacked)
        robust_loss = F.binary_cross_entropy(wm_pred_robust, m.float())

        # 总损失
        total_loss = (recon_loss +
                      self.beta * kl_loss +
                      self.lambda_wm * wm_loss +
                      self.lambda_adv * adv_loss +
                      self.lambda_robust * robust_loss)

        total_loss.backward()
        self.opt_vae.step()
        self.opt_wm.step()

        return {
            'loss/recon': recon_loss.item(),
            'loss/kl': kl_loss.item(),
            'loss/wm': wm_loss.item(),
            'loss/adv': adv_loss.item(),
            'loss/robust': robust_loss.item()
        }


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 参数配置
    latent_dim = 256
    msg_length = 64  # 64-bit水印

    # 初始化
    trainer = Trainer(latent_dim=latent_dim, msg_length=msg_length)

    # 模拟数据（实际应替换为真实数据加载）
    batch_size = 32
    x_demo = torch.rand(batch_size, 3, 64, 64).to(trainer.device)
    m_demo = torch.randint(0, 2, (batch_size, msg_length)).float().to(trainer.device)

    # 训练步骤
    losses = trainer.train_step(x_demo, m_demo)
    print(f"训练损失: {losses}")
