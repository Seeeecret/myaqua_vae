import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


# ===================== 核心模块定义（包含Hessian-aware更新）=====================
class WatermarkEncoder(nn.Module):
    """ 水印信息编码器（含Hessian敏感度参数）"""

    def __init__(self, msg_length=64, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(msg_length, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # 初始化敏感度系数（通过EMA更新）
        self.register_buffer('gamma_mu', torch.ones(latent_dim))  # 使用buffer非梯度参数
        self.register_buffer('gamma_sigma', torch.ones(latent_dim))

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


def compute_diag_hessian(model, x, m):
    """ 计算ELBO损失对mu和logvar的对角Hessian """
    model.zero_grad()

    # 前向计算
    mu, logvar = model.encode(x)
    mu_wm, logvar_wm = model.embed_watermark(mu, logvar, m)
    z = model.reparameterize(mu_wm, logvar_wm)
    x_rec = model.decoder(z)

    # 计算ELBO损失
    recon_loss = F.mse_loss(x_rec, x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar_wm - mu_wm.pow(2) - logvar_wm.exp())
    loss = recon_loss + kl_loss

    # 计算mu的Hessian对角线
    grad_mu = grad(loss, mu_wm, grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)[0]
    hessian_mu = []
    for i in range(mu_wm.size(1)):
        grad_i = grad(grad_mu[:, i].sum(), mu_wm, retain_graph=True)[0][:, i]
        hessian_mu.append(grad_i.mean().detach())  # 取均值作为期望估计

    # 计算logvar的Hessian对角线
    grad_logvar = grad(loss, logvar_wm, grad_outputs=torch.ones_like(loss), create_graph=True, retain_graph=True)[0]
    hessian_logvar = []
    for i in range(logvar_wm.size(1)):
        grad_i = grad(grad_logvar[:, i].sum(), logvar_wm, retain_graph=True)[0][:, i]
        hessian_logvar.append(grad_i.mean().detach())

    return torch.stack(hessian_mu), torch.stack(hessian_logvar)


# ===================== 训练逻辑增强 =====================
class HessianAwareTrainer:
    def __init__(self, latent_dim=256, msg_length=56, beta=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.vae = VAE_Watermark(latent_dim=latent_dim).to(self.device)
        self.D = Discriminator(latent_dim).to(self.device)
        self.wm_decoder = WatermarkDecoder(latent_dim, msg_length).to(self.device)

        # Hessian相关参数
        self.ema_gamma = 0.95  # Hessian估计的指数平滑系数
        self.hessian_update_interval = 10  # 每10步更新一次Hessian
        self.hessian_mu_ema = torch.ones(latent_dim).to(self.device)
        self.hessian_logvar_ema = torch.ones(latent_dim).to(self.device)
        # 初始化EMA缓冲
        self.register_buffer('hessian_mu_ema', torch.ones(latent_dim).to(self.device))
        self.register_buffer('hessian_logvar_ema', torch.ones(latent_dim).to(self.device))

        # 优化器
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=1e-4)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=1e-4)
        self.opt_wm = torch.optim.Adam(self.wm_decoder.parameters(), lr=2e-4)

        # 其他超参数保持不变
        self.beta = beta
        self.lambda_adv = 0.1
        self.lambda_wm = 1.0
        self.lambda_robust = 0.5
        self.step_counter = 0


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


    def update_hessian_ema(self, x, m):
        """ 更新Hessian估计并调整敏感度系数 """
        # 计算当前batch的Hessian
        hessian_mu, hessian_logvar = compute_diag_hessian(self.vae, x, m)

        # EMA更新
        self.hessian_mu_ema = self.ema_gamma * self.hessian_mu_ema + (1 - self.ema_gamma) * hessian_mu
        self.hessian_logvar_ema = self.ema_gamma * self.hessian_logvar_ema + (1 - self.ema_gamma) * hessian_logvar

        # 更新gamma参数（带数值稳定性）
        eps = 1e-6
        new_gamma_mu = 1.0 / torch.sqrt(torch.abs(self.hessian_mu_ema) + eps)
        new_gamma_sigma = 1.0 / torch.sqrt(torch.abs(self.hessian_logvar_ema) + eps)

        # 标准化处理
        new_gamma_mu = new_gamma_mu / new_gamma_mu.mean()
        new_gamma_sigma = new_gamma_sigma / new_gamma_sigma.mean()

        # 更新模型参数
        self.vae.wm_encoder.gamma_mu.copy_(new_gamma_mu)
        self.vae.wm_encoder.gamma_sigma.copy_(new_gamma_sigma)

    def train_step(self, x, m):
        # ===== Hessian更新 =====
        if self.step_counter % self.hessian_update_interval == 0 and self.step_counter > 0:
            with torch.no_grad():  # 不跟踪Hessian计算图的梯度
                self.update_hessian_ema(x, m)

        # ===== 判别器训练 =====
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

        # ===== VAE与水印联合训练 =====
        self.opt_vae.zero_grad()
        self.opt_wm.zero_grad()

        # 重建损失计算
        mu, logvar = self.vae.encode(x)
        mu_wm, logvar_wm = self.vae.embed_watermark(mu, logvar, m)
        z = self.vae.reparameterize(mu_wm, logvar_wm)
        x_rec = self.vae.decoder(z)

        # 计算各项损失
        recon_loss = F.mse_loss(x_rec, x)
        kl_loss = -0.5 * torch.sum(1 + logvar_wm - mu_wm.pow(2) - logvar_wm.exp()).mean()
        wm_pred = self.wm_decoder(z)
        wm_loss = F.binary_cross_entropy(wm_pred, m.float())

        # 对抗损失
        adv_loss = -torch.mean(self.D(z))

        # 鲁棒性损失
        with torch.no_grad():
            z_attacked = self.attack_simulation(z)
        wm_pred_robust = self.wm_decoder(z_attacked)
        robust_loss = F.binary_cross_entropy(wm_pred_robust, m.float())

        # 总损失
        total_loss = (recon_loss +
                      self.beta * kl_loss +
                      self.lambda_wm * wm_loss +
                      self.lambda_adv * adv_loss +
                      self.lambda_robust * robust_loss)

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)  # 梯度裁剪

        # 参数更新
        self.opt_vae.step()
        self.opt_wm.step()

        self.step_counter += 1
        return {
            'loss/recon': recon_loss.item(),
            'loss/kl': kl_loss.item(),
            'loss/wm': wm_loss.item(),
            'loss/adv': adv_loss.item(),
            'loss/robust': robust_loss.item(),
            'hessian/mu': self.hessian_mu_ema.mean().item(),
            'hessian/logvar': self.hessian_logvar_ema.mean().item()
        }


# ===================== 使用示例 =====================
if __name__ == "__main__":
    trainer = HessianAwareTrainer(latent_dim=256, msg_length=64)

    # 模拟数据
    x_demo = torch.rand(32, 3, 64, 64).to(trainer.device)
    m_demo = torch.randint(0, 2, (32, 64)).float().to(trainer.device)

    # 训练循环示例
    for step in range(100):
        losses = trainer.train_step(x_demo, m_demo)
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Hessian(mu): {losses['hessian/mu']:.4f}, Hessian(logvar): {losses['hessian/logvar']:.4f}")
            print(f"  Recon Loss: {losses['loss/recon']:.4f}, WM Loss: {losses['loss/wm']:.4f}")
