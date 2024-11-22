import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=256, input_length=54263808, kld_weight=0.005, kernel_size=5, stride=5, padding=1):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数
        self.padding = padding
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.stride = stride

        # 定义编码器的各个层，逐行显式定义
        # 假设输入通道为1，输出通道为4，重复使用默认的通道数列表
        # 共8个卷积块

        # 第一层
        self.norm0 = nn.InstanceNorm1d(1)
        self.conv0 = nn.Conv1d(1, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm1d(4)
        self.leaky_relu0 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu1 = nn.LeakyReLU()

        # 第二层
        self.norm2 = nn.InstanceNorm1d(4)
        self.conv2 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm3 = nn.InstanceNorm1d(4)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu3 = nn.LeakyReLU()

        # 第三层
        self.norm4 = nn.InstanceNorm1d(4)
        self.conv4 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm1d(4)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu5 = nn.LeakyReLU()

        # 第四层
        self.norm6 = nn.InstanceNorm1d(4)
        self.conv6 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm7 = nn.InstanceNorm1d(4)
        self.leaky_relu6 = nn.LeakyReLU()
        self.conv7 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu7 = nn.LeakyReLU()

        # 第五层
        self.norm8 = nn.InstanceNorm1d(4)
        self.conv8 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm9 = nn.InstanceNorm1d(4)
        self.leaky_relu8 = nn.LeakyReLU()
        self.conv9 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu9 = nn.LeakyReLU()

        # 第六层
        self.norm10 = nn.InstanceNorm1d(4)
        self.conv10 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm11 = nn.InstanceNorm1d(4)
        self.leaky_relu10 = nn.LeakyReLU()
        self.conv11 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu11 = nn.LeakyReLU()

        # 第七层
        self.norm12 = nn.InstanceNorm1d(4)
        self.conv12 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm13 = nn.InstanceNorm1d(4)
        self.leaky_relu12 = nn.LeakyReLU()
        self.conv13 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.leaky_relu13 = nn.LeakyReLU()

        # 第八层
        self.norm14 = nn.InstanceNorm1d(4)
        self.conv14 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=1, padding=1)
        self.norm15 = nn.InstanceNorm1d(4)
        self.leaky_relu14 = nn.LeakyReLU()
        self.conv15 = nn.Conv1d(4, 4, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.tanh = nn.Tanh()

        # 计算最后一个卷积层的输出维度，这里假设为556，您可以根据实际情况调整
        self.final_feature_length = 556

        # 全连接层，映射到潜在空间的均值和对数方差
        self.fc_mu = nn.Linear(4 * self.final_feature_length, self.latent_dim)
        self.fc_logvar = nn.Linear(4 * self.final_feature_length, self.latent_dim)

    def forward(self, x):
        """
        前向传播
        """
        # 第一层
        x = self.norm0(x)
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.leaky_relu0(x)
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # 第二层
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.leaky_relu3(x)

        # 第三层
        x = self.norm4(x)
        x = self.conv4(x)
        x = self.norm5(x)
        x = self.leaky_relu4(x)
        x = self.conv5(x)
        x = self.leaky_relu5(x)

        # 第四层
        x = self.norm6(x)
        x = self.conv6(x)
        x = self.norm7(x)
        x = self.leaky_relu6(x)
        x = self.conv7(x)
        x = self.leaky_relu7(x)

        # 第五层
        x = self.norm8(x)
        x = self.conv8(x)
        x = self.norm9(x)
        x = self.leaky_relu8(x)
        x = self.conv9(x)
        x = self.leaky_relu9(x)

        # 第六层
        x = self.norm10(x)
        x = self.conv10(x)
        x = self.norm11(x)
        x = self.leaky_relu10(x)
        x = self.conv11(x)
        x = self.leaky_relu11(x)

        # 第七层
        x = self.norm12(x)
        x = self.conv12(x)
        x = self.norm13(x)
        x = self.leaky_relu12(x)
        x = self.conv13(x)
        x = self.leaky_relu13(x)

        # 第八层
        x = self.norm14(x)
        x = self.conv14(x)
        x = self.norm15(x)
        x = self.leaky_relu14(x)
        x = self.conv15(x)
        x = self.tanh(x)

        # 展平成一维向量
        x = x.view(x.size(0), -1)

        # 计算潜在空间的均值和对数方差
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, final_feature_length, latent_dim=256, kernel_size=5, stride=5, padding=1):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        # 解码器的通道列表，逐行显式定义
        # 默认使用 [128, 64, 64, 32, 32, 16, 16, 4, 1]

        # 计算解码器的初始特征长度
        self.final_feature_length = final_feature_length

        # 全连接层，将潜在向量映射回特征空间
        self.fc = nn.Linear(self.latent_dim, 128 * self.final_feature_length)

        # 第一层
        self.deconv0 = nn.ConvTranspose1d(128, 64, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm0 = nn.InstanceNorm1d(64)
        self.leaky_relu0 = nn.LeakyReLU()
        self.conv0 = nn.Conv1d(64, 64, kernel_size=self.kernel_size, stride=1, padding=self.stride - 1)
        self.norm1 = nn.InstanceNorm1d(64)
        self.leaky_relu1 = nn.LeakyReLU()

        # 第二层
        self.deconv1 = nn.ConvTranspose1d(64, 64, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm2 = nn.InstanceNorm1d(64)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=self.kernel_size, stride=1, padding=self.stride - 1)
        self.norm3 = nn.InstanceNorm1d(32)
        self.leaky_relu3 = nn.LeakyReLU()

        # 第三层
        self.deconv2 = nn.ConvTranspose1d(32, 32, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm4 = nn.InstanceNorm1d(32)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(32, 32, kernel_size=self.kernel_size, stride=1, padding=self.stride - 1)
        self.norm5 = nn.InstanceNorm1d(32)
        self.leaky_relu5 = nn.LeakyReLU()

        # 第四层
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm6 = nn.InstanceNorm1d(16)
        self.leaky_relu6 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(16, 16, kernel_size=self.kernel_size, stride=1, padding=self.stride - 1)
        self.norm7 = nn.InstanceNorm1d(16)
        self.leaky_relu7 = nn.LeakyReLU()

        # 第五层
        self.deconv4 = nn.ConvTranspose1d(16, 16, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm8 = nn.InstanceNorm1d(16)
        self.leaky_relu8 = nn.LeakyReLU()
        self.conv4 = nn.Conv1d(16, 4, kernel_size=self.kernel_size, stride=1, padding=self.stride - 1)
        self.norm9 = nn.InstanceNorm1d(4)
        self.leaky_relu9 = nn.LeakyReLU()

        # 第六层
        self.deconv5 = nn.ConvTranspose1d(4, 1, kernel_size=self.kernel_size, stride=self.stride, padding=0)
        self.norm10 = nn.InstanceNorm1d(1)
        self.leaky_relu10 = nn.LeakyReLU()
        self.conv5 = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=1, padding=self.stride)
        self.norm11 = nn.InstanceNorm1d(1)
        self.leaky_relu11 = nn.LeakyReLU()

    def forward(self, z):
        """
        前向传播
        """
        # 全连接层
        x = self.fc(z)
        x = x.view(-1, 128, self.final_feature_length)

        # 第一层
        x = self.deconv0(x)
        x = self.norm0(x)
        x = self.leaky_relu0(x)
        x = self.conv0(x)
        x = self.norm1(x)
        x = self.leaky_relu1(x)

        # 第二层
        x = self.deconv1(x)
        x = self.norm2(x)
        x = self.leaky_relu2(x)
        x = self.conv1(x)
        x = self.norm3(x)
        x = self.leaky_relu3(x)

        # 第三层
        x = self.deconv2(x)
        x = self.norm4(x)
        x = self.leaky_relu4(x)
        x = self.conv2(x)
        x = self.norm5(x)
        x = self.leaky_relu5(x)

        # 第四层
        x = self.deconv3(x)
        x = self.norm6(x)
        x = self.leaky_relu6(x)
        x = self.conv3(x)
        x = self.norm7(x)
        x = self.leaky_relu7(x)

        # 第五层
        x = self.deconv4(x)
        x = self.norm8(x)
        x = self.leaky_relu8(x)
        x = self.conv4(x)
        x = self.norm9(x)
        x = self.leaky_relu9(x)

        # 第六层
        x = self.deconv5(x)
        x = self.norm10(x)
        x = self.leaky_relu10(x)
        x = self.conv5(x)
        x = self.norm11(x)
        x = self.leaky_relu11(x)

        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=256, input_length=54263808, kld_weight=0.005):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数
        self.input_length = input_length

        # 初始化编码器
        self.encoder = Encoder(latent_dim=latent_dim, input_length=input_length, kld_weight=kld_weight)

        # 获取编码器输出的特征长度
        self.final_feature_length = self.encoder.final_feature_length

        # 初始化解码器
        self.decoder = Decoder(latent_dim=latent_dim, final_feature_length=self.final_feature_length)

    def adjust_output(self, output):
        return output[:, :, :self.input_length].squeeze(1)

    def adjust_input(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)  # 添加通道维度

        # 补零操作，确保输入长度匹配
        real_input_dim = self.encoder.final_feature_length * (self.encoder.stride ** 8)
        padding_length = real_input_dim - self.input_length
        if padding_length > 0:
            input = torch.cat(
                [
                    input,
                    torch.zeros(input.shape[0], 1, padding_length).to(input.device),
                ],
                dim=2,
            )
        return input

    def encode(self, x, **kwargs):
        x = self.adjust_input(x)
        return self.encoder(x, **kwargs)

    def decode(self, x, **kwargs):
        decoded = self.decoder(x, **kwargs)
        return self.adjust_output(decoded)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧，从高斯分布中采样，允许梯度反向传播
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std  # 返回采样结果

    def forward(self, x):
        """
        前向传播过程
        """
        mu, logvar = self.encode(x)  # 编码器输出均值和对数方差
        z = self.reparameterize(mu, logvar)  # 重参数化采样潜在向量
        z = torch.clamp(z, -1, 1)
        recon_x = self.decode(z)  # 解码器生成重建数据
        return recon_x, mu, logvar  # 返回重建数据、均值和对数方差

    def loss_function(self, recon_x, x, mu, logvar):
        """
        计算VAE的损失函数，包括重建损失和KL散度损失
        """
        # 重建损失，使用均方误差（MSE）
        recon_x = recon_x.squeeze(1)
        x = x.squeeze(1)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        # KL散度损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        # 总损失
        loss = recon_loss + self.kld_weight * kld_loss
        return loss, recon_loss, kld_loss

    def sample(self, num_samples=1):
        """
        从潜在空间中采样生成新样本
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).cuda()
            samples = self.decoder(z)
        return samples
