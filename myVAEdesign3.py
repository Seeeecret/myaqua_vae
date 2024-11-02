# import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=256, input_length=135659520, kld_weight=0.005, kernel_size=5, stride=4, padding=1,
                 encoder_channel_list=None):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        # 如果未提供通道列表，则使用默认值
        if encoder_channel_list is None:
            self.encoder_channel_list = [1, 2, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128]  # 通道数列表

        # 构建编码器的卷积层序列
        layers = []
        in_channels_list = self.encoder_channel_list[:-1]
        out_channels_list = self.encoder_channel_list[1:]

        for in_c, out_c in zip(in_channels_list, out_channels_list):
            layers.append(
                nn.Conv1d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding
                )
            )
            layers.append(nn.InstanceNorm1d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 移除最后一个激活函数，添加Tanh激活函数
        layers = layers[:-1]
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        self.final_feature_length = self.calculate_final_length(input_length)

        # 全连接层，映射到潜在空间的均值和对数方差
        # forward里会将最后一个卷积层的输出展平，所以这里是 out_channels_list[-1] * self.final_feature_length
        # 而不是只是 self.final_feature_length
        self.fc_mu = nn.Linear(out_channels_list[-1] * self.final_feature_length, self.latent_dim)
        self.fc_logvar = nn.Linear(out_channels_list[-1] * self.final_feature_length, self.latent_dim)

    def calculate_final_length(self, input_length):
        """
        计算经过多次卷积后的输出长度
        """
        length = input_length
        for _ in range(len(self.encoder_channel_list) - 1):
            length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        return length

    def forward(self, x):
        """
        前向传播
        """
        x = self.layers(x)  # 通过卷积层序列
        x = x.view(x.size(0), -1)  # 展平成一维向量
        mu = self.fc_mu(x)  # 计算潜在空间的均值
        logvar = self.fc_logvar(x)  # 计算潜在空间的对数方差
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, final_feature_length, latent_dim=256, kernel_size=5, stride=4, padding=1,
                 decoder_channel_list=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        # 如果未提供通道列表，则使用与编码器对称的默认值
        if decoder_channel_list is None:
            self.decoder_channel_list = [128, 128, 64, 64, 32, 32, 16, 16, 8, 8, 4, 4, 2, 1]

        self.final_feature_length = final_feature_length

        # 全连接层，将潜在向量映射回特征空间
        self.fc = nn.Linear(self.latent_dim, self.decoder_channel_list[0] * self.final_feature_length)

        # 构建解码器的卷积转置层序列
        layers = []
        in_channels_list = self.decoder_channel_list[:-1]
        out_channels_list = self.decoder_channel_list[1:]

        for idx, (in_c, out_c) in enumerate(zip(in_channels_list, out_channels_list)):
            # 如果idx ==3 or 4 则 output_padding=1
            if idx == 2 or idx == 3:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=2  # 根据需要调整 output_padding 以匹配尺寸
                    )
                )
            elif idx == 4:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=3  # 根据需要调整 output_padding 以匹配尺寸
                    )
                )
            else:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=1  # 根据需要调整 output_padding 以匹配尺寸
                    )
                )
            # if out_c != 1:

            layers.append(nn.InstanceNorm1d(out_c))
            layers.append(nn.LeakyReLU())

            # if idx != len(out_channels_list) - 1:
            #     layers.append(nn.LeakyReLU(0.2, inplace=True))
            # else:
            #     layers.append(nn.Tanh())
            if idx == len(out_channels_list) - 1:
                layers.append(nn.Conv1d(out_c, 1, kernel_size=3, stride=1, padding=1))
                layers.append(nn.Tanh())
            # else:
            # layers.append(nn.Tanh())  # 最后一层使用 Tanh 激活函数

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        """
        前向传播
        """
        x = self.fc(z)  # 全连接层
        x = x.view(-1, self.decoder_channel_list[0], self.final_feature_length)  # 重塑为卷积输入形状
        x = self.layers(x)  # 通过卷积转置层序列
        # for idx, layer in enumerate(self.layers):
        #     x = layer(x)
        #     print(f"Decoder Layer {idx}: output shape {x.shape}")

        # 如果输出形状不符合预期在此调整
        # x = x.view
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_length, kld_weight=0.005,
                 encoder_kernel_size=5, encoder_stride=4, encoder_padding=1,
                 encoder_channel_list=None,
                 decoder_kernel_size=5, decoder_stride=4, decoder_padding=1,
                 decoder_channel_list=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数

        # 初始化编码器
        self.encoder = Encoder(
            latent_dim=latent_dim,
            input_length=input_length,
            kld_weight=kld_weight,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_padding,
            encoder_channel_list=encoder_channel_list
        )

        # 获取编码器输出的特征长度
        self.final_feature_length = self.encoder.final_feature_length

        # 初始化解码器
        self.decoder = Decoder(latent_dim=latent_dim, kernel_size=decoder_kernel_size, stride=decoder_stride,
                               padding=decoder_padding, decoder_channel_list=decoder_channel_list,
                               final_feature_length=self.final_feature_length)

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
        mu, logvar = self.encoder(x)  # 编码器输出均值和对数方差
        z = self.reparameterize(mu, logvar)  # 重参数化采样潜在向量
        # z = z.view(x.shape)
        # z = torch.clamp(z, -1, 1)
        recon_x = self.decoder(z)  # 解码器生成重建数据
        # recon_x = recon_x.squeeze(1)     # 去掉多余的维度
        return recon_x, mu, logvar  # 返回重建数据、均值和对数方差

    def loss_function(self, recon_x, x, mu, logvar):
        """
        计算VAE的损失函数，包括重建损失和KL散度损失
        """
        # 重建损失，使用均方误差（MSE）
        recon_x = recon_x.squeeze(1)
        x = x.squeeze(1)
        recon_loss = F.mse_loss(recon_x, x)
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
