# import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim=256, input_length=135659520, kld_weight=0.005, kernel_size=5, stride=4, padding=1,
                 encoder_channel_list=None, in_dim_list=None):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        # 如果未提供通道列表，则使用默认值
        if encoder_channel_list is None:
            self.encoder_channel_list = [4, 4, 4, 4, 4, 4, 4, 4, 4]  # 通道数列表
        else:
            self.encoder_channel_list = encoder_channel_list
        # 构建编码器的卷积层序列
        layers = []
        in_channels_list = self.encoder_channel_list[:-1]
        out_channels_list = self.encoder_channel_list[1:]

        for idx, (in_c, out_c, in_dim) in enumerate(zip(in_channels_list, out_channels_list, in_dim_list)):
            layers.append(nn.InstanceNorm1d(in_dim))
            if idx == 0:
                layers.append(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=out_c,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=1
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        in_channels=in_c,
                        out_channels=in_c,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=1
                    )
                )
            layers.append(nn.LeakyReLU())
            layers.append(nn.InstanceNorm1d(in_dim))
            layers.append(
                nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=self.kernel_size, stride=self.stride,
                          padding=0))
            if idx == len(out_channels_list) - 1:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.LeakyReLU())

        # 移除最后一个激活函数，添加Tanh激活函数
        # layers = layers[:-1]
        # layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        # self.final_feature_length = self.calculate_final_length(input_length)
        self.final_feature_length = in_dim_list[-1]

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
                 decoder_channel_list=None, in_dim_list=None):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride

        # 如果未提供通道列表，则使用与编码器对称的默认值
        if decoder_channel_list is None:
            self.decoder_channel_list = [4, 16, 64, 256, 256, 64, 16, 8, 1]
        else:
            self.decoder_channel_list = decoder_channel_list
        # self.final_feature_length = final_feature_length
        self.final_feature_length = in_dim_list[0]

        # 全连接层，将潜在向量映射回特征空间
        self.fc = nn.Linear(self.latent_dim, self.decoder_channel_list[0] * self.final_feature_length)

        # 构建解码器的卷积转置层序列
        layers = []
        in_channels_list = self.decoder_channel_list[:-1]
        out_channels_list = self.decoder_channel_list[1:]

        for idx, (in_c, out_c, in_dim) in enumerate(zip(in_channels_list, out_channels_list, in_dim_list)):
            layers.append(nn.LeakyReLU())  # ?为什么要在这里加激活函数
            layers.append(nn.InstanceNorm1d(in_dim))
            layers.append(nn.ConvTranspose1d(
                in_channels=in_c,
                out_channels=in_c,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0)
            )
            layers.append(nn.LeakyReLU())
            layers.append(nn.InstanceNorm1d(in_dim))
            layers.append(nn.ConvTranspose1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=self.kernel_size,
                stride=1,
                padding=stride if idx == len(out_channels_list) - 1 else stride - 1)
            )

        # for idx, (in_c, out_c, in_dim) in enumerate(zip(in_channels_list, out_channels_list, in_dim_list)):
        #     # 如果idx ==3 or 4 则 output_padding=1
        #     if idx == 2 or idx == 3:
        #         layers.append(
        #             nn.ConvTranspose1d(
        #                 in_channels=in_c,
        #                 out_channels=out_c,
        #                 kernel_size=self.kernel_size,
        #                 stride=self.stride,
        #                 padding=self.padding,
        #                 output_padding=2  # 根据需要调整 output_padding 以匹配尺寸
        #             )
        #         )
        #     elif idx == 4:
        #         layers.append(
        #             nn.ConvTranspose1d(
        #                 in_channels=in_c,
        #                 out_channels=out_c,
        #                 kernel_size=self.kernel_size,
        #                 stride=self.stride,
        #                 padding=self.padding,
        #                 output_padding=3  # 根据需要调整 output_padding 以匹配尺寸
        #             )
        #         )
        #     else:
        #         layers.append(
        #             nn.ConvTranspose1d(
        #                 in_channels=in_c,
        #                 out_channels=out_c,
        #                 kernel_size=self.kernel_size,
        #                 stride=self.stride,
        #                 padding=self.padding,
        #                 output_padding=1  # 根据需要调整 output_padding 以匹配尺寸
        #             )
        #         )
        # if out_c != 1:

        # layers.append(nn.InstanceNorm1d(out_c))
        # layers.append(nn.LeakyReLU())

        # if idx != len(out_channels_list) - 1:
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        # else:
        #     layers.append(nn.Tanh())
        # if idx == len(out_channels_list) - 1:
        #     layers.append(nn.Conv1d(out_c, 1, kernel_size=3, stride=1, padding=1))
        #     layers.append(nn.Tanh())
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
                 encoder_kernel_size=5, encoder_stride=5, encoder_padding=None,
                 encoder_channel_list=None,
                 decoder_kernel_size=5, decoder_stride=5, decoder_padding=None,
                 decoder_channel_list=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim  # 潜在空间维度
        self.kld_weight = kld_weight  # KL散度的权重系数
        self.encoder_stride = encoder_stride
        self.decoder_stride = decoder_stride

        if encoder_channel_list is None:
            encoder_channel_list = [1, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        if decoder_channel_list is None:
            decoder_channel_list = [4, 16, 64, 128, 256, 256, 64, 16, 8, 1]

        encoder_dim_list = []
        decoder_dim_list = []

        enc_layer_num = len(encoder_channel_list)
        real_input_dim = (
                int(input_length / self.encoder_stride ** enc_layer_num + 1) * self.encoder_stride ** enc_layer_num
        )
        for i in range(len(encoder_channel_list)):
            dim = real_input_dim // encoder_stride ** i
            encoder_dim_list.append(dim)

        for i in range(len(decoder_channel_list)):
            dim = real_input_dim // encoder_stride ** (len(decoder_channel_list) - i)
            decoder_dim_list.append(dim)

        self.real_input_dim = real_input_dim

        # 初始化编码器
        self.encoder = Encoder(
            latent_dim=latent_dim,
            input_length=input_length,
            kld_weight=kld_weight,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            padding=encoder_padding,
            encoder_channel_list=encoder_channel_list,
            in_dim_list=encoder_dim_list
        )

        # 获取编码器输出的特征长度
        self.final_feature_length = self.encoder.final_feature_length

        # 初始化解码器
        self.decoder = Decoder(latent_dim=latent_dim,
                               kernel_size=decoder_kernel_size,
                               stride=decoder_stride,
                               padding=decoder_padding,
                               decoder_channel_list=decoder_channel_list,
                               final_feature_length=self.final_feature_length,
                               in_dim_list=decoder_dim_list
                               )

    def adjust_output(self, output):
        return output[:, :, :self.in_dim].squeeze(1)

    def adjust_input(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)  # add channel dim ([10, 1, 2048])

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )  # 为什么需要补零？？？
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
        # z = z.view(x.shape)
        z = torch.clamp(z, -1, 1)
        recon_x = self.decode(z)  # 解码器生成重建数据
        # recon_x = recon_x.squeeze(1)     # 去掉多余的维度
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
