import torch
import torch.nn as nn
import torch.nn.functional as F


class OneDimAE(nn.Module):
    def __init__(self, latent_dim, input_length=1695744, kernel_size=7, divide_slice_length=64, kld_weight=0.005,
                 **kwargs):
        super(OneDimAE, self).__init__()
        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8]
        self.d_model = d_model
        self.d_latent = latent_dim
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length
        self.kld_weight = kld_weight
        # 确认self.last_length
        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
            if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0, \
            f"Please set divide_slice_length to {int(2 ** len(d_model))}."

        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        # 构建编码器
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size // 2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], latent_dim)

        # 构建解码器
        modules = []
        self.to_decode = nn.Linear(latent_dim, self.last_length * d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i + 1], kernel_size, 2, kernel_size // 2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size // 2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size // 2),
        )

    def pad_sequence(self, input_seq):
        """
        在序列末尾添加零以调整长度到 self.adjusted_input_length。
        """
        batch_size, channels, seq_length = input_seq.size()
        if seq_length < self.adjusted_input_length:
            padding_size = self.adjusted_input_length - seq_length
            # 在最后一个维度上填充
            input_seq = F.pad(input_seq, (0, padding_size), "constant", 0)
        elif seq_length > self.adjusted_input_length:
            # 截断多余的部分
            input_seq = input_seq[:, :, :self.adjusted_input_length]
        return input_seq

    def encode(self, input, **kwargs):
        # 检查输入维度
        if input.dim() == 2:  # [batch_size, sequence_length]
            input = input[:, None, :]  # 添加通道维度
        elif input.dim() == 3:  # [batch_size, 1, sequence_length]
            pass  # 输入形状已经正确

        # 填充序列
        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        latent = self.to_latent(result)
        return latent

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result[:, 0, :]

    def encode_decode(self, input, **kwargs):
        z = self.encode(input)
        recons = self.decode(z)
        return recons, input

    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    def forward(self, x, **kwargs):
        recons, input = self.encode_decode(input=x, **kwargs)

        # 填充输入
        padded_x = self.pad_sequence(x)
        # 如果recons有3个维度，那么就把它压缩到2个维度, 保证和padded_x的维度一致,因为此时recons的第二个维度一般是1
        if recons is not None and recons.dim() == 3:
            recons = recons.squeeze(1)
        if padded_x.dim() == 3:
            padded_x = padded_x.squeeze(1)

        kld_loss = torch.tensor([0.0, 0.0, 0.0])  # 仅计算重构损失
        # 仅计算重构损失
        recons_loss = F.mse_loss(input=recons, target=padded_x, reduction='mean')
        loss = recons_loss
        return loss, recons_loss, kld_loss

    @property
    def device(self):
        return next(self.parameters()).device
