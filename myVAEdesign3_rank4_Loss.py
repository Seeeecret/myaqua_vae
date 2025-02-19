# import accelerate
import os
from glob import glob

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
import util.utils_eval_alter as utils_eval
from utils.models import MapperNet


# 完全仿照NND的develop分支改造，rank4的都用这个版本
class OneDimVAE(nn.Module):
    def __init__(self, latent_dim, format_data_path, msgdecoder_path, input_length, kernel_size=7,
                 divide_slice_length=4096, kld_weight=0.02
                 , **kwargs):
        super(OneDimVAE, self).__init__()
        d_model = [8, 16, 32, 64, 128, 256, 256, 128, 64, 32, 16, 8]
        self.d_model = d_model
        self.d_latent = latent_dim
        self.kld_weight = kld_weight
        self.divide_slice_length = divide_slice_length
        self.initial_input_length = input_length
        # confirm self.last_length
        input_length = (input_length // divide_slice_length + 1) * divide_slice_length \
            if input_length % divide_slice_length != 0 else input_length
        assert input_length % int(2 ** len(d_model)) == 0, \
            f"Please set divide_slice_length to {int(2 ** len(d_model))}."
        # 新增: 将外部读入的 format_data_dict 暂存在这里
        self.format_data_dict = torch.load(format_data_path)
        self.msgdecoder_path = msgdecoder_path
        self.msg_bits = 4
        # 检查并修正 input_length
        self.adjusted_input_length = input_length
        self.last_length = input_length // int(2 ** len(d_model))

        self.mapper = MapperNet(input_size=4, output_size=4)
        self.mapper.load_state_dict(torch.load(f"/data/Tsinghua/wuzy/rank4_bits4_output_0203/mapper.pt"))
        # Build Encoder
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
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # Build Decoder
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
        # print(input.shape)
        # assert input.shape == [batch_size, num_parameters]
        # input = input[:, None, :]
        # Check input dimensions
        if input.dim() == 2:  # [batch_size, sequence_length]
            input = input[:, None, :]  # Add channel dimension
        elif input.dim() == 3:  # [batch_size, 1, sequence_length]
            pass  # Input shape is already correct

            # 填充序列
        input = self.pad_sequence(input)  # [B, 1, adjusted_input_length]

        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        # result = self.to_decode(z)
        # result = result.view(-1, self.d_model[-1], self.last_length)
        # result = self.decoder(result)
        # result = self.final_layer(result)
        # assert result.shape[1] == 1, f"{result.shape}"
        result = self.to_decode(z)
        # print(f"After to_decode: {result.shape}")
        result = result.view(-1, self.d_model[-1], self.last_length)
        # print(f"After reshape: {result.shape}")
        result = self.decoder(result)
        # print(f"After decoder: {result.shape}")
        result = self.final_layer(result)
        # print(f"After final_layer: {result.shape}")
        return result[:, 0, :]

    def reparameterize(self, mu, log_var, **kwargs):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if kwargs.get("manual_std") is not None:
            std = kwargs.get("manual_std")
        return eps * std + mu

    def encode_decode(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def sample(self, batch=1):
        z = torch.randn((batch, self.d_latent), device=self.device, dtype=torch.float32)
        recons = self.decode(z)
        return recons

    # 重构函数, 用于在训练过程中将重构的数据还原到原始形状, 并加上水印, 简单推理计算loss
    def reconstruct(self, reconstructions, hidinfo="0100", scale=1.03, save_folder=None):
        loss = 0
        resolution = 512
        restored_state_dict = self.inverseNormalization(reconstructions)
        if restored_state_dict is None:
            return None, None

            # 2) 准备 hidinfo
        if hidinfo is None:
            hid_tensor = torch.randint(0, 2, (1, self.msg_bits))
        else:
            hid_tensor = torch.tensor([int(i) for i in hidinfo]).unsqueeze(0)
        hidinfo_float = hid_tensor.float().to(self.device)

        # 3) 用 mapper 得到 mapped_loradiag
        mapped_loradiag = self.mapper(hidinfo_float)  # shape [1,4]
        hidinfo_str = ''.join(map(str, hid_tensor[0].int().tolist()))

        # 4) 仿照 create_watermark_lora, 遍历 restored_state_dict
        c_lora_state_dict = {}
        for key, value in restored_state_dict.items():
            if 'unet' in key:
                # 先看 'attn'/'ff'
                if ('attn' in key) or ('ff' in key):
                    if 'up.weight' in key:
                        # 原样复制
                        c_lora_state_dict[key] = value
                    elif 'down.weight' in key:
                        mid = torch.diag_embed(mapped_loradiag)[0]  # shape [4,4], 仅示例
                        # 你需要确保 value 的形状能被 mid 相乘
                        # e.g. if value.shape = [4, X], 这个就可 mid @ value
                        c_lora_state_dict[key] = mid @ value * scale
                    else:
                        c_lora_state_dict[key] = value
                # 现在看 'proj_in'/'proj_out'
                elif ('proj_in' in key) or ('proj_out' in key):
                    if 'up.weight' in key:
                        c_lora_state_dict[key] = value
                    elif 'down.weight' in key:
                        mid = mapped_loradiag[0]  # shape [4]
                        # 这里要保证 value 的通道维度 =4, 仅作示例
                        # 例如 c_lora_state_dict[key] = value * mid[:, None, None, None] * scale
                        c_lora_state_dict[key] = value * mid[:, None, None, None] * scale
                    else:
                        c_lora_state_dict[key] = value
                else:
                    # unet其它key, 不动
                    c_lora_state_dict[key] = value
            elif 'text_encoder' in key:
                # 不做操作
                c_lora_state_dict[key] = value
            else:
                # 其他情况: raise or ignore
                c_lora_state_dict[key] = value
        # 5) 如果需要存文件
        if save_folder is not None:
            out_dir = os.path.join(save_folder, hidinfo_str)
            os.makedirs(out_dir, exist_ok=True)
            safetensors_file = os.path.join(out_dir, "pytorch_lora_weights.safetensors")
            safetensors.torch.save_file(c_lora_state_dict, safetensors_file)
            print(f"[info] Watermark LoRA saved to {safetensors_file}")
        with open("prompt4train.txt", 'r') as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
        # 待填充，调用推理评估水印识别率的代码
        # 先用 simple_sample 生成图像
        # 生成的图像会保存在 out_dir 下
        watermarked_img_dir = os.path.join("/gpfs/essfs/iat/Tsinghua/shaoyh/wzy/code/myaqua_vae/", "images")
        if not os.path.exists(watermarked_img_dir):
            os.makedirs(watermarked_img_dir)
        # 调用 simple_sample => 生成多张图
        seed_val =2024
        utils_eval.simple_sample(
            model="stable-diffusion-v1-5/stable-diffusion-v1-5",
            sampler="dpms_m",
            prompt=list(prompts),  # 传入一个prompt列表
            output_dir=watermarked_img_dir,
            lora=c_lora_state_dict,  # 刚才保存的LoRA
            lora_scale=1.0,
            negative_prompt=None,
            height=resolution,
            width=resolution,
            seed=[seed_val + i for i in range(len(prompts))],
            num_inference_steps=25,
            guidance_scale=7.5,
            batch_size=1,
            save=True
        )
        # 如果给定解码器和真值, 调用 simple_decode
        img_paths = sorted(glob(os.path.join(watermarked_img_dir, "*.png")))
        if len(img_paths) > 0:
            bitacc, TPR, results = utils_eval.simple_decode(
                bitnum=self.msg_bits,
                msgdecoder_path=self.msgdecoder_path,
                img_paths=img_paths,
                msg_gt=hidinfo,
                resolution=resolution,
                tpr_threshold=1e-6
            )
            loss = 1.0 - (bitacc if bitacc else 0.0)
        return loss

    def inverseNormalization(self, flattened_recons):
        """
        接收VAE重构后的 1D 向量.
        根据 self.format_data_dict 提供的 mean/std/shape/length 信息,
        对其进行 "逆归一化 + reshape", 并返回一个字典 restored_state_dict.

        注意:
          - 如果你只有单条数据( batch_size=1 ), flattened_recons = [length].
          - 如果 batch_size>1, 需要额外设计怎么区分/拆分。这里仅演示单条情形。
        """
        if self.format_data_dict is None:
            print("[warn] self.format_data_dict is None, cannot do inverse normalization.")
            return None

        # 取出所有 key, 并排除 data_dict['data'](若存在)
        data_keys = [k for k in self.format_data_dict.keys() if k != 'data']

        # 如果 flattened_recons 多于1维, 先 squeeze
        # 这里假设只处理 batch_size=1 的情况
        if len(flattened_recons.shape) > 1:
            flattened_recons = flattened_recons.squeeze(0)
            # 现在 flattened_recons 是 [total_length]

        # 收集每个 key 的长度, 用于切分
        lengths = [self.format_data_dict[k]['length'] for k in data_keys]
        total_length = sum(lengths)

        if flattened_recons.shape[0] < total_length:
            print(f"[warn] 你的重建向量长度 {flattened_recons.shape[0]} 小于所需 {total_length}，请检查。")
            print(flattened_recons.shape)
            return None

        # 切分
        split_data = torch.split(flattened_recons, lengths)

        # 开始逐段逆归一化 & reshape
        restored_state_dict = {}
        for i, key in enumerate(data_keys):
            chunk = split_data[i]
            mean = self.format_data_dict[key]['mean']
            std = self.format_data_dict[key]['std']
            shape = self.format_data_dict[key]['shape']

            denormalized_data = chunk * std + mean
            denormalized_data = denormalized_data.reshape(shape)
            restored_state_dict[key] = denormalized_data

        return restored_state_dict

    def forward(self, x, **kwargs):
        recons, input, mu, log_var = self.encode_decode(input=x, **kwargs)
        print("recons shape: ", recons.shape)

        padded_x = self.pad_sequence(x)
        # 如果recons有3个维度，那么就把它压缩到2个维度, 保证和padded_x的维度一致,因为此时recons的第二个维度一般是1
        acc_loss = self.reconstruct(recons)
        if recons is not None and recons.dim() == 3:
            recons = recons.squeeze(1)
        if padded_x.dim() == 3:
            padded_x = padded_x.squeeze(1)

        recons_loss = F.mse_loss(input=recons, target=padded_x, reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss + acc_loss

        return loss, recons_loss, kld_loss

    @property
    def device(self):
        return next(self.parameters()).device
