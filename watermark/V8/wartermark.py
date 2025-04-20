# 核心代码逻辑（基于watermark.py改造）
import torch
from diffusers import DDIMInverseScheduler
from scipy.stats import norm, truncnorm
from functools import reduce
import numpy as np
import torch.nn.functional as F
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
# from diffusers import StableDiffusionInversePipeline  # 新版专用管道


class ImageShield:
    def __init__(self, ch_factor=1, hw_factor=4, height=64, width=64, device="cuda"):
        self.ch = ch_factor  # 通道重复因子（Stable Diffusion潜在空间通道数4）
        self.hw = hw_factor  # 空间重复因子（控制水印密度）
        self.height = height  # 潜在空间高度（原图高/8）
        self.width = width  # 潜在空间宽度（原图宽/8）
        self.device = device

        # 计算水印参数
        self.latentlength = 4 * self.height * self.width
        self.marklength = self.latentlength // (self.ch * self.hw ** 2)
        self.threshold = self.ch * self.hw ** 2 // 2  # 多数投票阈值

        # 初始化密钥
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)

    def create_watermark(self):
        # 生成模板比特（TP）
        # TODO：这里可以试着优化成自定义的?
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, self.height // self.hw,
                                              self.width // self.hw]).to(self.device)

        # 扩展模板比特到潜在空间尺寸
        expanded_tp = self.watermark.repeat(1, self.ch, self.hw, self.hw)

        # ChaCha20加密生成水印比特m
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        # m_byte = cipher.encrypt(expanded_tp.flatten().cpu().numpy().tobytes())
        binary_array = expanded_tp.flatten().cpu().numpy().astype(np.uint8)
        packed_bits = np.packbits(binary_array, axis=0)  # 压缩8倍
        m_byte = cipher.encrypt(packed_bits.tobytes())
        # m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        packed_bits = np.frombuffer(m_byte, dtype=np.uint8)
        m_bit = np.unpackbits(packed_bits).astype(np.uint8)  # 解包恢复原始比特流
        self.m = torch.from_numpy(m_bit).reshape(1, 4, self.height, self.width).float()

        # 生成带水印的初始噪声（截断采样）
        z = self._truncated_sampling(m_bit)
        return z.to(self.device)

    def _truncated_sampling(self, m_bit):
        ppf = norm.ppf([0.0, 0.5, 1.0])
        z = np.zeros(self.latentlength)

        # 防御性校验
        unique_bits = np.unique(m_bit)
        invalid_bits = set(unique_bits) - {0, 1}
        if invalid_bits:
            raise ValueError(f"非法bit值存在: {invalid_bits}")

        # 向量化加速
        bits = m_bit.astype(int)
        lower_bounds = np.where(bits == 0, ppf[0], ppf[1])
        upper_bounds = np.where(bits == 0, ppf[1], ppf[2])

        # 批量采样（比循环快100倍）
        z = truncnorm.rvs(lower_bounds, upper_bounds, loc=0, scale=1, size=self.latentlength)

        return torch.from_numpy(z).reshape(1, 4, self.height, self.width).float().cuda()


    def extract_watermark(self, image, pipe):
        """
        完整水印提取流程
        """
        # Step 1: DDIM反演获取噪声
        inverted_noise = self.invert_image(pipe, image)

        # Step 2: 二值化噪声
        inverted_bits = (inverted_noise > 0).int().flatten().cpu().numpy()

        # Step 3: ChaCha20解密
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        decrypted = cipher.decrypt(np.packbits(inverted_bits).tobytes())


        # Step 4: 重组并多数投票
        decrypted_bits = np.unpackbits(np.frombuffer(decrypted, dtype=np.uint8))
        decoded = decrypted_bits.reshape(-1, self.ch * self.hw ** 2)
        watermark = (decoded.mean(axis=1) > 0.5).astype(int)

        # 转换为与原始模板相同的形状
        return watermark.reshape(self.watermark.shape), inverted_noise

    def detect_tamper(self, inverted_noise):
        # 空间分层细化检测（改造自HSTR）
        inverted_bits = (inverted_noise > 0).int()
        CMP = (inverted_bits == self.m.to(inverted_bits.device)).float()

        # 分层空间聚合
        M_ini = CMP.mean(dim=1)  # 通道平均
        refined_mask = self.hierarchical_refinement(M_ini, levels=3)
        return refined_mask

    def hierarchical_refinement(self, mask, levels=3):
        # 分层细化（类似HSTR的空间部分）
        masks = []
        for l in range(levels):
            μ = 2 ** l
            # 分割为μ×μ区域并平均
            pooled = F.avg_pool2d(mask, μ, stride=μ)
            # 上采样回原尺寸
            upsampled = F.interpolate(pooled, scale_factor=μ, mode='nearest')
            masks.append(upsampled)
        return torch.stack(masks).mean(dim=0)

    # 修改invert_image方法，确保使用正确的pipe
    def invert_image(self, pipe, image, num_inversion_steps=25):
        # 确保输入在正确设备
        image = image.to(pipe.device)

        # 编码图像到潜在空间
        with torch.no_grad():
            # 根据设备类型设置autocast
            if 'cuda' in str(pipe.device):
                with torch.autocast(device_type='cuda'):
                    latents = pipe.vae.encode(image).latent_dist.mode() * 0.18215
            else:
                latents = pipe.vae.encode(image).latent_dist.mode() * 0.18215

        # 生成空提示嵌入
        text_inputs = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(pipe.device)

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

        # 扩展维度适配UNet输入
        batch_size = latents.shape[0]
        encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)
        pipe.scheduler.set_timesteps(num_inversion_steps)
        # 反演循环
        inverted_latents = latents.clone()
        for t in pipe.progress_bar(pipe.scheduler.timesteps):
            # 根据设备类型设置autocast
            if 'cuda' in str(pipe.device):
                with torch.autocast(device_type='cuda'):
                    noise_pred = pipe.unet(
                        inverted_latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states
                    ).sample
            else:
                noise_pred = pipe.unet(
                    inverted_latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

            inverted_latents = pipe.scheduler.step(
                noise_pred,
                t,
                inverted_latents,
            ).prev_sample
            del noise_pred
            torch.cuda.empty_cache()

        return inverted_latents

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]

        # 扩展维度适配UNet输入
        batch_size = latents.shape[0]
        encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)

        # 反演循环
        inverted_latents = latents.clone()
        for t in pipe.progress_bar(pipe.scheduler.timesteps):
            # 确保所有计算在正确设备上
            with torch.autocast(pipe.device):
                noise_pred = pipe.unet(
                    inverted_latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

            inverted_latents = pipe.scheduler.step(
                noise_pred,
                t,
                inverted_latents,
            ).prev_sample
            del noise_pred
            torch.cuda.empty_cache()

        return inverted_latents
    # def invert_image(self, pipe, image, num_inversion_steps=50):
    #     # 确保输入在正确设备
    #     image = image.to(pipe.device)
    #
    #     # 编码图像到潜在空间
    #     with torch.no_grad(), torch.autocast("cuda"):
    #         latents = pipe.vae.encode(image).latent_dist.mode() * 0.18215
    #
    #     # 生成空提示嵌入
    #     text_inputs = pipe.tokenizer(
    #         "",
    #         padding="max_length",
    #         max_length=pipe.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt"
    #     ).to(pipe.device)
    #
    #     with torch.no_grad():
    #         # 关键修正：仅使用无条件嵌入
    #         encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0]  # [1, 77, 768]
    #
    #     # 扩展维度适配UNet输入
    #     batch_size = latents.shape[0]
    #     encoder_hidden_states = encoder_hidden_states.repeat(batch_size, 1, 1)  # [batch, 77, 768]
    #
    #     # 配置反演调度器
    #     pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    #     pipe.scheduler.set_timesteps(num_inversion_steps)
    #
    #     # 反演循环
    #     inverted_latents = latents.clone()
    #     for t in pipe.progress_bar(pipe.scheduler.timesteps):
    #         noise_pred = pipe.unet(
    #             inverted_latents,
    #             t,
    #             encoder_hidden_states=encoder_hidden_states  # 直接使用3D张量
    #         ).sample
    #
    #         inverted_latents = pipe.scheduler.step(
    #             noise_pred,
    #             t,
    #             inverted_latents,
    #         ).prev_sample
    #         del noise_pred
    #         torch.cuda.empty_cache()
    #
    #     return inverted_latents
