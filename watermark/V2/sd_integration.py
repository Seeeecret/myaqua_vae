from contextlib import contextmanager
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

class SDIntegration(nn.Module):
    def __init__(self, sd_model, format_data_dict):
        super().__init__()
        self.sd_model = sd_model
        self.format_data_dict = format_data_dict

    def merge_lora(self, lora_dict):
        """将LoRA字典合并到SD模型"""
        # 具体实现取决于SD的API
        merged_model = copy.deepcopy(self.sd_model)
        for name, param in lora_dict.items():
            merged_model.load_state_dict({name: param}, strict=False)
        return merged_model

    def generate(self, prompts):
        """生成图像（需适配具体SD版本）"""
        return self.sd_model.sample(prompts)

    def validate_watermark(self, gen_images, lora_params, vae_model):
        """端到端水印验证工具"""
        # 从LoRA参数提取水印
        w_lora = vae_model.watermark.detect_from_lora(lora_params, vae_model)

        # 从图像提取水印
        w_img = self.img_detector(gen_images)

        # 验证一致性
        match_rate = (w_lora == w_img).float().mean()
        return match_rate > 0.9  # 阈值可调


import torch
import copy
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from typing import Dict, List, Optional


class SDIntegration2:
    def __init__(self,
                 model_name: str = "runwayml/stable-diffusion-v1-5",
                 torch_dtype=torch.float16,
                 device: str = "cuda"):
        """
        初始化Stable Diffusion集成环境
        :param model_name: 模型名称或路径
        :param torch_dtype: 计算精度
        :param device: 运行设备
        """
        # 加载原始模型
        self.base_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        ).to(device)

        # 创建可修改的副本
        self.current_pipeline = copy.deepcopy(self.base_pipeline)
        self.device = device
        self._lora_cache = {}  # LoRA参数缓存

    def merge_lora(self,
                   lora_dict: Dict[str, torch.Tensor],
                   alpha: float = 1.0,
                   layer_patterns: List[str] = ["attn1", "attn2"]):
        """
        动态合并LoRA参数到UNet模型
        :param lora_dict: LoRA参数字典（来自VAE的输出）
        :param alpha: 合并强度系数
        :param layer_patterns: 需要合并的层类型模式
        """
        # 参数验证
        self._validate_lora(lora_dict)

        # 获取UNet状态字典
        unet = self.current_pipeline.unet
        state_dict = unet.state_dict()

        # 遍历所有LoRA参数
        for lora_key in lora_dict.keys():
            # 解析层信息（示例键名：lora.down_blocks.1.attentions.0.proj_in）
            parts = lora_key.split('.')
            layer_type = parts[2]  # 例如：down_blocks

            # 仅处理指定类型的层
            if not any(p in layer_type for p in layer_patterns):
                continue

            # 构造原始参数键名
            base_key = '.'.join(parts[1:])  # 移除lora前缀

            # 分离down/up权重
            if 'lora_down' in lora_key:
                lora_down = lora_dict[lora_key]
                lora_up = lora_dict[lora_key.replace('down', 'up')]

                # 计算delta weights
                delta_w = alpha * (lora_up @ lora_down)

                # 合并到原始参数
                if base_key in state_dict:
                    state_dict[base_key] += delta_w.to(state_dict[base_key].dtype)
                else:
                    raise KeyError(f"参数键 {base_key} 不存在于UNet中")

        # 加载更新后的参数
        unet.load_state_dict(state_dict, strict=False)

    def generate(self,
                 prompt: str,
                 num_images: int = 1,
                 **generation_args) -> torch.Tensor:
        """
        生成图像并保留中间状态
        :param prompt: 生成提示词
        :param num_images: 生成数量
        :return: 生成图像张量 [B, C, H, W]
        """
        # 设置默认生成参数
        default_args = {
            "height": 512,
            "width": 512,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_images_per_prompt": num_images
        }
        default_args.update(generation_args)

        # 执行生成
        with torch.inference_mode():
            outputs = self.current_pipeline(
                prompt=[prompt] * num_images,
                output_type="tensor",
                **default_args
            )

        # 返回归一化的图像张量 [0,1]
        images = outputs.images
        return images.clamp(0, 1)

    def validate_watermark(self,
                           images: torch.Tensor,
                           watermark_detector: torch.nn.Module,
                           threshold: float = 0.8) -> float:
        """
        验证图像中的水印有效性
        :param images: 输入图像 [B, C, H, W]
        :param watermark_detector: 水印检测模型
        :param threshold: 判定阈值
        :return: 验证通过率
        """
        # 预处理图像
        inputs = self._preprocess_images(images)

        # 检测水印
        with torch.no_grad():
            preds = watermark_detector(inputs)

        # 计算通过率
        passed = (preds > threshold).float().mean()
        return passed.item()

    def reset_model(self):
        """重置模型到初始状态"""
        self.current_pipeline = copy.deepcopy(self.base_pipeline)

    def _validate_lora(self, lora_dict: dict):
        """LoRA参数格式验证"""
        required_keys = [
            'lora.down_blocks.0.attentions.0.proj_in.lora_down',
            'lora.down_blocks.0.attentions.0.proj_out.lora_up'
        ]

        for key in required_keys:
            if key not in lora_dict:
                raise ValueError(f"缺失必要LoRA参数键：{key}")

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """图像预处理（尺寸调整、归一化）"""
        if images.shape[2:] != (512, 512):
            images = F.interpolate(images, size=(512, 512), mode='bilinear')
        return images

    @contextmanager
    def temporary_merge(self, lora_dict: dict):
        """
        上下文管理器：临时合并LoRA参数
        用法：
        with sd_integration.temporary_merge(lora_dict):
            images = sd_integration.generate(...)
        """
        original_state = copy.deepcopy(self.current_pipeline.unet.state_dict())
        try:
            self.merge_lora(lora_dict)
            yield
        finally:
            self.current_pipeline.unet.load_state_dict(original_state)

    def save_custom_lora(self,
                         lora_dict: dict,
                         save_path: str,
                         metadata: Optional[dict] = None):
        """
        保存带水印的LoRA参数
        :param lora_dict: LoRA参数字典
        :param save_path: 保存路径
        :param metadata: 元数据（水印信息等）
        """
        torch.save({
            'state_dict': lora_dict,
            'metadata': metadata or {},
            'sd_version': '1.5'
        }, save_path)

class SDIntegration3():
    def __init__(self,
                 model_name: str = "runwayml/stable-diffusion-v1.5",
                 torch_dtype=torch.float16,
                 device: str = "cuda"):
        """
        初始化Stable Diffusion集成环境
        :param model_name: 模型名称或路径
        :param torch_dtype: 计算精度
        :param device: 运行设备
        """
        self.base_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch_dtype
        ).to(device)
        self.current_pipeline = copy.deepcopy(self.base_pipeline)
        self.device = device
        self.torch_dtype = torch_dtype
        self._reverse_key_map = self._build_reverse_key_map()
    def _build_reverse_key_map(self) -> Dict[str, str]:
        """构建LoRA键名到原始参数的逆向映射表"""
        with open('../../utils/unet_keys_withU.json') as f:  # 用户提供的UNET结构定义
            unet_keys = json.load(f)
        reverse_map = {}
        for key in unet_keys:
            processed_key = key
            # 逆向处理键名转换逻辑
            processed_key = processed_key.replace('.proj_in', '.proj_in.lora')
            processed_key = processed_key.replace('.proj_out', '.proj_out.lora')
            processed_key = processed_key.replace('.to_q', '.processor.to_q_lora')
            processed_key = processed_key.replace('.to_k', '.processor.to_k_lora')
            processed_key = processed_key.replace('.to_v', '.processor.to_v_lora')
            processed_key = processed_key.replace('.to_out.0', '.processor.to_out_lora')
            if 'ff' in processed_key:
                processed_key += '.lora'
            reverse_map[processed_key] = key
        return reverse_map

    def to(self, device: str):
        """将SDIntegration3的所有模块移动到指定的设备"""
        self.base_pipeline = self.base_pipeline.to(device)
        self.current_pipeline = self.current_pipeline.to(device)
        self.device = device
        return self
    def _get_target_parameter(self, original_key: str):
        """根据原始键名定位目标参数"""
        current_module = self.current_pipeline.unet
        parts = original_key.split('.')
        for sub_key in parts:
            if hasattr(current_module, sub_key):
                current_module = getattr(current_module, sub_key)
            else:
                raise AttributeError(f"无效的模块路径: {original_key} 在{sub_key}处中断")
        if isinstance(current_module, torch.nn.Parameter):
            return current_module
        elif hasattr(current_module, 'weight'):
            return current_module.weight
        else:
            raise ValueError(f"未找到有效参数: {original_key}")
    def merge_lora(self,
                   lora_dict: Dict[str, torch.Tensor],
                   alpha: float = 1.0):
        """
        完整合并LoRA参数到UNet模型
        :param lora_dict: 符合unet_attn_processors_state_dict结构的参数字典
        :param alpha: 合并强度系数
        """
        processed_keys = set()
        self.current_pipeline = copy.deepcopy(self.base_pipeline)

        # 如果lora_dict是一个列表，则取第一个元素
        if isinstance(lora_dict, list):
            lora_dict = lora_dict[0]
        # 修改lora_dict键名，将每个键名前面的"unet."去掉
        for key in list(lora_dict.keys()):
            lora_dict[key.replace("unet.", "")] = lora_dict.pop(key)


        for lora_key in lora_dict:
            if lora_key in processed_keys:
                continue
            if not lora_key.endswith(".down.weight"):
                continue
            # 提取基础键名
            base_key = lora_key.rsplit(".", 2)[0]
            original_key = self._reverse_key_map.get(base_key)
            if not original_key:
                raise KeyError(f"未注册的LoRA键名: {base_key}")
            # 获取参数对
            up_key = lora_key.replace(".down.weight", ".up.weight")
            if up_key not in lora_dict:
                raise KeyError(f"缺失对应的up参数: {up_key}")
            lora_down = lora_dict[lora_key].to(self.device, self.torch_dtype)
            lora_up = lora_dict[up_key].to(self.device, self.torch_dtype)
            if (lora_down.dim()) == 4:  # use linear projection mismatch
                lora_down = lora_down.squeeze(3).squeeze(2)
                lora_up = lora_up.squeeze(3).squeeze(2)
                # 计算增量权重
                delta_w = alpha * torch.mm(lora_up, lora_down)
                # 定位并合并参数
                # 拓展delta_w的维度至四维
                delta_w = delta_w.view(delta_w.size(0), delta_w.size(1), 1, 1)
            else:
                delta_w = alpha * torch.mm(lora_up, lora_down)

            # 定位并合并参数
            target_param = self._get_target_parameter(original_key)
            target_param.data += delta_w.to(target_param.dtype)
            processed_keys.update({lora_key, up_key})
        # 验证完整性
        unprocessed = set(lora_dict.keys()) - processed_keys
        if unprocessed:
            raise KeyError(f"未处理的LoRA参数键: {unprocessed}")
    @contextmanager
    def temporary_merge(self, lora_dict: Dict, alpha: float = 1.0):
        """带自动回滚的上下文管理器"""
        original_state = copy.deepcopy(self.current_pipeline.unet.state_dict())
        try:
            self.merge_lora(lora_dict, alpha)
            yield
        finally:
            self.current_pipeline.unet.load_state_dict(original_state)
            torch.cuda.empty_cache()
    def generate(self,
                 prompt: str,
                 **generation_kwargs) -> torch.Tensor:
        """执行图像生成"""
        outputs = self.current_pipeline(
            prompt=prompt,
            output_type="tensor",
            height=256,
            width=256,
            **generation_kwargs
        )
        return outputs.images