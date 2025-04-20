from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
import watermark
from watermark.V8.wartermark import ImageShield
import torch
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,5"

shield = ImageShield(ch_factor=1, hw_factor=8, height=64, width=64)  # 适配512x512图像
# 加载 Stable Diffusion v1.5 模型
print("Loading Stable Diffusion model...")
model_id = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/stable-diffusion-v1-5"

# 主Pipeline (生成图像) - 使用GPU 4 (cuda:0)
torch.cuda.set_device(0)  # 设置主设备
pipe_gen = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
pipe_gen.to("cuda:0")

# 反演Pipeline - 使用GPU 6 (cuda:1)
torch.cuda.set_device(1)
pipe_inv = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
pipe_inv.to("cuda:1")
pipe_inv.scheduler = DDIMInverseScheduler.from_config(pipe_inv.scheduler.config)

# 生成带水印的初始噪声
watermarked_noise = shield.create_watermark().to("cuda:0")

# 使用噪声生成图像
image = pipe_gen(prompt="a cat", latents=watermarked_noise,num_inference_steps=25).images[0]
# 预处理图像（转换为模型输入格式）
processed_image = pipe_gen.image_processor.preprocess(image.resize((512, 512))).to("cuda:1")

# 水印检测 - 使用反演Pipeline
print("Detecting watermark...")
# inverted_noise = shield.invert_image(pipe_inv, processed_image)
extracted, inverted_noise = shield.extract_watermark(processed_image, pipe_inv)
extracted = extracted if isinstance(extracted, torch.Tensor) else torch.tensor(extracted)

correct = (shield.watermark.cpu() == extracted).sum().item()
total_elements = shield.watermark.numel()
similarity_percentage = (correct / total_elements) * 100

print(f"Watermark detection accuracy: {similarity_percentage:.4f}")

# 篡改检测（示例：假设图像被修改）
# tampered_mask = shield.detect_tamper(inverted_noise)
# tampered_mask = tampered_mask.cpu().numpy()

# 可视化结果
# import matplotlib.pyplot as plt
#
# plt.imshow(image)
# plt.imshow(tampered_mask, alpha=0.5, cmap='jet')
# plt.show()
