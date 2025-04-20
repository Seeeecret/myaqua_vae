from datetime import datetime
# 这个代码能跑起来
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler
from watermark.V8.wartermark2 import ImageShield
import torch
import os

# 定义输出目录
output_dir = "./watermarked_images"
os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
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
# pipe_inv = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
# pipe_inv.to("cuda:1")
# pipe_inv.scheduler = DDIMInverseScheduler.from_config(pipe_inv.scheduler.config)

# 生成带水印的初始噪声
watermarked_noise = shield.create_watermark().to("cuda:0")

diff_prompt="A cute cartoon character with a potted plant on his head"
# 使用噪声生成图像
image = pipe_gen(prompt=diff_prompt, latents=watermarked_noise, num_inference_steps=50).images[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
prompt_hash = abs(hash(diff_prompt)) % 1000  # 取prompt哈希后3位
filename = f"wm_{timestamp}_{prompt_hash}.png"
save_path = os.path.join(output_dir, filename)
# 保存图像并打印路径
image.save(save_path)
print(f"\nGenerated image saved to: {save_path}")
# 预处理图像（转换为模型输入格式）
processed_image = pipe_gen.image_processor.preprocess(image.resize((512, 512))).to("cuda:1")
print("Detecting watermark...")

inverted_latents = shield.invert_image(pipe_gen, image, num_inversion_steps=50)
decoded_watermark, accuracy = shield.extract_watermark(inverted_latents)
# 水印检测 - 使用反演Pipeline

print(f"Watermark detection accuracy: {accuracy:.4f}")
# tamper_mask = shield.detect_tamper(inverted_latents)

# 篡改检测（示例：假设图像被修改）
# tampered_mask = shield.detect_tamper(inverted_noise)
# tampered_mask = tampered_mask.cpu().numpy()

# 可视化结果
# import matplotlib.pyplot as plt
#
# plt.imshow(image)
# plt.imshow(tampered_mask, alpha=0.5, cmap='jet')
# plt.show()
