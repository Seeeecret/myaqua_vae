from watermark_embed import LoRAWatermarker
# from watermark_key import WatermarkKeyManager
from safetensors.torch import load_file

# Step 1: 初始化配置
config = {
    "img_size": 512,
    "freq_bands": [[5,5], [5,6], [6,5], [6,6], [7,5], [7,7]],
    "dct_channels": 3,
    "num_bits": 18,
    "rank": 4
}

# Step 2: 生成水印信息
msg_bits = "001011000100011101"
# msg_bits = "001011000111101010001011110010110101001011001010"

# Step 3: 嵌入水印
watermarker = LoRAWatermarker(config)
orig_lora = load_file("/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank4_bits4_output_0216/pytorch_lora_weights.safetensors")
watermarked_lora = watermarker.embed(orig_lora, msg_bits, "watermarked_lora.safetensors")


# Step 4: 保存密钥
# WatermarkKeyManager.generate(config, msg_bits, "watermark_key.json")

# Step 5: 生成测试图像
# ... 使用Stable Diffusion生成带水印图像

# Step 6: 训练检测器
# ... 使用watermark_detect模块训练
