import json
import os

# 配置参数
config = {
    "num_bits": 18,
    "msg_bits": "001011000100011101"  # 你的18位水印信息
}

# 图像路径参数
image_dir = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/V6output/sampleDataset"
start_idx = 0
end_idx = 999  # 包含1000


def generate_labels():
    # 验证水印长度
    assert len(config["msg_bits"]) == config["num_bits"], "水印长度与配置不符"

    labels = {}

    # 遍历所有索引
    for idx in range(start_idx, end_idx + 1):
        filename = f"watermarked_{idx:04d}"  # 格式化为4位数字
        labels[filename] = config["msg_bits"]

    # 保存为JSON
    output_path = os.path.join('./', "watermark_labels.json")
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"成功生成标签文件：{output_path}")
    print(f"共包含 {len(labels)} 个样本的标签")


if __name__ == "__main__":
    generate_labels()
