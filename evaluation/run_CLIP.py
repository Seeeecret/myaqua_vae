#!/usr/bin/env python
# Filename: evaluate_clip_score.py

import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm

# diffusers & transformers
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor


def load_lora_weights(pipe: StableDiffusionPipeline, lora_path: str):
    """
    使用 diffusers 提供的相关接口加载和融合 LoRA。
    根据 .safetensors 文件名调用 pipe.load_lora_weights() 和 pipe.fuse_lora()。
    """
    print(f"[INFO] Loading LoRA weights from: {lora_path}")
    # 下面这行会尝试从指定路径加载名为 "pytorch_lora_weights.safetensors" 的文件，
    # 如果需要自定义文件名，可在 load_lora_weights() 中手动指定。
    pipe.load_lora_weights(lora_path)
    # 将加载的 LoRA 融合到基础模型中
    pipe.fuse_lora(lora_scale=1.0)


def compute_clip_score(
        clip_model: CLIPModel,
        clip_processor: CLIPProcessor,
        image: Image.Image,
        prompt: str,
        device: torch.device
) -> float:
    """
    使用CLIP模型计算图像与文本prompt之间的相似度 (CLIP Score)。
    返回一个余弦相似度分数。
    """
    inputs = clip_processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    # 获取图像/文本嵌入向量
    image_embeds = outputs.image_embeds  # [batch_size, embed_dim]
    text_embeds = outputs.text_embeds  # [batch_size, embed_dim]

    # 归一化后计算余弦相似度
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    similarity = (image_embeds * text_embeds).sum(dim=-1)  # 余弦相似度
    return similarity.item()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP Score using a dataset structure similar to run_tree_ring_watermark_fid.py")

    parser.add_argument('--run_name', default='test', help="Experiment name (used to organize outputs).")
    parser.add_argument('--start', default=0, type=int, help="Start index of the dataset slice.")
    parser.add_argument('--end', default=50, type=int, help="End index of the dataset slice, max 5000.")
    parser.add_argument('--lora', type=str, default=None,
                        help="Path to LoRA weights. If specified, LoRA fusion is enabled.")
    parser.add_argument('--image_length', default=512, type=int, help="Output image size (height=width).")
    parser.add_argument('--model_id', default='stable-diffusion-v1-5/stable-diffusion-v1-5',
                        help="Model repo id or local path of the base Stable Diffusion model.")
    parser.add_argument('--num_images', default=1, type=int, help="How many images to generate per prompt.")
    parser.add_argument('--guidance_scale', default=7.5, type=float, help="Classifier-free guidance scale.")
    parser.add_argument('--num_inference_steps', default=50, type=int, help="Steps for diffusion sampling.")
    parser.add_argument('--gen_seed', default=0, type=int, help="Base random seed for generation.")
    parser.add_argument('--prompt_file', default='coco/meta_data.json', type=str,
                        help="Path to the JSON file containing dataset with 'images' and 'annotations'.")
    parser.add_argument('--gt_folder', default='coco/ground_truth', type=str,
                        help="(Not used in CLIP Score) Ground truth images folder, if needed for other metrics.")
    parser.add_argument('--output_dir', default='./clip_evaluation', type=str,
                        help="Where to save generated images.")
    parser.add_argument('--clip_model_id', type=str, default="openai/clip-vit-large-patch14",
                        help="Which CLIP model to use from HuggingFace/transformers.")

    args = parser.parse_args()

    # 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Stable Diffusion
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None
    ).to(device)

    # 如果用户指定了 LoRA 路径，则进行融合
    if args.lora is not None:
        load_lora_weights(pipe, args.lora)
    else:
        print("[INFO] Not using LoRA.")

    # 2. 加载 CLIP 模型
    print(f"[INFO] Loading CLIP model from: {args.clip_model_id}")
    clip_model = CLIPModel.from_pretrained(args.clip_model_id).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_id)

    # 3. 读取与 run_tree_ring_watermark_fid.py 相同结构的数据
    #   dataset = {
    #       "images": [{"file_name": "...", "id": ...}, ...],
    #       "annotations": [{"caption": "...", "id": ...}, ...]
    #   }
    prompt_key = "caption"
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        image_files = data["images"]  # 一个包含 {file_name, id, ...} 的列表
        annotations = data["annotations"]  # 一个包含 {caption, id, ...} 的列表

    # 4. 创建输出目录
    output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(output_dir, exist_ok=True)

    clip_scores = []

    # 5. 根据索引范围循环生成图像并计算 CLIP Score
    end_index = min(args.end, len(annotations))
    for i in tqdm(range(args.start, end_index), desc="Generating & Evaluating"):
        # 取对应的 Prompt
        current_prompt = annotations[i][prompt_key]
        image_file_name = image_files[i]["file_name"]  # 用于保存图像

        # 计算随机种子
        seed = i + args.gen_seed
        generator = torch.Generator(device=device).manual_seed(seed)

        # 生成图像
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
            output = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                generator=generator
            )
        gen_image = output.images[0]

        # 保存图像
        # 每隔20个图像保存一次，避免生成过多图像导致溢出
        if i % 20 == 0:
            save_path = os.path.join(output_dir, image_file_name)
            gen_image.save(save_path)
            print(f"Saved to {save_path}\n")

        # 计算 CLIP Score
        score = compute_clip_score(clip_model, clip_processor, gen_image, current_prompt, device)
        clip_scores.append(score)

        print(f"\nIndex: {i}, Prompt: {current_prompt}\nFile: {image_file_name}")
        print(f"CLIP Score: {score:.4f}")

    # 6. 统计并打印平均 CLIP Score
    if len(clip_scores) > 0:
        avg_clip_score = sum(clip_scores) / len(clip_scores)
    else:
        avg_clip_score = 0.0

    print("=" * 50)
    print(f"Finished generating from index {args.start} to {end_index - 1}")
    print(f"Average CLIP Score: {avg_clip_score:.4f}")
    print(f"Results saved in: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
