import os
import glob
import json
import shutil
from pathlib import Path


def rename_and_label_images(folder_path, output_json='image_labels.json', prefix='msg'):
    """
    扫描指定文件夹下的所有图片，将图片重命名为prefix+原文件名，
    并创建一个JSON文件，将每个图片文件名与标签"0010"关联起来

    参数:
    folder_path (str): 图片文件夹路径
    output_json (str): 输出的JSON文件名
    prefix (str): 添加到图片名前的前缀
    """
    # 获取所有常见图片格式文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []

    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))

    print(f"找到 {len(image_files)} 个图片文件")

    # 创建标签数据并重命名文件
    image_labels = {}
    renamed_count = 0

    for img_path in image_files:
        # 获取文件路径、文件名和扩展名
        file_dir = os.path.dirname(img_path)
        filename = os.path.basename(img_path)


        # 构造新文件名
        new_filename = f"{prefix}_{filename}"
        new_path = os.path.join(file_dir, new_filename)
        # 重命名文件
        try:
            shutil.move(img_path, new_path)
            renamed_count += 1
            print(f"重命名: {filename} -> {new_filename}")
        except Exception as e:
            print(f"重命名 {filename} 时出错: {e}")
            new_filename = filename  # 如果重命名失败，使用原文件名
        # 添加到标签数据
        image_labels[new_filename] = {"msg": prefix}

    # 保存到JSON文件
    with open(os.path.join(folder_path, output_json), 'w', encoding='utf-8') as f:
        json.dump(image_labels, f, ensure_ascii=False, indent=4)

    print(f"成功重命名 {renamed_count} 个文件")
    print(f"标签已成功保存到 {os.path.join(folder_path, output_json)}")
    return image_labels


def batch_process_folders(base_folder, output_json='image_labels.json', prefix='msg'):
    """
    处理基础文件夹下的所有子文件夹中的图片

    参数:
    base_folder (str): 基础文件夹路径
    output_json (str): 输出的JSON文件名
    prefix (str): 添加到图片名前的前缀
    """
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(base_folder)
               if os.path.isdir(os.path.join(base_folder, d))]

    if not subdirs:
        # 如果没有子文件夹，直接处理基础文件夹
        print(f"处理文件夹: {base_folder}")
        rename_and_label_images(base_folder, output_json, prefix)
    else:
        # 处理每个子文件夹
        for subdir in subdirs:
            folder_path = os.path.join(base_folder, subdir)
            print(f"处理子文件夹: {folder_path}")
            rename_and_label_images(folder_path, output_json, prefix)

    print("所有文件夹处理完成！")


if __name__ == "__main__":
    # 可以从命令行接收参数，或者直接在这里设置路径
    import argparse

    # parser = argparse.ArgumentParser(description='重命名图片并创建标签JSON文件')
    # parser.add_argument('--folder', type=str, required=True,
    #                     help='图片文件夹路径')
    # parser.add_argument('--output', type=str, default='image_labels.json',
    #                     help='输出的JSON文件名')
    # parser.add_argument('--prefix', type=str, default='msg',
    #                     help='添加到图片名前的前缀')
    # parser.add_argument('--batch', action='store_true',
    #                     help='是否处理所有子文件夹')
    #
    # args = parser.parse_args()



    # if args.batch:
    #     batch_process_folders(args.folder, args.output, args.prefix)
    # else:
    #     rename_and_label_images(args.folder, args.output, args.prefix)

    folder_path = '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/evaluation/rank4_bits4_output_0321_0010'
    output_json = os.path.join(folder_path,"image_labels_prefix.json")
    msg='0010'

    rename_and_label_images(folder_path, output_json, msg)

    print("处理完成！")