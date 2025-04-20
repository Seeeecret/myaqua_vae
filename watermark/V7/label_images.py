import os
import glob
import json


def create_image_labels(folder_path, output_json='image_labels.json',msg='0010'):
    """
    扫描指定文件夹下的所有图片，并创建一个JSON文件，
    将每个图片文件名与标签"0010"关联起来

    参数:
    folder_path (str): 图片文件夹路径
    output_json (str): 输出的JSON文件名
    """
    # 获取所有常见图片格式文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_files = []
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))

    print(f"找到 {len(image_files)} 个图片文件")

    # 创建标签数据
    image_labels = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        image_labels[filename] = {"msg": msg}

    # 保存到JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(image_labels, f, ensure_ascii=False, indent=4)

    print(f"标签已成功保存到 {output_json}")
    return image_labels


if __name__ == "__main__":

    folder_path = '/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan/myaqua_vae/evaluation/rank4_bits4_output_0323_0101'
    output_json = os.path.join(folder_path,"image_labels.json")
    msg = '0101'
    create_image_labels(folder_path, output_json,msg)
    print("处理完成！")