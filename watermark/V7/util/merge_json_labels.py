import os
import json
import glob
import argparse
from pathlib import Path


def find_all_json_files(path):
    """
    递归查找指定路径下的所有JSON文件

    参数:
    path (str): 文件或文件夹路径

    返回:
    list: JSON文件路径列表
    """
    if os.path.isfile(path) and path.lower().endswith('.json'):
        return [path]

    result = []
    if os.path.isdir(path):
        # 首先查找当前目录下的JSON文件
        result.extend(glob.glob(os.path.join(path, "*.json")))

        # 然后递归查找所有子目录
        for item in os.listdir(path):
            subpath = os.path.join(path, item)
            if os.path.isdir(subpath):
                result.extend(find_all_json_files(subpath))

    return result
def merge_json_label_files(input_paths, output_path, allow_duplicates=False, verbose=True):
    """
    合并多个JSON标签文件到一个文件中

    参数:
    input_paths (list): JSON文件路径列表或包含JSON文件的文件夹路径列表
    output_path (str): 输出的合并后的JSON文件路径
    allow_duplicates (bool): 是否允许重复的键(文件名)，True表示后面的覆盖前面的
    verbose (bool): 是否打印详细信息

    返回:
    dict: 合并后的标签字典
    """
    merged_labels = {}
    total_files = 0
    duplicate_count = 0
    processed_files = 0

    # 处理输入路径，可能是JSON文件或文件夹（递归搜索子文件夹）
    json_files = []
    for path in input_paths:
        # 递归查找所有JSON文件
        json_files.extend(find_all_json_files(path))

    if verbose:
        print(f"在所有文件夹及子文件夹中找到 {len(json_files)} 个JSON文件需要合并")

    if verbose:
        print(f"找到 {len(json_files)} 个JSON文件需要合并")

    # 遍历每个JSON文件并合并
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                if verbose:
                    print(f"处理文件: {json_file}")
                labels = json.load(f)

                # 统计信息
                total_files += 1
                file_key_count = len(labels)
                processed_files += file_key_count

                # 合并标签
                for key, value in labels.items():
                    if key in merged_labels and not allow_duplicates:
                        duplicate_count += 1
                        if verbose:
                            print(f"警告: 重复的键 '{key}' 在文件 {json_file} 中被跳过")
                    else:
                        merged_labels[key] = value

                if verbose:
                    print(f"  - 添加了 {file_key_count} 个条目")

        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")

    # 保存合并后的JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_labels, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"\n合并完成:")
            print(f"- 处理文件总数: {total_files}")
            print(f"- 总条目数: {len(merged_labels)}")
            print(f"- 重复键数量: {duplicate_count}")
            print(f"- 已保存到: {output_path}")
    except Exception as e:
        print(f"保存合并文件时出错: {e}")

    return merged_labels


def validate_merged_labels(merged_labels):
    """
    验证合并后的标签数据是否有效

    参数:
    merged_labels (dict): 合并后的标签字典

    返回:
    bool: 是否有效
    """
    try:
        # 检查是否为空
        if not merged_labels:
            print("警告: 合并后的标签为空")
            return False

        # 检查标签格式
        for key, value in list(merged_labels.items())[:5]:  # 只检查前5个
            if not isinstance(value, dict) or "msg" not in value:
                print(f"警告: 键 '{key}' 的值不符合预期格式")
                return False

        print("合并后的标签验证通过")
        return True
    except Exception as e:
        print(f"验证标签时出错: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并多个JSON标签文件')
    parser.add_argument('--inputs', nargs='+', required=True,
                        help='输入的JSON文件路径或文件夹路径列表')
    parser.add_argument('--output', type=str, required=True,
                        help='输出的合并后的JSON文件路径')
    parser.add_argument('--allow-duplicates', action='store_true',
                        help='是否允许重复的键(文件名)，允许则后面的覆盖前面的')
    parser.add_argument('--quiet', action='store_true',
                        help='减少输出信息')

    args = parser.parse_args()

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 合并标签
    merged_labels = merge_json_label_files(
        args.inputs,
        args.output,
        allow_duplicates=args.allow_duplicates,
        verbose=not args.quiet
    )

    # 验证合并结果
    if not args.quiet:
        validate_merged_labels(merged_labels)

    print("处理完成！")