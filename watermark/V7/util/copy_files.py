import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def copy_files(src_dir, dst_dir, recursive=True, overwrite=False):
    """
    将源文件夹中的所有文件复制到目标文件夹

    参数:
    src_dir (str): 源文件夹路径
    dst_dir (str): 目标文件夹路径
    recursive (bool): 是否递归复制子文件夹
    overwrite (bool): 是否覆盖已存在的文件
    """
    # 转换为Path对象
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # 确保源文件夹存在
    if not src_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_dir}")

    # 创建目标文件夹（如果不存在）
    dst_path.mkdir(parents=True, exist_ok=True)

    # 获取所有需要复制的文件
    if recursive:
        # 递归获取所有文件
        files = [f for f in src_path.rglob("*") if f.is_file()]
    else:
        # 只获取当前目录下的文件
        files = [f for f in src_path.glob("*") if f.is_file()]

    # 计算总文件数
    total_files = len(files)
    copied_files = 0
    skipped_files = 0
    failed_files = 0

    print(f"\n开始复制文件:")
    print(f"源文件夹: {src_dir}")
    print(f"目标文件夹: {dst_dir}")
    print(f"总文件数: {total_files}\n")

    # 使用tqdm创建进度条
    with tqdm(total=total_files, desc="复制进度") as pbar:
        for src_file in files:
            try:
                # 计算目标文件路径
                # 获取相对路径
                rel_path = src_file.relative_to(src_path)
                dst_file = dst_path / rel_path

                # 创建目标文件所在的文件夹
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # 检查目标文件是否已存在
                if dst_file.exists() and not overwrite:
                    print(f"跳过已存在的文件: {rel_path}")
                    skipped_files += 1
                else:
                    # 复制文件
                    shutil.copy2(src_file, dst_file)
                    copied_files += 1

            except Exception as e:
                print(f"复制文件失败 {src_file}: {str(e)}")
                failed_files += 1

            pbar.update(1)

    # 打印复制结果
    print("\n复制完成!")
    print(f"成功复制: {copied_files} 个文件")
    print(f"跳过文件: {skipped_files} 个文件")
    print(f"失败文件: {failed_files} 个文件")


def main():
    parser = argparse.ArgumentParser(description='复制文件夹中的所有文件到目标文件夹')
    parser.add_argument('src_dir', help='源文件夹路径')
    parser.add_argument('dst_dir', help='目标文件夹路径')
    parser.add_argument('--no-recursive', action='store_false', dest='recursive',
                        help='不递归复制子文件夹')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')

    args = parser.parse_args()

    try:
        copy_files(args.src_dir, args.dst_dir, args.recursive, args.overwrite)
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()