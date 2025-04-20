#!/bin/bash

# 检查是否为root用户
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：此脚本必须以root用户身份运行" >&2
    exit 1
fi

# 目标目录
TARGET_DIR="/baai-cwm-1/baai_cwm_ml/public_data/scenes/lightwheelocc-v1.0/vae_data/rank8_bits8_output_0321_valc"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误：目录 $TARGET_DIR 不存在" >&2
    exit 1
fi

# 修改目录及其所有子目录的权限为777
echo "正在修改目录权限..."
find "$TARGET_DIR" -type d -exec chmod 777 {} \;

# 修改所有文件的权限为666
echo "正在修改文件权限..."
find "$TARGET_DIR" -type f -exec chmod 777 {} \;

echo "权限修改完成。"