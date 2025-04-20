#!/bin/bash

# 检查是否为root用户
if [ "$(id -u)" -ne 0 ]; then
    echo "错误：此脚本必须以root用户身份运行" >&2
    exit 1
fi

# 目标目录
TARGET_DIR="/baai-cwm-1/baai_cwm_ml/algorithm/ziyang.yan"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误：目录 $TARGET_DIR 不存在" >&2
    exit 1
fi

# 查找并删除所有以checkpoint开头的文件夹
echo "正在查找并删除以checkpoint开头的文件夹..."
find "$TARGET_DIR" -type d -name "myaqua_va*" -exec rm -rf {} \;

echo "删除操作完成。"