#!/bin/bash

# 设置文件ID和输出文件名
FILE_ID="1mjIqU-19aDupa9xtsimA5s6WxsXZ2eUX71p5IZJ"   # 替换成实际文件ID
OUTPUT="fid_outputs.zip"        # 替换成你想保存的文件名

# 获取下载页面并保存响应
RESPONSE=$(wget --quiet --save-cookies cookies.txt \
    --keep-session-cookies --no-check-certificate \
    "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
    -O -)

# 保存响应以供调试
echo "$RESPONSE" > response.html

# 提取完整的下载参数
DOWNLOAD_URL="https://drive.usercontent.google.com/download"
ID=$(echo "$RESPONSE" | grep -o 'name="id" value="[^"]*"' | cut -d'"' -f4)
EXPORT=$(echo "$RESPONSE" | grep -o 'name="export" value="[^"]*"' | cut -d'"' -f4)
CONFIRM=$(echo "$RESPONSE" | grep -o 'name="confirm" value="[^"]*"' | cut -d'"' -f4)
UUID=$(echo "$RESPONSE" | grep -o 'name="uuid" value="[^"]*"' | cut -d'"' -f4)

# 检查是否所有参数都已获取
if [ -z "$ID" ] || [ -z "$EXPORT" ] || [ -z "$CONFIRM" ] || [ -z "$UUID" ]; then
    echo "无法提取所有必要的下载参数"
    echo "ID: $ID"
    echo "EXPORT: $EXPORT"
    echo "CONFIRM: $CONFIRM"
    echo "UUID: $UUID"
    exit 1
fi

# 构建完整的下载URL
FULL_URL="${DOWNLOAD_URL}?id=${ID}&export=${EXPORT}&confirm=${CONFIRM}&uuid=${UUID}"

echo "开始下载文件..."
echo "使用URL: $FULL_URL"

# 执行下载
wget --load-cookies cookies.txt \
    --no-check-certificate \
    "$FULL_URL" \
    -O "$OUTPUT"

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "文件下载成功：$OUTPUT"
else
    echo "下载失败"
    exit 1
fi

# 清理临时文件
rm -f cookies.txt

exit 0