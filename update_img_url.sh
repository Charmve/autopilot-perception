#!/bin/bash

# Markdown 文件路径
MD_FILE=${1:-"docs/01-感知系统整体概述.md"}
# 临时文件
TEMP_FILE="temp.md"

# 创建目录用于保存下载的图片
IMAGE_DIR="../images"
mkdir -p "$IMAGE_DIR"

# grep -r "在这里插入图片描述" .  | sed -n 's/.*!\[.*\](\([^)]*\)).*/\1/p' | xargs wget

# 提取并处理指定描述的图片链接
grep -r '在这里插入图片描述\|\[Image\]' "$MD_FILE" | while read -r LINE; do    
    # 提取图片 URL
    IMAGE_URL=$(echo "$LINE" | sed -n 's/.*!\[.*\](\([^)]*\)).*/\1/p')
    MD_FILE=$(echo "$LINE" | cut -d ':' -f 1 | xargs basename)

    echo $MD_FILE
    
    if [ -n "$IMAGE_URL" ]; then
        # 获取文件名
        FILE_NAME=$(basename "$IMAGE_URL")
        
        # 下载图片
        wget -q -P "$IMAGE_DIR" "$IMAGE_URL"
        
        # 替换 Markdown 文件中的链接
        sed -i "s|$IMAGE_URL|$IMAGE_DIR/$FILE_NAME|g" "$MD_FILE"
        
        echo "下载并替换链接: $IMAGE_URL -> $IMAGE_DIR/$FILE_NAME"
    fi
done

echo "所有图片下载完成，Markdown 链接已更新。"
