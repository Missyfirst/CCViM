import cv2
import os
from PIL import Image
import numpy as np

def crop_and_save_images(image, output_dir):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算每个小块的高度和宽度
    crop_height = height // 4
    crop_width = width // 4

    for i in range(4):
        for j in range(4):
            # 计算当前小块的左上角和右下角坐标
            x1, y1 = j * crop_width, i * crop_height
            x2, y2 = (j + 1) * crop_width, (i + 1) * crop_height

            # 切割图像
            cropped_image = image[y1:y2, x1:x2]

            # 构建输出文件路径
            output_path = os.path.join(output_dir, f"cropped_image_{i}_{j}.jpg")

            # 保存切割后的小块
            cv2.imwrite(output_path, cropped_image)

# 读取 GIF 图像文件
image = Image.open('/opt/data/private/zhuyun/dataset/image_collect/images/20240316_01.gif')

# 将图像转换为 numpy 数组并转换为 RGB 格式
image_np = np.array(image.convert('RGB'))

# 切割图像并保存
output_dir = '/opt/data/private/zhuyun/dataset/image_collect/split_image/'
crop_and_save_images(image_np, output_dir)

print("切割后的小块已保存到", output_dir, "文件夹中")
