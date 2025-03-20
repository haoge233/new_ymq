import cv2
import os
import re

# 自定义排序函数，确保按数字顺序排序
def natural_sort_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

def create_video_from_images(image_folder, output_video_path, fps=25):#  1080的网盘资源下载的是25帧 奇怪
    # 获取文件夹中的所有图片，按文件名排序
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)

    # 检查是否有图片
    if not images:
        print("Error: No images found in the folder.")
        return
    
    # 获取第一张图片的路径
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    
    # 获取图像的大小
    height, width, _ = first_image.shape
    print(f"Image size: {width}x{height}")
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # 按顺序读取图片并写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        print(f"Reading image: {image_path}")  # 调试输出，确保顺序正确
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image {image_name}")
            continue
        
        # 可选：如果图片大小不一致，可以对其进行缩放
        if (width, height) != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, (width, height))
        
        out.write(image)  # 将图片写入视频
    
    # 释放资源
    out.release()
    print(f"Video saved to {output_video_path}")

# 使用示例
image_folder = '/home/haoge/ymq11/keyframes_output_2016-1'  # 图片文件夹路径
output_video_path = '20160501-1_key.mp4'  # 输出视频路径

create_video_from_images(image_folder, output_video_path)
