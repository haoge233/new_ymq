import json
import os
import shutil

# JSON 文件路径
json_file_path = '/home/haoge/solopose/SoloShuttlePose/res vedio1/players/player_kp/output_video1.json'

# 源图片文件夹路径
source_images_folder = '/home/haoge/ymq1/keyframes_output'

# 目标新文件夹路径
destination_images_folder = 'path_to_your_destination_images_folder'

# 给定的 frame_ids 列表
frame_ids =['12260', '8005', '11831', '10272', '9588'] # 示例帧 ID，根据实际情况替换

# 创建目标文件夹（如果不存在）
if not os.path.exists(destination_images_folder):
    os.makedirs(destination_images_folder)
 
# 遍历 frame_ids 列表并提取图片
for frame_id in frame_ids:
    # 构建可能的图片文件名模式（这里假设文件名以 frame_ 开头，帧 ID 结尾，且扩展名为 .jpg）
    # 你可以根据实际情况调整这个模式
    file_pattern = f'keyframe_{frame_id}.jpg'
    
    # 在源文件夹中查找匹配的文件
    for filename in os.listdir(source_images_folder):
        if filename == file_pattern:
            # 构建源图片文件的完整路径
            source_image_path = os.path.join(source_images_folder, filename)
            
            # 构建目标图片文件的完整路径
            destination_image_path = os.path.join(destination_images_folder, filename)
            
            # 复制文件到新文件夹
            shutil.copy(source_image_path, destination_image_path)
            print(f"Copied {source_image_path} to {destination_image_path}")
            # 找到匹配的文件后，可以跳出循环（如果每个帧 ID 只对应一个文件）
            break
    else:
        # 如果循环结束都没有找到匹配的文件，则打印一条消息
        print(f"No image found for frame ID {frame_id}")
