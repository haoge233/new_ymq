import cv2
import numpy as np
import os
from find import extract_keyframes
from sp import create_video_from_images
import subprocess
import os
import json
from kmeans import PoseClustering
import logging
from test2 import get_similarity

# 初始化日志配置
def setup_logger(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),  # 日志文件
            logging.StreamHandler()             # 控制台输出
        ]
    )
log_file_path = "/home/haoge/ymq1/all_log.txt"
setup_logger(log_file_path)


logging.info("start.")
# 提取关键帧
video_path = 'videoplayback.mp4'  # 输入视频文件路径
output_dir = 'keyframes_output'  # 输出目录

keyframes = extract_keyframes(video_path, output_dir)

# 关键帧回写为视频
image_folder = output_dir  # 图片文件夹路径
output_video_path = 'output_video_2016.mp4'  # 输出视频路径

create_video_from_images(image_folder, output_video_path)


# 调用main。py
# 设置被调用脚本的路径
main_script_path = os.path.join("..", "solopose/SoloShuttlePose", "main.py")  # 调整路径到 main.py 所在位置

# 设置参数
folder_path = "videos"
result_path = "res"
force_flag = True  # 表示是否传递 --force

# 构建命令行参数
command = [
    "python", main_script_path,
    "--folder_path", folder_path,
    "--result_path", result_path,
]

# 添加 --force 标志（如果需要）
if force_flag:
    command.append("--force")

# 调用 main.py
try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print("Script output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("Error occurred:\n", e.stderr)

# 调用kmeans
# 假设我们有一个JSON文件  


# 根据视频路径生成 JSON 文件名
def generate_json_name(video_path, json_folder):
    # 获取视频文件名（不包括扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 生成对应的 JSON 文件路径
    json_name = os.path.join(json_folder, f"{base_name}.json")
    return json_name
json_folder_path = '/home/haoge/solopose/SoloShuttlePose/res/players/player_kp'
json_file_path = generate_json_name(output_video_path, json_folder_path)

# 读取JSON数据
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

pose_clustering = PoseClustering(json_data, clustering_method="hierarchical")

# pose_clustering.pnt()

# 获取5个核心姿态
core_poses = pose_clustering.predict_core_poses()
# 将结果记录到日志
logging.info("Core poses prediction completed.")
logging.info(f"Predicted core poses: {core_poses}")

# 计算D
output_file = 'frame_similarity_results.json'
min_similarity_file = 'min_similarity_indices.json'
get_similarity(json_file_path,core_poses,output_file,min_similarity_file)


# show
