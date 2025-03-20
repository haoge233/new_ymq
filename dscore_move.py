import os
import json
import cv2
import numpy as np
import re

# 设定路径
video_folder = '/home/haoge/solopose/SoloShuttlePose/res/videos/20160501-3'
json_file_path = '/home/haoge/ymq1/frame_distances.json'
keyframe_folder = '/home/haoge/ymq1/20160501_3_frames_output'
output_folder = 'move_score_output_folder_2016-3'

# 得分计算公式
def calculate_score(x):
    theta_0 = -4
    theta_1 = 1
    return 100 / (1 + np.exp(-(theta_0 + theta_1 * x/10)))

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    frame_similarity_results = json.load(f)

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
valid_video_pattern = re.compile(r'20160501-1+_(\d+)-(\d+)\.mp4')

# 遍历视频文件夹
for video_file in os.listdir(video_folder):
    # print(video_file)
    # 检查文件名是否符合有效的命名规则
    match = valid_video_pattern.match(video_file)
    
    if match:
        # 从命名规则中提取起始和结束帧
        start_frame, end_frame = map(int, match.groups())
        print(start_frame,end_frame)

        # 获取该回合的相关帧信息
        relevant_frames = {frame: frame_similarity_results.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
        print(len(relevant_frames))

        # 初始化累计得分
        cumulative_score = 0
        # 变量用于存储最后一个有similarity信息的帧
        last_similar_frame = None

        # 遍历相关帧
        for frame, similarity in relevant_frames.items():
            if similarity:
                last_similar_frame = frame  # 记录最后一个有similarity的帧
                x_value = similarity  # 获取第二个数据作为x值
                score = calculate_score(x_value)
                cumulative_score += score

                # 生成帧图像文件名
                keyframe_filename = f'frame_{frame}.jpg'
                keyframe_path = os.path.join(keyframe_folder, keyframe_filename)

                if os.path.exists(keyframe_path):
                    # 读取帧图像
                    img = cv2.imread(keyframe_path)

                    # 标注得分和累计得分
                    text = f'D: {x_value:.2f}\nScore: {score:.2f}\nCumulative: {cumulative_score:.2f}'
                    position = (50, 50)  # 标注位置
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, text, position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # 保存修改后的图像
                    output_filename = f'{frame}_annotated.jpg'
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, img)

        # 如果有找到最后一个有similarity的帧，输出帧号和与end的差值
        if last_similar_frame is not None:
            frame_difference = end_frame - last_similar_frame
            print(f"最后一个有similarity的帧号: {last_similar_frame}, 与end_frame的差值: {frame_difference}")
        else:
            print("没有找到有similarity信息的帧。")
print("Processing completed.")