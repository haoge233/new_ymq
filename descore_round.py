import os
import json
import cv2
import numpy as np
import re
import pandas as pd

# 设定路径
video_folder = '/home/haoge/ymq1/20160430-lround_30_D40'
json_file_path = '/home/haoge/ymq1/20160430-json/frame_distances.json'
json_file_path_pose = '/home/haoge/ymq1/20160430-json/frame_similarity_results.json'
excel_file = '/home/haoge/ymq1/20160430-round-score_30_D40_pow09.xlsx'  # 需要写入得分的 Excel 文件路径

# 得分计算公式
def calculate_score(x):
    theta_0 = -16
    theta_1 = 2
    return 100 / (1 + np.exp(-(theta_0 + theta_1 * x / 10)))

def calculate_score_pose(x):
    theta_0 = -5
    theta_1 = 1
    return 100 / (1 + np.exp(-(theta_0 + theta_1 * x/10)))

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    frame_similarity_results = json.load(f)

with open(json_file_path_pose , 'r') as f:
    frame_similarity_results_pose  = json.load(f)

# 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
valid_video_pattern = re.compile(r'segment_(\d+)_([\d]+)\.mp4')

# 获取并排序视频文件夹中的文件列表，按数字范围的结束值倒序排列
def sort_key(video_file):
    match = valid_video_pattern.match(video_file)
    if match:
        # 获取视频的结束时间
        end_time = int(match.group(2))
        return end_time  # 按结束帧排序
    return 0  # 如果不匹配，默认返回0，保证这些文件排在最后

# 获取并排序视频文件夹中的文件列表
video_files = sorted(os.listdir(video_folder), key=sort_key)



# 初始化变量
last_end_frame = None
cumulative_score = 0  # 初始化累计得分
round_scores = []  # 存储每个回合的得分
video_names = []  # 存储每个视频的名称

# 遍历视频文件夹
for video_file in video_files:
    print(video_file)
    match = valid_video_pattern.match(video_file)
    
    if match:
        # 从命名规则中提取起始和结束帧
        start_frame, end_frame = map(int, match.groups())
        print(start_frame)
        print(end_frame)
        print(f"处理回合: {start_frame}-{end_frame}")

        # 获取该回合的相关帧信息
        relevant_frames = {frame: frame_similarity_results.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
        print(f"相关帧数量: {len(relevant_frames)}")
        relevant_frames_pose = {frame: frame_similarity_results_pose.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
        print(f"相关pose帧数量: {len(relevant_frames_pose)}")

        cumulative_score = 0  # 重置累计得分
        
        # 遍历相关帧并累计得分
        for frame, similarity in relevant_frames.items():
            if similarity:  # 如果帧没有被处理过
                # 计算得分
                x_value = similarity  # 获取第二个数据作为x值
                x_value_pose = relevant_frames_pose[frame][1]
                x_value_pose = calculate_score_pose(x_value_pose)
                print(x_value_pose)
                score = x_value_pose * pow(x_value, 0.9)
                cumulative_score += score  # 累加得分

        # 保存视频文件名和该回合的得分
        video_names.append(video_file)  # 保存视频文件名
        round_scores.append(cumulative_score)  # 保存每回合的得分


# print(video_names)
# print(round_scores)
# 创建一个DataFrame并保存到 Excel
score_df = pd.DataFrame({
    '视频名称': video_names,
    '回合得分': round_scores
})

# 将数据保存到Excel文件
score_df.to_excel(excel_file, index=False, sheet_name='得分汇总')

# 输出保存路径
print(f"视频得分已保存到: {excel_file}")




print("Processing completed.")