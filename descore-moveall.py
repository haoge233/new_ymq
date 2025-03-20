# import os
# import json
# import cv2
# import numpy as np
# import re
# import pandas as pd

# # 设定路径
# video_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-1/videos/20160501-3'
# json_file_path = '/home/haoge/ymq1/frame_distances.json'
# keyframe_folder = '/home/haoge/ymq1/20160501_1_frames_output'
# output_folder = '/'
# excel_file = '/home/haoge/ymq1/2016-1-mtest.xlsx'  # 需要写入得分的 Excel 文件路径

# # 得分计算公式
# def calculate_score(x):
#     theta_0 = -4
#     theta_1 = 1
#     return 100 / (1 + np.exp(-(theta_0 + theta_1 * x / 10)))

# # 读取 JSON 文件
# with open(json_file_path, 'r') as f:
#     frame_similarity_results = json.load(f)

# # 创建输出文件夹
# os.makedirs(output_folder, exist_ok=True)

# # 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
# valid_video_pattern = re.compile(r'20160501-3+_(\d+)-(\d+)\.mp4')

# # 获取并排序视频文件夹中的文件列表，按数字范围的结束值倒序排列
# def sort_key(video_file):
#     match = valid_video_pattern.match(video_file)
#     if match:
#         # 获取视频的结束时间
#         end_time = int(match.group(2))
#         return end_time  # 取正值进行倒序排序
#     return 0  # 如果不匹配，默认返回0，保证这些文件排在最后

# # 获取并排序视频文件夹中的文件列表
# video_files = sorted(os.listdir(video_folder), key=sort_key)

# # 读取现有的 Excel 文件
# df = pd.read_excel(excel_file)

# # 用于存储每个回合的累计得分
# round_scores = []
# # 初始化累计得分
# cumulative_score = 0
# last_end_frame = None
import os
import json
import cv2
import numpy as np
import re
import pandas as pd

# 设定路径
video_folder = '/home/haoge/solopose/SoloShuttlePose/res/videos/20160501-3'
json_file_path = '/home/haoge/ymq1/frame_distances.json'
keyframe_folder = '/home/haoge/ymq1/20160501_3_frames_output'
excel_file = '/home/haoge/ymq1/2016-3-pose-t.xlsx'  # 需要写入得分的 Excel 文件路径

# 得分计算公式
def calculate_score(x):
    theta_0 = -16
    theta_1 = 2
    return 100 / (1 + np.exp(-(theta_0 + theta_1 * x / 10)))

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    frame_similarity_results = json.load(f)



# 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
valid_video_pattern = re.compile(r'20160501-3+_(\d+)-(\d+)\.mp4')

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

# 读取现有的 Excel 文件
df = pd.read_excel(excel_file)

# 初始化变量
last_end_frame = None
cumulative_score = 0  # 初始化累计得分
round_scores = []  # 存储每个回合的得分
processed_frames = set()  # 用于记录已经处理过的帧

# 遍历视频文件夹
for video_file in video_files:
    print(video_file)
    match = valid_video_pattern.match(video_file)
    
    if match:
        # 从命名规则中提取起始和结束帧
        start_frame, end_frame = map(int, match.groups())
        print(f"处理回合: {start_frame}-{end_frame}")

        # 获取该回合的相关帧信息
        relevant_frames = {frame: frame_similarity_results.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
        print(f"相关帧数量: {len(relevant_frames)}")

        # 判断是否需要重置累计得分
        if last_end_frame is not None and (start_frame - last_end_frame) > 100:
            # 如果上一回合的结束帧和当前回合的开始帧的差值大于100，重置得分
            processed_frames.clear()  # 清空已处理帧记录
            round_scores.append(cumulative_score)  # 把得分添加到结果
            cumulative_score = 0  # 重置累计得分
        
        last_frame_in_round=None
       # 遍历相关帧并累计得分
        for frame, similarity in relevant_frames.items():
            if frame not in processed_frames and similarity:  # 如果帧没有被处理过
                # 如果当前帧与上一帧的差值大于100，保存当前累计得分，并重置累计得分
                if last_frame_in_round is not None and (frame - last_frame_in_round) > 100:
                    round_scores.append(cumulative_score)  # 把当前得分保存到结果
                    cumulative_score = 0  # 重置累计得分

                # 计算得分
                x_value = similarity  # 获取第二个数据作为x值
                score = calculate_score(x_value)
                cumulative_score += score  # 累加得分
                processed_frames.add(frame)  # 标记该帧已经处理

                # 更新回合中的最后一帧
                last_frame_in_round = frame




        # 更新上一个回合的结束帧
        last_end_frame = end_frame

# 如果最后一个回合没有添加得分，需要手动添加
if cumulative_score > 0:
    round_scores.append(cumulative_score)





# 将回合得分保存到Excel文件中的“回合总分”这一列
df['回合总分'] = round_scores

print(df['回合总分'])
# 保存新的Excel文件
df.to_excel('2016-3-move-t.xlsx', index=False)

print("Processing completed.")