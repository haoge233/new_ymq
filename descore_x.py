# # import os
# # import json
# # import cv2
# # import numpy as np
# # import re
# # import pandas as pd

# # # 设定路径
# # video_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-1/videos/20160501-3'
# # json_file_path = '/home/haoge/ymq1/frame_distances.json'
# # keyframe_folder = '/home/haoge/ymq1/20160501-3_frames_output'
# # output_folder = '/'
# # excel_file = '/home/haoge/ymq1/2016-1-mtest.xlsx'  # 需要写入得分的 Excel 文件路径

# # # 得分计算公式
# # def calculate_score(x):
# #     theta_0 = -4
# #     theta_1 = 1
# #     return 100 / (1 + np.exp(-(theta_0 + theta_1 * x / 10)))

# # # 读取 JSON 文件
# # with open(json_file_path, 'r') as f:
# #     frame_similarity_results = json.load(f)

# # # 创建输出文件夹
# # os.makedirs(output_folder, exist_ok=True)

# # # 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
# # valid_video_pattern = re.compile(r'20160501-3+_(\d+)-(\d+)\.mp4')

# # # 获取并排序视频文件夹中的文件列表，按数字范围的结束值倒序排列
# # def sort_key(video_file):
# #     match = valid_video_pattern.match(video_file)
# #     if match:
# #         # 获取视频的结束时间
# #         end_time = int(match.group(2))
# #         return end_time  # 取正值进行倒序排序
# #     return 0  # 如果不匹配，默认返回0，保证这些文件排在最后

# # # 获取并排序视频文件夹中的文件列表
# # video_files = sorted(os.listdir(video_folder), key=sort_key)

# # # 读取现有的 Excel 文件
# # df = pd.read_excel(excel_file)

# # # 用于存储每个回合的累计得分
# # round_scores = []
# # # 初始化累计得分
# # cumulative_score = 0
# # last_end_frame = None
import os
import json
import cv2
import numpy as np
import re
import pandas as pd

# 设定路径
video_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/videos/20160501-3'
json_file_path = '/home/haoge/ymq1/frame_distances.json'
json_file_path_pose = '/home/haoge/ymq1/frame_similarity_results.json'
keyframe_folder = '/home/haoge/ymq1/20160501-3_frames_output'
excel_file = '/home/haoge/ymq1/2016-3-pose-t.xlsx'  # 需要写入得分的 Excel 文件路径

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
        relevant_frames_pose = {frame: frame_similarity_results_pose.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
        print(f"相关pose帧数量: {len(relevant_frames_pose)}")

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
                print(x_value)
                # print(type(relevant_frames_pose))
                # print(relevant_frames_pose)
                x_value_pose=relevant_frames_pose[frame][0]
                x_value_pose=calculate_score_pose(x_value_pose)
                print(x_value_pose)
                score = x_value_pose*pow(x_value,1)
                cumulative_score += score  # 累加得分
                processed_frames.add(frame)  # 标记该帧已经处理

                # 更新回合中的最后一帧
                last_frame_in_round = frame




        # 更新上一个回合的结束帧
        last_end_frame = end_frame

# 如果最后一个回合没有添加得分，需要手动添加
if cumulative_score > 0:
    round_scores.append(cumulative_score)

# # 2016-1需要这个
# print(round_scores[19])
# score_to_move = round_scores[19]
# round_scores.pop(19)  # 删除第 19 个元素
# round_scores.append(score_to_move)  # 将元素添加到末尾


def custom_sigmoid(x, theta_0=6, theta_1=-4.4):
    return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5
df['回合总分'] = round_scores

df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})

# 计算 "有效时长"
df['有效时长'] = df['回合时长'] 

# # 定义网格搜索的参数范围
# theta_0_values = np.linspace(3, 7, 9)  # 你可以调整这个范围和步长
# theta_1_values = np.linspace(-7, -4, 16)  # 你可以调整这个范围和步长

theta_0_values = np.arange(3, 7.1, 0.1)  # 间隔 0.1，包含 7.0
theta_1_values = np.arange(-7, -4.1, 0.1)  # 间隔 0.1，包含 -4.0


# 进行网格搜索
for theta_0 in theta_0_values:
    for theta_1 in theta_1_values:
        xx=df['回合总分']
        xx=xx/(custom_sigmoid(df['回合时长'],theta_0,theta_1)*df['回合时长'])
        print(xx)
        df['有效得分'] = (xx + df['是否回合被分割'] * 1000) 

                # 按 "有效得分" 排序（降序：从高到低）
        df_sorted = df.sort_values(by='有效得分', ascending=False)

        # 保存排序后的结果到新的 Excel 文件
        output_file_path = f"/home/haoge/ymq1/grid_x_3_sig_pow1/x_grid_search_result_theta0_{theta_0}_theta1_{theta_1}.xlsx"
        df_sorted.to_excel(output_file_path, index=False)

        # 输出保存路径
        print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

# theta_0=5.5
# theta_1=-4.5
# xx=df['回合总分']
# xx=xx/(custom_sigmoid(df['回合时长'],theta_0,theta_1)*df['回合时长'])
# print(xx)
# df['有效得分'] = (xx + df['是否回合被分割'] * 1000) 

#         # 按 "有效得分" 排序（降序：从高到低）
# df_sorted = df.sort_values(by='有效得分', ascending=False)

# # 保存排序后的结果到新的 Excel 文件
# output_file_path = f"2016-3-x_{theta_0}_{theta_1}.xlsx"
# df_sorted.to_excel(output_file_path, index=False)

# # 输出保存路径
# print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

# 将回合得分保存到Excel文件中的“回合总分”这一列


# # 保存新的Excel文件
# df.to_excel('2016-1-mtest_with_scores_theta0_x_t6-44_20.xlsx', index=False)

print("Processing completed.")



# import os
# import json
# import cv2
# import numpy as np
# import re
# import pandas as pd

# # 设定路径
# video_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-1/videos/20160501-3'
# json_file_path = '/home/haoge/ymq1/frame_distances.json'
# keyframe_folder = '/home/haoge/ymq1/20160501-3_frames_output'
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



# 2016-1
# import os
# import json
# import cv2
# import numpy as np
# import re
# import pandas as pd

# # 设定路径
# video_folder = '/home/haoge/solopose/SoloShuttlePose/res 2016-1/videos/20160501-1'
# json_file_path = '/home/haoge/ymq1/20160501-1_json/frame_distances.json'
# json_file_path_pose = '/home/haoge/ymq1/20160501-1_json/frame_similarity_results.json'
# keyframe_folder = '/home/haoge/ymq1/20160501-1_frames_output'
# excel_file = '/home/haoge/ymq1/xlsx/2016-1.xlsx'  # 需要写入得分的 Excel 文件路径

# # 得分计算公式
# def calculate_score(x):
#     theta_0 = -16
#     theta_1 = 2
#     return 100 / (1 + np.exp(-(theta_0 + theta_1 * x / 10)))

# def calculate_score_pose(x):
#     theta_0 = -5
#     theta_1 = 1
#     return 100 / (1 + np.exp(-(theta_0 + theta_1 * x/10)))

# # 读取 JSON 文件
# with open(json_file_path, 'r') as f:
#     frame_similarity_results = json.load(f)

# with open(json_file_path_pose , 'r') as f:
#     frame_similarity_results_pose  = json.load(f)

# # 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
# valid_video_pattern = re.compile(r'20160501-1+_(\d+)-(\d+)\.mp4')

# # 获取并排序视频文件夹中的文件列表，按数字范围的结束值倒序排列
# def sort_key(video_file):
#     match = valid_video_pattern.match(video_file)
#     if match:
#         # 获取视频的结束时间
#         end_time = int(match.group(2))
#         return end_time  # 按结束帧排序
#     return 0  # 如果不匹配，默认返回0，保证这些文件排在最后

# # 获取并排序视频文件夹中的文件列表
# video_files = sorted(os.listdir(video_folder), key=sort_key)

# # 读取现有的 Excel 文件
# df = pd.read_excel(excel_file)

# # 初始化变量
# last_end_frame = None
# cumulative_score = 0  # 初始化累计得分
# round_scores = []  # 存储每个回合的得分
# processed_frames = set()  # 用于记录已经处理过的帧

# # 遍历视频文件夹
# for video_file in video_files:
#     print(video_file)
#     match = valid_video_pattern.match(video_file)
    
#     if match:
#         # 从命名规则中提取起始和结束帧
#         start_frame, end_frame = map(int, match.groups())
#         print(f"处理回合: {start_frame}-{end_frame}")

#         # 获取该回合的相关帧信息
#         relevant_frames = {frame: frame_similarity_results.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
#         print(f"相关帧数量: {len(relevant_frames)}")
#         relevant_frames_pose = {frame: frame_similarity_results_pose.get(str(frame)) for frame in range(start_frame, end_frame + 1)}
#         print(f"相关pose帧数量: {len(relevant_frames_pose)}")

#         # 判断是否需要重置累计得分
#         if last_end_frame is not None and (start_frame - last_end_frame) > 100:
#             # 如果上一回合的结束帧和当前回合的开始帧的差值大于100，重置得分
#             processed_frames.clear()  # 清空已处理帧记录
#             round_scores.append(cumulative_score)  # 把得分添加到结果
#             cumulative_score = 0  # 重置累计得分
        
#         last_frame_in_round=None
#        # 遍历相关帧并累计得分
#         for frame, similarity in relevant_frames.items():
#             if frame not in processed_frames and similarity:  # 如果帧没有被处理过
#                 print(frame)
#                 # 如果当前帧与上一帧的差值大于100，保存当前累计得分，并重置累计得分
#                 if last_frame_in_round is not None and (frame - last_frame_in_round) > 100:
#                     round_scores.append(cumulative_score)  # 把当前得分保存到结果
#                     cumulative_score = 0  # 重置累计得分

#                 # 计算得分
#                 x_value = similarity  # 获取第二个数据作为x值
#                 # print(x_value)
#                 # print(type(relevant_frames_pose))
#                 # print(relevant_frames_pose)
#                 x_value_pose=relevant_frames_pose[frame][3]
#                 x_value_pose=calculate_score_pose(x_value_pose)
#                 # print(x_value_pose)
#                 score = x_value_pose*pow(x_value,0)
#                 # score = x_value_pose*1
#                 cumulative_score += score  # 累加得分
#                 processed_frames.add(frame)  # 标记该帧已经处理

#                 # 更新回合中的最后一帧
#                 last_frame_in_round = frame




#         # 更新上一个回合的结束帧
#         last_end_frame = end_frame

# # 如果最后一个回合没有添加得分，需要手动添加
# if cumulative_score > 0:
#     round_scores.append(cumulative_score)

# # # 2016-1需要这个
# # print(round_scores[19])
# score_to_move = round_scores[19]
# round_scores.pop(19)  # 删除第 19 个元素
# round_scores.append(score_to_move)  # 将元素添加到末尾


# def custom_sigmoid(x, theta_0=6, theta_1=-4.4):
#     return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5
# df['回合总分'] = round_scores
# print(df['回合总分'])
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})

# # 计算 "有效时长"
# df['有效时长'] = df['回合时长'] -df['回合尾多余帧']/25

# # # 定义网格搜索的参数范围
# # theta_0_values = np.linspace(3, 7, 9)  # 你可以调整这个范围和步长
# # theta_1_values = np.linspace(-7, -4, 16)  # 你可以调整这个范围和步长

# theta_0_values = np.arange(3, 7.1, 0.1)  # 间隔 0.1，包含 7.0
# theta_1_values = np.arange(-7, -4.1, 0.1)  # 间隔 0.1，包含 -4.0


# # 进行网格搜索
# for theta_0 in theta_0_values:
#     for theta_1 in theta_1_values:
#         xx=df['回合总分']+ df['是否回合被分割'] * 1000
#         xx=xx/(custom_sigmoid(df['有效时长'],theta_0,theta_1)*df['有效时长'])
#         # print(xx)
#         df['有效得分'] = xx

#                 # 按 "有效得分" 排序（降序：从高到低）
#         df_sorted = df.sort_values(by='有效得分', ascending=False)

#         # 保存排序后的结果到新的 Excel 文件
#         output_file_path = f"/home/haoge/ymq1/grid_x_1_pow1/x_grid_search_result_theta0_{theta_0}_theta1_{theta_1}.xlsx"
#         df_sorted.to_excel(output_file_path, index=False)

#         # 输出保存路径
#         # print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

# # theta_0=5.5
# # theta_1=-4.5
# # xx=df['回合总分']
# # xx=xx/(custom_sigmoid(df['回合时长'],theta_0,theta_1)*df['回合时长'])
# # print(xx)
# # df['有效得分'] = (xx + df['是否回合被分割'] * 1000) 

# #         # 按 "有效得分" 排序（降序：从高到低）
# # df_sorted = df.sort_values(by='有效得分', ascending=False)

# # # 保存排序后的结果到新的 Excel 文件
# # output_file_path = f"2016-3-x_{theta_0}_{theta_1}.xlsx"
# # df_sorted.to_excel(output_file_path, index=False)

# # # 输出保存路径
# # print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

# # 将回合得分保存到Excel文件中的“回合总分”这一列


# # # 保存新的Excel文件
# # df.to_excel('2016-1-mtest_with_scores_theta0_x_t6-44_20.xlsx', index=False)

# print("Processing completed.")