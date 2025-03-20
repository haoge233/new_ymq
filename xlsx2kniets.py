# import os
# import json
# import shutil
# import pandas as pd

# # 读取excel文件
# def read_xlsx(xlsx_path):
#     data = pd.read_excel(xlsx_path)
#     return data

# # 读取json文件并提取指定帧范围的骨骼点
# def extract_skeleton_data(json_file, frame_range):
#     with open(json_file, 'r') as f:
#         json_data = json.load(f)
    
#     skeleton_data = []
#     for frame_id in range(frame_range[0], frame_range[1] + 1):
#         if str(frame_id) in json_data:
#             if json_data[str(frame_id)]['bottom'] is not None:
#                 skeleton_data.append(json_data[str(frame_id)]['bottom'])  # 提取bottom骨骼点数据
#     return skeleton_data

# # 保存转化后的数据为Kinetics格式
# def save_kinetics_format(skeleton_data, output_json_file):
#     output_data = []
#     for frame_data in skeleton_data:
#         formatted_data = {
#             "frame_index": len(output_data),
#             "skeleton": [{"pose": [coord[0], coord[1]], "score": 1} for coord in frame_data]
#         }
#         output_data.append(formatted_data)
    
#     with open(output_json_file, 'w') as f:
#         json.dump(output_data, f, indent=4)

# # 根据xlsx文件中的比例分配数据到不同的文件夹
# def split_and_save_data(data, json_file, source_video_folder, output_root):
#     # 划分比例
#     split_ratios = [0.1, 0.2, 0.2, 0.2, 0.3]
    
#     # 创建目标文件夹
#     for i in range(6):
#         output_folder = os.path.join(output_root, str(i))
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
    
#     # 处理每一行（视频名称和得分）
#     video_count = len(data)
#     video_names = data['视频名称']
#     frame_scores = data['回合得分']
    
#     # 计算划分点
#     split_points = [int(video_count * sum(split_ratios[:i])) for i in range(1, len(split_ratios) + 1)]
#     split_indexes = [range(0, split_points[0])]
#     for i in range(1, len(split_ratios)):
#         split_indexes.append(range(split_points[i-1], split_points[i]))
#     split_indexes.append(range(split_points[-1], video_count))
    
#     # 遍历每个视频，处理其骨骼数据
#     for idx, video_name in enumerate(video_names):
#         frame_range_str = video_name.replace('.mp4', '')
#         # print(video_name.split('_'))
#         # frame_range = tuple(map(int, video_name.split('_')[1].split('_')))
#         frame_range = tuple(map(int, frame_range_str.split('_')[1:3]))
#         print(frame_range)
        
#         # 提取骨骼数据
#         skeleton_data = extract_skeleton_data(json_file, frame_range)
#         print(skeleton_data)
        
#         # 保存为Kinetics格式
#         output_json_file = os.path.join(output_root, str(idx % 6), f"{video_name.replace('.mp4', '.json')}")
#         save_kinetics_format(skeleton_data, output_json_file)
        
#         # 复制视频文件到目标文件夹
#         video_file = os.path.join(source_video_folder, video_name)
#         target_video_folder = os.path.join(output_root, str(idx % 6))
#         shutil.copy(video_file, os.path.join(target_video_folder, video_name))

# # 主函数
# def main():
#     xlsx_path = '/home/haoge/ymq1/2016-3-round-score_D40.xlsx' # 修改为xlsx文件的路径
#     json_file = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'  # JSON文件所在的文件夹路径
#     source_video_folder = '/home/haoge/ymq1/2016-3-lround_30_D40'  # 视频文件夹路径
#     output_root = 'myjsondata'  # 修改为输出文件夹路径
#     # xlsx_file = '/home/haoge/ymq1/2016-3-round-score_D40.xlsx'  # 你的xlsx文件路径
#     # output_dir = 'myjsondata'  # 输出文件夹路径
#     # data_dir = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'  # JSON文件所在的文件夹路径
#     # video_folder = '/home/haoge/ymq1/2016-3-lround_30_D40'  # 视频文件夹路径
    
#     # 读取xlsx文件
#     data = read_xlsx(xlsx_path)
    
#     # 分配并保存数据
#     split_and_save_data(data, json_file, source_video_folder, output_root)

# if __name__ == "__main__":
#     main()

import os
import json
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

def load_xlsx_data(xlsx_path):
    """
    读取Excel文件，获取视频名称及其对应的回合得分
    """
    df = pd.read_excel(xlsx_path)
    return df

def extract_frame_range_from_filename(filename):
    """
    从视频文件名中提取帧范围
    示例：segment_7983_8035.mp4 -> (7983, 8035)
    """
    parts = filename.replace('segment_', '').replace('.mp4', '').split('_')
    return int(parts[0]), int(parts[1])

def extract_skeleton_data_from_json(json_data, start_frame, end_frame):
    """
    从JSON数据中提取指定范围的骨骼点数据
    """
    extracted_data = []
    for frame_idx in range(start_frame, end_frame + 1):
        if str(frame_idx) in json_data:
            frame_data = json_data[str(frame_idx)]
            extracted_data.append({
                'frame_index': frame_idx-start_frame+1,
                'bottom': frame_data.get('bottom', [])
            })
    return extracted_data

def convert_to_kinetics_format(extracted_data):
    """
    将提取的骨骼点数据转化为Kinetics数据集格式
    """
    kinetics_data = []
    for frame_data in extracted_data:
        bottom_points = frame_data['bottom']
        if bottom_points is not None:
            skeleton_data = {
                'frame_index': frame_data['frame_index'],
                'skeleton': [{
                    'pose': np.array(bottom_points).flatten().tolist(),
                    'score': [1] * len(bottom_points)
                }]
            }
            kinetics_data.append(skeleton_data)
        else:
            skeleton_data = {
                'frame_index': frame_data['frame_index'],
                'skeleton': []
            }
            kinetics_data.append(skeleton_data)
    return kinetics_data

def save_data_to_json(output_path, data):
    """
    将数据保存为JSON文件
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def split_data(data, ratio=[0.1, 0.2, 0.2, 0.2, 0.3]):
    """
    根据比例划分数据
    """
    total = len(data)
    idx_0 = int(total * ratio[0])
    idx_1 = idx_0 + int(total * ratio[1])
    idx_2 = idx_1 + int(total * ratio[2])
    idx_3 = idx_2 + int(total * ratio[3])

    return {
        0: data[:idx_0],
        1: data[idx_0:idx_1],
        2: data[idx_1:idx_2],
        3: data[idx_2:idx_3],
        4: data[idx_3:]
    }

def process_videos_and_json(xlsx_path, json_path, video_dir, output_dir, json_output_dir):
    """
    处理视频和JSON数据，按比例划分并保存
    """
    # 加载xlsx数据
    df = load_xlsx_data(xlsx_path)
    video_data = df['视频名称'].tolist()
    
    # 加载JSON数据
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 按照比例划分数据
    split_video_data = split_data(video_data)

    # 输出文件夹
    video_output_dirs = [os.path.join(output_dir, str(i)) for i in range(5)]
    json_output_dirs = [os.path.join(json_output_dir, str(i)) for i in range(5)]
    
    # 创建文件夹
    for dir in video_output_dirs + json_output_dirs:
        os.makedirs(dir, exist_ok=True)

    # 处理每个视频
    for i in tqdm(range(5), desc="Processing"):
        # 获取当前文件夹的视频和json数据
        video_files = split_video_data[i]
        for video_file in video_files:
            # 获取视频文件和JSON文件路径
            video_filename = video_file.strip()
            frame_range = extract_frame_range_from_filename(video_filename)
            video_path = os.path.join(video_dir, video_filename)
            
            # 提取对应的骨骼数据
            extracted_data = extract_skeleton_data_from_json(json_data, *frame_range)

            # 转换为Kinetics格式
            kinetics_data = convert_to_kinetics_format(extracted_data)

            # 保存转化后的骨骼数据到json
            json_filename = video_filename.replace('.mp4', '_kinetics.json')
            json_output_path = os.path.join(json_output_dirs[i], json_filename)
            save_data_to_json(json_output_path, kinetics_data)

            # 复制视频文件到对应的文件夹
            video_output_path = os.path.join(video_output_dirs[i], video_filename)
            copyfile(video_path, video_output_path)

# 调用函数进行处理
xlsx_path = '/home/haoge/ymq1/2016-3-round-score_D40.xlsx'
json_path = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'
video_dir = '/home/haoge/ymq1/2016-3-lround_30_D40'
output_dir = 'myvideosdata' 
json_output_dir = 'myjsondata' 



process_videos_and_json(xlsx_path, json_path, video_dir, output_dir, json_output_dir)
