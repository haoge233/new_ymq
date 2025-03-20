import os
import json
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm
from scipy.stats import mode

def load_xlsx_data(xlsx_path):
    """
    读取Excel文件，获取视频名称及其对应的标注
    """
    df = pd.read_excel(xlsx_path)
    
    # 创建一个字典，将视频名称与标注列表关联
    video_labels = {}
    for _, row in df.iterrows():
        video_name = row['视频名称'].strip()
        labels = row.iloc[1:].tolist()  # 获取除“视频名称”列之外的其他列
        video_labels[video_name] = labels
    
    return video_labels

def load_split_xlsx_data(split_xlsx_path):
    """
    读取第二个Excel文件（用于划分类别）
    格式：视频名称 + 多个标注值
    """
    df = pd.read_excel(split_xlsx_path)
    
    split_labels = {}
    for _, row in df.iterrows():
        video_name = row['视频名称'].strip()
        
        # 去除非法值和空值
        labels = []
        for label in row.iloc[2:]:
            try:
                label = int(float(label))
                # if 0 <= label <= 5:  # 仅保留合法标注（假设 0~5 是合法标注）
                labels.append(label)
            except (ValueError, TypeError):
                pass
        
        if len(labels) > 0:
            split_labels[video_name] = labels
    
    return split_labels

def extract_frame_range_from_filename(filename):
    """
    从视频文件名中提取帧范围
    示例：segment_7983_8035.mp4 -> (7983, 8035)
    """
    parts = filename.replace('segment_', '').replace('.mp4', '').split('_')
    return int(parts[0]), int(parts[1])

def extract_skeleton_data_from_json(json_data, start_frame, end_frame):
    """
    从JSON数据中提取top和bottom的骨骼数据
    """
    extracted_data = []
    for frame_idx in range(start_frame, end_frame + 1):
        if str(frame_idx) in json_data:
            frame_data = json_data[str(frame_idx)]
            extracted_data.append({
                'frame_index': frame_idx - start_frame,
                'bottom': frame_data.get('bottom', []),
                'top': frame_data.get('top', [])
            })
    return extracted_data

def convert_to_kinetics_format(extracted_data):
    """
    将top和bottom数据合并并转换为Kinetics格式
    """
    kinetics_data = []
    
    for frame_data in extracted_data:
        frame_index = frame_data['frame_index']
        top_points = frame_data['top']
        bottom_points = frame_data['bottom']
        
        skeleton = []

        # 合并 top 和 bottom 的 pose 和 score
        if bottom_points:
            pose = np.array(bottom_points).flatten().tolist()
            score = [1.0] * len(pose)
            skeleton.append({
                'pose': pose,
                'score': score
            })

        if top_points:
            pose = np.array(top_points).flatten().tolist()
            score = [1.0] * len(pose)
            skeleton.append({
                'pose': pose,
                'score': score
            })

        # 组装单帧数据
        frame_kinetics_data = {
            'frame_index': frame_index,
            'skeleton': skeleton
        }
        kinetics_data.append(frame_kinetics_data)

    return kinetics_data

def save_data_to_json(output_path, data):
    """
    保存数据到JSON文件
    """
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def process_videos_and_json(xlsx_path, split_xlsx_path, json_path, video_dir, output_dir, json_output_dir):
    """
    处理视频和JSON数据，按Excel文件中的标注众数进行划分
    """
    # 加载第一个Excel文件（视频和帧范围信息）
    video_labels = load_xlsx_data(xlsx_path)

    # 加载第二个Excel文件（标注信息）
    split_labels = load_split_xlsx_data(split_xlsx_path)
    print(split_labels)

    # 加载JSON数据
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # 创建输出文件夹（五个类别）
    output_dirs = [os.path.join(output_dir, str(i)) for i in range(5)]
    json_output_dirs = [os.path.join(json_output_dir, str(i)) for i in range(5)]

    for dir in output_dirs + json_output_dirs:
        os.makedirs(dir, exist_ok=True)

    # 处理每个视频
    for video_file in tqdm(os.listdir(video_dir), desc="Processing"):
        video_filename = video_file.strip()
        frame_range = extract_frame_range_from_filename(video_filename)
        video_path = os.path.join(video_dir, video_filename)

        if video_filename in video_labels and video_filename in split_labels:
            # 获取标注众数（mode）
            labels = split_labels[video_filename]
            print(labels)
            # print(mode(labels,keepdims=True).mode[0])
            try:
                mode_label = int(mode(labels,keepdims=True).mode[0])  # 取众数
            except:
                print(f"⚠️ Warning: '{video_filename}' 的众数为空，跳过处理。")
                continue

            # 提取top和bottom数据
            extracted_data = extract_skeleton_data_from_json(json_data, *frame_range)

            # 转换为Kinetics格式
            kinetics_data = convert_to_kinetics_format(extracted_data)

            # 保存数据
            json_filename = video_filename.replace('.mp4', '_kinetics.json')
            json_output_path = os.path.join(json_output_dirs[mode_label], json_filename)
            save_data_to_json(json_output_path, kinetics_data)

            # 复制视频文件
            video_output_path = os.path.join(output_dirs[mode_label], video_filename)
            if os.path.exists(video_path):
                copyfile(video_path, video_output_path)

# 调用函数
xlsx_path = '/home/haoge/ymq1/2016-3-xlsx/2016-3-round-score_D40.xlsx'
split_xlsx_path = '/home/haoge/ymq1/round-grade1/精彩程度5分类_标注汇总.xlsx'
json_path = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'
video_dir = '/home/haoge/ymq1/2016-3-lround_30_D40'
output_dir = 'new_myvideosdata'
json_output_dir = 'new_myjsondata'

process_videos_and_json(xlsx_path, split_xlsx_path, json_path, video_dir, output_dir, json_output_dir)

