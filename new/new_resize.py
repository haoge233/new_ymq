# #  reize
# import os
# import json
# import numpy as np

# def resize_keypoints(keypoints, original_width, original_height, new_width, new_height):
#     """
#     将关键点坐标从原始分辨率缩放到新的分辨率。
#     假设每个关键点包含 (x, y) 这样的数据结构。
#     """
#     resized_keypoints = []
#     for i in range(0, len(keypoints), 2):  # 每个点由 (x, y) 组成
#         new_x = keypoints[i] * (new_width / original_width)
#         new_y = keypoints[i + 1] * (new_height / original_height)
#         resized_keypoints.extend([new_x, new_y])
#     return resized_keypoints

# def process_json_files(input_json_dir, output_json_dir, original_width, original_height, new_width, new_height):
#     """
#     处理所有JSON文件，调整关键点坐标并添加 label 和 label_index
#     """
#     # 获取所有类别文件夹
#     category_folders = [f for f in os.listdir(input_json_dir) if os.path.isdir(os.path.join(input_json_dir, f))]
    
#     for category_folder in category_folders:
#         # 获取类别标签（文件夹名）
#         label = label_index = int(category_folder)
        
#         # 获取当前类别文件夹中的所有JSON文件
#         json_files = [f for f in os.listdir(os.path.join(input_json_dir, category_folder)) if f.endswith('.json')]
        
#         for json_file in json_files:
#             input_path = os.path.join(input_json_dir, category_folder, json_file)
#             output_path = os.path.join(output_json_dir, category_folder, json_file)
            
#             # 创建输出文件夹（如果不存在）
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
#             with open(input_path, 'r') as f:
#                 json_data = json.load(f)
            
#             updated_data = []
#             for entry in json_data:
#                 # 提取骨骼点数据（可能存在多个人）
#                 skeletons = entry.get('skeleton', [])
                
#                 resized_skeletons = []
#                 for skeleton in skeletons:
#                     # 获取pose并resize
#                     if 'pose' in skeleton and skeleton['pose']:
#                         resized_keypoints = resize_keypoints(
#                             skeleton['pose'], 
#                             original_width, 
#                             original_height, 
#                             new_width, 
#                             new_height
#                         )
#                         resized_skeleton = {
#                             'pose': resized_keypoints,
#                             'score': [1.0] * len(resized_keypoints)  # 置信度全部置为1.0
#                         }
#                         resized_skeletons.append(resized_skeleton)
                
#                 # 更新当前帧的数据
#                 updated_entry = {
#                     'frame_index': entry['frame_index'],
#                     'skeleton': resized_skeletons
#                 }
#                 updated_data.append(updated_entry)

#             # 为整个数据加上标签和 label_index
#             final_data = {
#                 "data": updated_data,
#                 "label": label,
#                 "label_index": label_index
#             }

#             # 保存更新后的数据到新的文件
#             with open(output_path, 'w') as f:
#                 json.dump(final_data, f, indent=4)

# # 设置参数
# input_json_dir = '/home/haoge/ymq1/o_new_myjsondata'  # 输入的JSON文件夹路径
# output_json_dir = '/home/haoge/ymq1/o_nosize_myjsondataresize'  # 输出的JSON文件夹路径
# original_width = 1920
# original_height = 1080
# # new_width = 340
# # new_height = 256
# # 下面是不size的改动---
# new_width = 1920
# new_height = 1080

# # 处理所有JSON文件
# process_json_files(input_json_dir, output_json_dir, original_width, original_height, new_width, new_height)

# # 7：2：1划分数据集   文件夹不包含类别
# import os
# import json
# import random
# import shutil

# def process_json_files(input_json_dir, output_json_dir):
#     """
#     处理所有JSON文件，按7:2:1的比例在每个类别中进行划分。
#     """
#     # 获取所有类别文件夹
#     category_folders = [f for f in os.listdir(input_json_dir) if os.path.isdir(os.path.join(input_json_dir, f))]
    
#     # 创建输出文件夹
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_train'), exist_ok=True)
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_val'), exist_ok=True)
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_test'), exist_ok=True)
    
#     for category_folder in category_folders:
#         category_path = os.path.join(input_json_dir, category_folder)
#         json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
        
#         # 打乱当前类别的JSON文件顺序
#         random.shuffle(json_files)
        
#         # 按比例划分 70% 训练集，20% 验证集，10% 测试集
#         total_files = len(json_files)
#         train_size = int(0.7 * total_files)
#         val_size = int(0.2 * total_files)

#         train_files = json_files[:train_size]
#         val_files = json_files[train_size:train_size + val_size]
#         test_files = json_files[train_size + val_size:]

#         # 输出路径设置
#         target_folders = {
#             'train': os.path.join(output_json_dir, 'kinetics_train'),
#             'val': os.path.join(output_json_dir, 'kinetics_val'),
#             'test': os.path.join(output_json_dir, 'kinetics_test')
#         }

#         # 遍历划分好的文件，写入对应路径
#         for json_file in train_files:
#             move_json(category_path, json_file, target_folders['train'], category_folder)
        
#         for json_file in val_files:
#             move_json(category_path, json_file, target_folders['val'], category_folder)
        
#         for json_file in test_files:
#             move_json(category_path, json_file, target_folders['test'], category_folder)

# def move_json(source_dir, json_file, target_folder, category):
#     """
#     读取 JSON 文件，增加标签信息并保存到目标目录。
#     """
#     source_path = os.path.join(source_dir, json_file)
#     target_path = os.path.join(target_folder, json_file)

#     # 读取原始 JSON 数据
#     with open(source_path, 'r') as f:
#         json_data = json.load(f)
    
#     # 为每个数据添加 label 和 label_index
#     label = int(category)  # 文件夹名即为类别标签
#     json_data['label'] = label
#     json_data['label_index'] = label

#     # 保存到目标文件夹
#     with open(target_path, 'w') as f:
#         json.dump(json_data, f, indent=4)

# def create_categories_mapping(categories_folder):
#     """
#     获取类别文件夹的名称，返回每个类别的标签。
#     """
#     categories = {}
#     for foldername in os.listdir(categories_folder):
#         if os.path.isdir(os.path.join(categories_folder, foldername)):
#             category_name = foldername
#             label_index = int(category_name)
#             categories[foldername] = label_index
#     return categories

# # 设置参数
# input_json_dir = '/home/haoge/ymq1/o_nosize_myjsondataresize'  # 输入的JSON文件夹路径
# output_json_dir = '/home/haoge/ymq1/o_nosize_mykinetdata'      # 输出的JSON文件夹路径

# # 创建类别映射
# categories = create_categories_mapping(input_json_dir)

# # 处理所有JSON文件并划分数据集
# process_json_files(input_json_dir, output_json_dir)


import os
import json
import shutil

def modify_pose_coordinates(json_data):
    """
    更改 pose 中第 5 组点的坐标为第 12 组点和第 13 组点的中点，
    并将 frame_index 从 0 开始调整为从 1 开始。
    """
    for entry in json_data['data']:
        # 将 frame_index +1，使其从 1 开始
        entry['frame_index'] += 1
        
        skeletons = entry.get('skeleton', [])
        for skel in skeletons:
            pose = skel.get('pose', [])
            score = skel.get('score', [])
            
            # 确保 pose 数据长度足够（每个点包含 x, y 坐标）
            if len(pose) >= 26:  # 13 组点需要 26 个数（x, y）
                # 读取第 12 组和第 13 组的坐标
                point_12_x = pose[22]
                point_12_y = pose[23]
                point_13_x = pose[24]
                point_13_y = pose[25]
                
                # 计算中点
                midpoint_x = (point_12_x + point_13_x) / 2
                midpoint_y = (point_12_y + point_13_y) / 2
                
                # 将第 5 组点的坐标替换为中点
                pose[8] = midpoint_x
                pose[9] = midpoint_y

                # 如果 score 的长度不一致，进行截断或填充
                if len(score) < 17:
                    score = score[:17] + [1] * (17 - len(score))
                else:
                    score = score[:17]

                # 更新到 skeleton 数据中
                skel['pose'] = pose
                skel['score'] = score
    
    return json_data

def process_json_files(input_json_dir, output_json_dir):
    """
    处理文件夹下的所有JSON文件，修改每个人的第5组点的坐标，并保存到新结构的文件夹。
    """
    for root, _, files in os.walk(input_json_dir):
        for file in files:
            if file.endswith('.json'):
                # 构造输入和输出文件的完整路径
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_json_dir)
                output_folder = os.path.join(output_json_dir, relative_path)
                
                # 创建输出目录
                os.makedirs(output_folder, exist_ok=True)
                
                output_path = os.path.join(output_folder, file)
                
                # 读取原始 JSON 数据
                with open(input_path, 'r') as f:
                    json_data = json.load(f)
                
                # 修改 pose 坐标和 frame_index
                modified_data = modify_pose_coordinates(json_data)
                
                # 保存修改后的数据到新文件夹
                with open(output_path, 'w') as f:
                    json.dump(modified_data, f, indent=4)

# 设置参数
input_json_dir = '/home/haoge/ymq1/o_nosize_mykinetdata'     # 输入的 JSON 文件夹路径
output_json_dir = '/home/haoge/ymq1/o_nosize_mykinetdata2'    # 输出的 JSON 文件夹路径

# 处理所有JSON文件并修改坐标
process_json_files(input_json_dir, output_json_dir)
