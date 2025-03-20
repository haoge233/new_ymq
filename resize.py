# resize
# import os
# import json
# import numpy as np

# def resize_keypoints(keypoints, original_width, original_height, new_width, new_height):
#     """
#     将关键点坐标从原始分辨率缩放到新的分辨率。
#     假设每个关键点包含 (x, y, confidence) 这样的数据结构。
#     """
#     resized_keypoints = []
#     # print(keypoints)
#     for i in range(0, len(keypoints), 2):  # 每个点有3个元素（x, y, confidence）
#         new_x = keypoints[i] * (new_width / original_width)
#         new_y = keypoints[i+1] * (new_height / original_height)
#         resized_keypoints.extend([new_x, new_y])  # 保留原始的置信度等信息
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
#                 # 提取骨骼点数据（假设数据是以 'skeleton' 键存在）
#                 skeleton_points = entry.get('skeleton', [])
#                 print(skeleton_points)
                
#                 # 将每个框架中的关键点进行 resize
#                 if len(skeleton_points)>0:
#                     resized_keypoints = resize_keypoints(skeleton_points[0]['pose'], original_width, original_height, new_width, new_height)
                    
#                     # 更新当前骨骼数据
#                     updated_entry = {
#                         'frame_index': entry['frame_index'],
#                         'skeleton': [{
#                             'pose': resized_keypoints,
#                             'score': [1] * len(resized_keypoints)  # 假设置信度全部为 1
#                         }]
#                     }
#                 else:
#                     updated_entry = {
#                         'frame_index': entry['frame_index'],
#                         'skeleton': []
#                     }
                
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
# input_json_dir = '/home/haoge/ymq1/myjsondata'  # 输入的JSON文件夹路径
# output_json_dir = '/home/haoge/ymq1/myjsondataresize'  # 输出的JSON文件夹路径
# original_width = 1920
# original_height = 1080
# new_width = 340
# new_height = 256

# # 处理所有JSON文件
# process_json_files(input_json_dir, output_json_dir, original_width, original_height, new_width, new_height)

# 7：2：1划分数据集   文件夹包含类别
# import os
# import shutil
# import random
# import json

# def process_json_files(input_json_dir, output_json_dir):
#     """
#     处理所有JSON文件，按7:2:1比例划分文件到不同的数据集
#     """
#     # 获取所有类别文件夹
#     category_folders = [f for f in os.listdir(input_json_dir) if os.path.isdir(os.path.join(input_json_dir, f))]
    
#     for category_folder in category_folders:
#         # 获取类别标签（文件夹名）
#         label = label_index = int(category_folder)
        
#         # 获取当前类别文件夹中的所有JSON文件
#         json_files = [f for f in os.listdir(os.path.join(input_json_dir, category_folder)) if f.endswith('.json')]
        
#         # 打乱顺序，便于按比例划分
#         random.shuffle(json_files)
        
#         # 划分比例：70% 训练集，20% 验证集，10% 测试集
#         total_files = len(json_files)
#         train_size = int(0.7 * total_files)
#         val_size = int(0.2 * total_files)
        
#         train_files = json_files[:train_size]
#         val_files = json_files[train_size:train_size+val_size]
#         test_files = json_files[train_size+val_size:]
        
#         # 创建输出文件夹
#         train_folder = os.path.join(output_json_dir, 'kinetics_train', category_folder)
#         val_folder = os.path.join(output_json_dir, 'kinetics_val', category_folder)
#         test_folder = os.path.join(output_json_dir, 'kinetics_test', category_folder)
        
#         os.makedirs(train_folder, exist_ok=True)
#         os.makedirs(val_folder, exist_ok=True)
#         os.makedirs(test_folder, exist_ok=True)
        
#         # 复制文件到对应文件夹
#         for json_file in train_files + val_files + test_files:
#             input_path = os.path.join(input_json_dir, category_folder, json_file)
#             output_path = None
            
#             if json_file in train_files:
#                 output_path = os.path.join(train_folder, json_file)
#             elif json_file in val_files:
#                 output_path = os.path.join(val_folder, json_file)
#             elif json_file in test_files:
#                 output_path = os.path.join(test_folder, json_file)
            
#             # 读取原始JSON文件
#             with open(input_path, 'r') as f:
#                 json_data = json.load(f)
            
#             # # 为每个数据添加 label 和 label_index
#             # json_data['label'] = label
#             # json_data['label_index'] = label_index
            
#             # 保存更新后的数据到新的文件
#             with open(output_path, 'w') as f:
#                 json.dump(json_data, f, indent=4)

# def create_categories_mapping(categories_folder):
#     """
#     创建类别映射，假设类别文件夹为 0, 1, 2, ..., n。
#     """
#     categories = {}
#     for foldername in os.listdir(categories_folder):
#         if os.path.isdir(os.path.join(categories_folder, foldername)):
#             category_name = foldername  # 类别名就是文件夹的名称
#             label_index = int(category_name)  # 假设文件夹名是数字
#             categories[foldername] = label_index
#     return categories

# # 设置参数
# input_json_dir = '/home/haoge/ymq1/myjsondataresize'  # 输入的JSON文件夹路径
# output_json_dir = '/home/haoge/ymq1/mykinetdata'  # 输出的JSON文件夹路径

# # 创建类别映射
# categories = create_categories_mapping(input_json_dir)

# # 处理所有JSON文件并划分数据集
# process_json_files(input_json_dir, output_json_dir)

# 7：2：1划分数据集   文件夹不包含类别
# import os
# import shutil
# import random
# import json

# def process_json_files(input_json_dir, output_json_dir):
#     """
#     处理所有JSON文件，按7:2:1比例划分文件到不同的数据集。
#     所有JSON文件直接分配到 `kinetics_train`, `kinetics_val`, `kinetics_test` 文件夹。
#     """
#     # 获取所有类别文件夹（假设这些文件夹的名字是 0, 1, 2, ...）
#     category_folders = [f for f in os.listdir(input_json_dir) if os.path.isdir(os.path.join(input_json_dir, f))]
    
#     # 存储所有的 JSON 文件
#     all_json_files = []
    
#     # 遍历每个类别文件夹，获取其中的所有 JSON 文件
#     for category_folder in category_folders:
#         category_path = os.path.join(input_json_dir, category_folder)
#         json_files = [f for f in os.listdir(category_path) if f.endswith('.json')]
#         all_json_files.extend([os.path.join(category_path, f) for f in json_files])
    
#     # 打乱所有的 JSON 文件顺序
#     random.shuffle(all_json_files)
    
#     # 按照比例划分数据集：70% 训练集，20% 验证集，10% 测试集
#     total_files = len(all_json_files)
#     train_size = int(0.7 * total_files)
#     val_size = int(0.2 * total_files)
    
#     train_files = all_json_files[:train_size]
#     val_files = all_json_files[train_size:train_size+val_size]
#     test_files = all_json_files[train_size+val_size:]
    
#     # 创建输出文件夹
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_train'), exist_ok=True)
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_val'), exist_ok=True)
#     os.makedirs(os.path.join(output_json_dir, 'kinetics_test'), exist_ok=True)
    
#     # 将文件复制到对应文件夹，并添加 label 和 label_index
#     for json_file in train_files + val_files + test_files:
#         # 获取目标文件夹
#         if json_file in train_files:
#             target_folder = os.path.join(output_json_dir, 'kinetics_train')
#         elif json_file in val_files:
#             target_folder = os.path.join(output_json_dir, 'kinetics_val')
#         elif json_file in test_files:
#             target_folder = os.path.join(output_json_dir, 'kinetics_test')
        
#         # 读取原始 JSON 数据
#         with open(json_file, 'r') as f:
#             json_data = json.load(f)
        
#         # # 为每个数据添加 label 和 label_index
#         # # 标签从文件夹名称中获取，文件夹名即为类别标签
#         # category_folder_name = os.path.basename(os.path.dirname(json_file))
#         # label = int(category_folder_name)  # 类别标签
        
#         # json_data['label'] = label
#         # json_data['label_index'] = label
        
#         # 保存更新后的数据到目标文件夹
#         target_path = os.path.join(target_folder, os.path.basename(json_file))
#         with open(target_path, 'w') as f:
#             json.dump(json_data, f, indent=4)

# def create_categories_mapping(categories_folder):
#     """
#     获取类别文件夹的名称，返回每个类别的标签。
#     """
#     categories = {}
#     for foldername in os.listdir(categories_folder):
#         if os.path.isdir(os.path.join(categories_folder, foldername)):
#             category_name = foldername  # 类别名就是文件夹的名称
#             label_index = int(category_name)  # 假设文件夹名是数字
#             categories[foldername] = label_index
#     return categories

# # 设置参数
# # input_json_dir = 'path/to/your/data'  # 输入的JSON文件夹路径
# # output_json_dir = 'path/to/your/output'  # 输出的JSON文件夹路径
# input_json_dir = '/home/haoge/ymq1/myjsondataresize'  # 输入的JSON文件夹路径
# output_json_dir = '/home/haoge/ymq1/mykinetdata'  # 输出的JSON文件夹路径

# # 创建类别映射
# categories = create_categories_mapping(input_json_dir)

# # 处理所有JSON文件并划分数据集
# process_json_files(input_json_dir, output_json_dir)


import os
import json
import shutil

def modify_pose_coordinates(json_data):
    """
    更改 pose 中第 5 组点的坐标为第 12 组点和第 13 组点的中点。
    """

    for entry in json_data['data']:
        # print(entry)
        skeleton = entry.get('skeleton', [])
        for skel in skeleton:
            pose = skel.get('pose', [])
            score = skel.get('score', [])
            # 每个 pose 包含 17 组点，每组点有 34 个数值（x, y 坐标）
            if len(pose) >= 13:
                # # 第 12 组和第 13 组的点的坐标（注意索引从 0 开始，所以是 11 和 12）
                # point_12 = pose[22]  # 第 12 组点 (索引从 0 开始)
                # point_13 = pose[24]  # 第 13 组点 (索引从 0 开始)
                
                # 计算中点
                midpoint_x = (pose[22] + pose[24]) / 2
                midpoint_y = (pose[23] + pose[25]) / 2
                
                # 将第 5 组点的坐标更改为中点
                pose[8] = midpoint_x
                pose[9] = midpoint_y
                skel['score'] = score[:17]  # 保留 score 数组的前 17 个数
                # [midpoint_x, midpoint_y] + pose[4][2:]  # 复制原有的其他信息（如置信度等）

    return json_data

def process_json_files(input_json_dir, output_json_dir):
    """
    处理文件夹下的所有JSON文件，修改第5组点的坐标为第12组和第13组的中点，并保存到新结构的文件夹。
    """
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(input_json_dir):
        for file in files:
            if file.endswith('.json'):
                # 构造输入和输出文件的完整路径
                input_path = os.path.join(root, file)
                print(input_path)
                
                # 计算相对路径
                relative_path = os.path.relpath(root, input_json_dir)
                output_folder = os.path.join(output_json_dir, relative_path)
                
                # 确保输出文件夹存在
                os.makedirs(output_folder, exist_ok=True)
                
                # 构造输出路径
                output_path = os.path.join(output_folder, file)
                
                # 读取原始JSON数据
                with open(input_path, 'r') as f:
                    json_data = json.load(f)
                
                # 修改pose坐标
                modified_data = modify_pose_coordinates(json_data)
                
                # 保存修改后的数据到新的文件夹
                with open(output_path, 'w') as f:
                    json.dump(modified_data, f, indent=4)




# 设置参数
input_json_dir = '/home/haoge/ymq1/mykinetdata'  # 输入的JSON文件夹路径
output_json_dir = '/home/haoge/ymq1/mykinetdata2'  # 输出的JSON文件夹路径

# 处理所有JSON文件并修改坐标
process_json_files(input_json_dir, output_json_dir)
