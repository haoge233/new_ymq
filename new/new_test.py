# import numpy as np

# # 加载 .npy 文件
# data = np.load('/home/haoge/PaddleVideo/core_poses_2.npy')
# # 打印数组的基本信息
# print("数组形状:", data.shape)  # 数组的维度结构
# print("数组数据类型:", data.dtype)  # 数组元素的数据类型
# print("数组维度:", data.ndim)  # 数组的维度数
# print("前 5 个元素:", data)


# import pickle

# # 从文件中读取并反序列化对象
# with open("/home/haoge/ymq1/o_nosize_kinetics-skeleton/test_label.pkl", "rb") as f:
#     loaded_data = pickle.load(f)

# print(loaded_data)  # 输出: {'name': 'Alice', 'age': 25, 'city': 'New York'}

# import pandas as pd
# import numpy as np
# from scipy.stats import mode

# def get_closest_mode(labels, my_label):
#     """
#     获取与我的标注最接近的众数，若有多个相同距离，则选较大值
#     """
#     mode_result = mode(labels, keepdims=True)
#     mode_values = mode_result.mode  # 可能是多个众数

#     if len(mode_values) > 1:  # 多个众数情况
#         return max(mode_values, key=lambda x: (abs(x - my_label), x))  # 先按距离排序，再按数值大小排序
#     return mode_values[0]  # 只有一个众数时直接返回

# def process_xlsx_and_add_column(xlsx_path, output_xlsx_path):
#     """
#     读取Excel文件，计算新的标注结果，并添加新列保存
#     """
#     # 读取Excel文件
#     df = pd.read_excel(xlsx_path)

#     # 假设第一列是 '视频名称'，第二列是 '我的标注'，后续列是其他标注
#     df['计算标注结果'] = None  # 初始化新列

#     for idx, row in df.iterrows():
#         try:
#             my_label = int(row.iloc[1])  # 读取“我的标注”
#             labels = [int(x) for x in row.iloc[2:] if not pd.isna(x)]  # 读取其他标注，并去除空值
#             if labels:
#                 computed_label = get_closest_mode(labels, my_label)
#                 df.at[idx, '计算标注结果'] = computed_label  # 存入新列
#         except Exception as e:
#             print(f"⚠️ 处理行 {idx} 时出错: {e}")

#     # 保存新的Excel文件
#     df.to_excel(output_xlsx_path, index=False)
#     print(f"✅ 处理完成，新文件已保存: {output_xlsx_path}")

# # 调用函数
# xlsx_path = "/home/haoge/ymq1/round-grade/round-grade/精彩程度5分类_标注汇总.xlsx"
# output_xlsx_path = "/home/haoge/ymq1/round-grade/round-grade/精彩程度5分类_标注汇总.xlsx_with_result.xlsx"

# process_xlsx_and_add_column(xlsx_path, output_xlsx_path)


import paddle

# 创建测试数据
batch_size = 32
fusion_size = 2
channels = 256

# 模拟 weights，形状为 [batch_size, fusion_size]
weights = paddle.rand([batch_size, fusion_size])

# 还原形状为 [batch_size, fusion_size, 1, 1]
weights = weights.unsqueeze(-1).unsqueeze(-1)  
print("weights.shape:", weights.shape)  # 预期: [32, 2, 1, 1]

# 模拟 features，features 是一个列表，包含 2 个 shape=[batch_size, channels, 1, 1] 的 Tensor
features = [paddle.rand([batch_size, channels, 1, 1]) for _ in range(fusion_size)]
print("features[0].shape:", features[0].shape)  # 预期: [32, 256, 1, 1]

# 按照 axis=1 拆分 weights
w1 = weights.split(1, axis=1)
print("w1[0].shape:", w1[0].shape)  # 预期: [32, 1, 1, 1]

# 检查拆分后是否符合预期
assert len(w1) == fusion_size, "Split 结果数量不正确"
assert w1[0].shape == [batch_size, 1, 1, 1], "Split 结果形状不匹配"

# 计算加权和
output = sum(w * f for w, f in zip(w1, features))
print("output.shape:", output.shape)  # 预期: [32, 256, 1, 1]



# import numpy as np
# import pandas as pd
# import glob
# import os

# # 读取指定文件夹下的2-7分类的Excel文件
# folder_path = "/home/haoge/ymq1/round-grade/round-grade"  # 替换为实际路径
# file_paths = glob.glob(os.path.join(folder_path, "精彩程度*分类_标注汇总.xlsx"))

# # 遍历 2-7 分类的文件
# for file in sorted(file_paths, key=lambda x: int(x.split("精彩程度")[1][0])):  # 按分类数字排序
#     df = pd.read_excel(file)  # 读取 Excel 文件
#     annotations = df.iloc[:, 2:].values  # 取所有标注数据（去掉视频名称列）

#     categories = np.unique(annotations.astype(int))  # 获取所有唯一类别
#     num_categories = len(categories)  # 统计类别数量
#     print(num_categories)

#     # 计算每个标注人员的类别方差
#     annotator_variances = []
#     for i in range(annotations.shape[1]):  # 遍历每个标注员
#         counts = np.bincount(annotations[:, i], minlength=num_categories)  # 计算该标注员的类别分布
#         print(counts)
#         variance = np.var(counts)  # 计算方差
#         annotator_variances.append(variance)

#     # 计算每个标注人员的类别方差的平均值
#     avg_annotator_variance = np.mean(annotator_variances)

#     # 计算整体标注结果的类别方差
#     overall_counts = np.bincount(annotations.flatten(), minlength=num_categories)  # 计算整体类别分布
#     overall_variance = np.var(overall_counts)

#     # 输出结果
#     print(f"文件: {os.path.basename(file)}")
#     print(f"  每个标注人员的类别方差的平均值: {avg_annotator_variance:.4f}")
#     print(f"  整体标注结果的类别方差: {overall_variance:.4f}")
#     print("-" * 50)


# import numpy as np
# import pandas as pd
# import glob
# import os
# from scipy.stats import mode  # 用于计算众数

# # 读取指定文件夹下的2-7分类的Excel文件
# folder_path = "/home/haoge/ymq1/round-grade/round-grade"  # 替换为实际路径
# file_paths = glob.glob(os.path.join(folder_path, "精彩程度*分类_标注汇总.xlsx"))

# # 遍历 2-7 分类的文件
# for file in sorted(file_paths, key=lambda x: int(x.split("精彩程度")[1][0])):  # 按分类数字排序
#     df = pd.read_excel(file)  # 读取 Excel 文件
#     annotations = df.iloc[:, 1:].values  # 取所有标注数据（去掉视频名称列）

#     # 计算每一行的众数，作为该行的最终标注结果
#     row_modes = mode(annotations, axis=1, keepdims=True)[0].flatten()  # 计算行众数
#     print(row_modes)
    
#     # 获取唯一类别
#     categories = np.unique(row_modes)
#     num_categories = len(categories)  # 统计类别数量

#     # 计算最终标注结果的类别分布
#     final_counts = np.bincount(row_modes, minlength=num_categories)  # 统计类别出现次数
#     final_variance = np.var(final_counts)  # 计算类别方差

#     # 输出结果
#     print(f"文件: {os.path.basename(file)}")
#     print(f"  最终标注结果的类别方差: {final_variance:.4f}")
#     print("-" * 50)


# import numpy as np
# import pandas as pd
# import glob
# import os

# import matplotlib.pyplot as plt
# from scipy.stats import mode
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import zhplot
# from statsmodels.stats.inter_rater import fleiss_kappa
# from itertools import combinations



# # 读取文件夹路径
# folder_path = "/home/haoge/ymq1/round-grade/round-grade"  # 替换为实际路径
# file_paths = glob.glob(os.path.join(folder_path, "精彩程度*分类_标注汇总.xlsx"))

# # 存储各分类的方差数据
# categories_list = []
# annotator_variances_list = []
# final_variances_list = []



# # 遍历 2-7 分类的文件
# for file in sorted(file_paths, key=lambda x: int(x.split("精彩程度")[1][0])):  # 按分类数字排序
#     df = pd.read_excel(file)  # 读取 Excel 文件
#     annotations = df.iloc[:, 2:].values  # 取所有标注数据（去掉视频名称列）

#     # 计算每个标注人员的类别方差
#     annotator_variances = []
#     for i in range(annotations.shape[1]):  # 遍历每个标注员
#         counts = np.bincount(annotations[:, i])  # 统计该标注员的类别分布
#         variance = np.var(counts)  # 计算方差
#         annotator_variances.append(variance)

#     avg_annotator_variance = np.mean(annotator_variances)  # 计算标注人员类别方差的平均值

#     # 计算每一行的众数作为最终标注结果
#     row_modes = mode(annotations, axis=1, keepdims=True)[0].flatten()

#     # 计算最终标注结果的类别方差
#     final_counts = np.bincount(row_modes)  # 计算众数的类别分布
#     final_variance = np.var(final_counts)  # 计算类别方差

#     # 记录数据
#     category_num = int(file.split("精彩程度")[1][0])  # 获取分类数
#     categories_list.append(category_num)
#     annotator_variances_list.append(avg_annotator_variance)
#     final_variances_list.append(final_variance)

#     # 输出结果
#     print(f"文件: {os.path.basename(file)}")
#     print(f"  人工标注类别方差的平均值: {avg_annotator_variance:.4f}")
#     print(f"  本文标注方法的类别方差: {final_variance:.4f}")
#     print("-" * 50)

# # 自定义横坐标标签
# category_labels = [f"精彩等级{cat}分类" for cat in categories_list]

# # 绘制折线图
# plt.figure(figsize=(8, 5))
# plt.plot(categories_list, annotator_variances_list, marker='o', linestyle='-', label="人工标注类别方差")
# plt.plot(categories_list, final_variances_list, marker='s', linestyle='--', label="混合标注类别方差")

# plt.ylabel("方差")
# plt.xticks(categories_list, category_labels)  # 使用自定义标签，旋转角度防止重叠
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.savefig('cat-var.png')







# import os
# import json
# import numpy as np
# import pandas as pd
# from shutil import copyfile
# from tqdm import tqdm

# def extract_frame_range_from_filename(filename):
#     """
#     从视频文件名中提取帧范围
#     示例：segment_7983_8035.mp4 -> (7983, 8035)
#     """
#     parts = filename.replace('segment_', '').replace('.mp4', '').split('_')
#     return int(parts[0]), int(parts[1])

# def extract_skeleton_data_from_json(json_data, start_frame, end_frame):
#     """
#     从JSON数据中提取top和bottom的骨骼数据
#     """
#     extracted_data = []
#     for frame_idx in range(start_frame, end_frame + 1):
#         if str(frame_idx) in json_data:
#             frame_data = json_data[str(frame_idx)]
#             extracted_data.append({
#                 'frame_index': frame_idx - start_frame,
#                 'bottom': frame_data.get('bottom', []),
#                 'top': frame_data.get('top', [])
#             })
#     return extracted_data

# def convert_to_kinetics_format(extracted_data):
#     """
#     将top和bottom数据合并并转换为Kinetics格式
#     """
#     kinetics_data = []
    
#     for frame_data in extracted_data:
#         frame_index = frame_data['frame_index']
#         top_points = frame_data['top']
#         bottom_points = frame_data['bottom']
        
#         skeleton = []

#         # 合并 top 和 bottom 的 pose 和 score
#         if bottom_points:
#             pose = np.array(bottom_points).flatten().tolist()
#             score = [1.0] * len(pose)
#             skeleton.append({
#                 'pose': pose,
#                 'score': score
#             })

#         if top_points:
#             pose = np.array(top_points).flatten().tolist()
#             score = [1.0] * len(pose)
#             skeleton.append({
#                 'pose': pose,
#                 'score': score
#             })

#         # 组装单帧数据
#         frame_kinetics_data = {
#             'frame_index': frame_index,
#             'skeleton': skeleton
#         }
#         kinetics_data.append(frame_kinetics_data)

#     return kinetics_data

# def save_data_to_json(output_path, data):
#     """
#     保存数据到JSON文件
#     """
#     with open(output_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4)

# def split_videos_by_ratio(video_list, ratios):
#     """
#     按照给定比例拆分视频列表
#     """
#     total_videos = len(video_list)
#     indices = np.arange(total_videos)
#     # np.random.shuffle(indices)  # 打乱顺序
    
#     split_indices = []
#     start_idx = 0
    
#     for ratio in ratios:
#         end_idx = start_idx + int(ratio * total_videos)
#         split_indices.append(indices[start_idx:end_idx])
#         start_idx = end_idx
    
#     return split_indices

# def process_videos_by_ratio(xlsx_path, json_path, video_dir, output_dir, json_output_dir):
#     """
#     处理视频数据，按照固定比例划分
#     """
#     # 读取Excel文件获取视频列表
#     df = pd.read_excel(xlsx_path)
#     video_list = df['视频名称'].tolist()
    
#     # 加载JSON数据
#     with open(json_path, 'r') as f:
#         json_data = json.load(f)
    
#     # 定义划分比例（对应类别 4, 3, 2, 1, 0）
#     ratios = [0.1, 0.2, 0.2, 0.2, 0.3]
#     split_indices = split_videos_by_ratio(video_list, ratios)
    
#     # 创建类别文件夹
#     output_dirs = [os.path.join(output_dir, str(i)) for i in range(5)]
#     json_output_dirs = [os.path.join(json_output_dir, str(i)) for i in range(5)]
    
#     for dir in output_dirs + json_output_dirs:
#         os.makedirs(dir, exist_ok=True)
    
#     # 处理视频
#     for label, indices in enumerate(split_indices[::-1]):  # 反转索引，确保4->0匹配
#         for idx in indices:
#             video_filename = video_list[idx]
#             frame_range = extract_frame_range_from_filename(video_filename)
#             video_path = os.path.join(video_dir, video_filename)
            
#             if os.path.exists(video_path):
#                 # 复制视频
#                 video_output_path = os.path.join(output_dirs[label], video_filename)
#                 copyfile(video_path, video_output_path)
                
#                 # 处理JSON骨骼数据
#                 extracted_data = extract_skeleton_data_from_json(json_data, *frame_range)
#                 kinetics_data = convert_to_kinetics_format(extracted_data)
                
#                 # 保存JSON数据
#                 json_filename = video_filename.replace('.mp4', '_kinetics.json')
#                 json_output_path = os.path.join(json_output_dirs[label], json_filename)
#                 save_data_to_json(json_output_path, kinetics_data)

# # 调用新方法
# xlsx_path = '/home/haoge/ymq1/2016-3-xlsx/2016-3-round-score_D40.xlsx'
# json_path = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'
# video_dir = '/home/haoge/ymq1/2016-3-lround_30_D40'
# output_dir = 'test_myvideosdata'
# json_output_dir = 'test_myjsondata'

# process_videos_by_ratio(xlsx_path, json_path, video_dir, output_dir, json_output_dir)


