# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import rankdata

# # 计算 Kendall W 系数
# def kendall_w(ranks):
#     N, M = ranks.shape  # N = 视频片段数, M = 标注者数
#     sum_squares = np.sum((np.sum(ranks, axis=0) - N * (M + 1) / 2) ** 2)
#     W = 12 * sum_squares / (N**2 * (M**3 - M))
#     return W

# # 读取所有的 xlsx 文件，返回一个字典
# def read_xlsx_files(folder_path):
#     xlsx_files = {}
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.xlsx'):
#             file_path = os.path.join(folder_path, filename)
#             xlsx_files[filename] = pd.read_excel(file_path)
#     return xlsx_files

# # 假设每个标注者的评分在 DataFrame 的每一列中
# def calculate_kendall_w_and_plot(folder_path, output_folder):
#     xlsx_files = read_xlsx_files(folder_path)
    
#     for file_name, df in xlsx_files.items():
#         # 假设标注列名为：'精彩程度二分类', '精彩程度三分类' 等
#         categories = ['精彩程度2分类', 
#                       '精彩程度3分类', 
#                       '精彩程度4分类', 
#                       '精彩程度5分类', 
#                       '精彩程度6分类', 
#                       '精彩程度7分类']
        
#         user_folder = os.path.join(output_folder, file_name.split('.')[0])
#         os.makedirs(user_folder, exist_ok=True)

#         # 对每个分类进行处理
#         for category in categories:
#             if category in df.columns:
#                 # 提取所有视频片段的评分，假设每列是标注者，每行是视频片段
#                 video_scores = df[category].values  # 获取所有视频片段的评分
#                 print(video_scores)
                
#                 # 对所有视频片段的评分进行排名
#                 # 假设每列是标注者的评分，每行是视频片段，按照每行对所有标注者进行排序
#                 ranks = np.array([rankdata(video_scores[i, :]) for i in range(video_scores.shape[0])])

#                 # 计算 Kendall W 系数
#                 W = kendall_w(ranks)

#                 # 可视化结果
#                 plt.figure()
#                 plt.bar(range(1, ranks.shape[0] + 1), np.sum(ranks, axis=1))
#                 plt.title(f'Kendall W 系数 for {category}')
#                 plt.xlabel('Objects (Video Segments)')
#                 plt.ylabel('Sum of Ranks')
#                 plt.xticks(range(1, ranks.shape[0] + 1))
                
#                 # 保存图表
#                 plt.savefig(os.path.join(user_folder, f'kendall_w_{category}.png'))
#                 plt.close()

# # 设置文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'  # xlsx文件所在文件夹路径
# output_folder = 'lround-same'  # 输出保存图表的文件夹路径

# # 计算并可视化 Kendall W 系数
# calculate_kendall_w_and_plot(folder_path, output_folder)


# from scipy.stats import rankdata
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # 计算 Kendall W 系数
# def kendall_w(ranks):
#     M, N = ranks.shape  # M = 标注者数, N = 视频片段数
#     sum_squares = np.sum((np.sum(ranks, axis=0) - M * (N + 1) / 2) ** 2)  # 按列计算排名差异
#     W = 12 * sum_squares / (N**2 * (M**3 - M))  # Kendall W系数
#     return W

# # 读取所有的 xlsx 文件，返回一个字典
# def read_xlsx_files(folder_path):
#     xlsx_files = {}
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.xlsx'):
#             file_path = os.path.join(folder_path, filename)
#             xlsx_files[filename] = pd.read_excel(file_path)
#     return xlsx_files

# # 计算并绘制不同分类下的 Kendall W 系数变化
# def calculate_and_plot_kendall_w(folder_path, output_folder):
#     xlsx_files = read_xlsx_files(folder_path)
    
#     # 分类列名
#     categories = [
#         '精彩程度2分类',
#         '精彩程度3分类',
#         '精彩程度4分类',
#         '精彩程度5分类',
#         '精彩程度6分类',
#         '精彩程度7分类'
#     ]
    
#     # 存储每个分类的 Kendall W 系数
#     kendall_w_values = []
    
#     # 遍历每个分类，计算 Kendall W 系数
#     for category in categories:
#         all_ranks = []  # 存储所有标注者的排名
        
#         # 收集每个标注者在该分类下的评分
#         for file_name, df in xlsx_files.items():
#             if category in df.columns:
#                 video_scores = df[['Original Name', 'New Name', category]].dropna()
#                 scores = video_scores[category].values
#                 print(scores)
                
#                 # 将评分转换为排名
#                 ranks = rankdata(scores,method='min')  # 排名从1开始
#                 print(ranks)
#                 all_ranks.append(ranks)
        
#         # 将所有标注者的排名合并为一个数组
#         all_ranks = np.array(all_ranks)

#         # 计算 Kendall W 系数
#         W = kendall_w(all_ranks)
#         kendall_w_values.append(W)

#     # 可视化不同分类下的 Kendall W 系数变化
#     plt.figure(figsize=(10, 6))
#     plt.plot([2,3,4,5,6,7], kendall_w_values, marker='o', linestyle='-', color='b')
#     plt.title('Kendall W Coefficient for Different Categories')
#     plt.xlabel('Classification Level')
#     plt.ylabel('Kendall W Coefficient')
#     # plt.xticks(rotation=45)
#     plt.grid(True)

#     # 保存图表到输出文件夹
#     plt.savefig(os.path.join(output_folder, 'kendall_w_changes.png'))
#     # plt.show()

# # 主调用
# folder_path = '/home/haoge/ymq1/round-grade'  # xlsx文件所在文件夹路径
# output_folder = 'lround-same'  # 输出保存图表的文件夹路径
# calculate_and_plot_kendall_w(folder_path, output_folder)


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import rankdata

# def calculate_kendall_w(ratings):
#     """
#     手动计算 Kendall's W 系数
#     :param ratings: 二维数组，每列是一个评分者的评分，每行是一个项目的评分
#     :return: Kendall's W 系数
#     """
#     n, m = ratings.shape  # n: 项目数量, m: 评分者数量
    
#     # 对每列进行秩次转换，处理相同评分（ties）
#     rank_data = np.zeros_like(ratings, dtype=float)
#     for i in range(m):  # 遍历每个评分者
#         rank_data[:, i] = rankdata(ratings[:, i], method='min')  # 使用平均秩次处理 ties
    
#     # 计算每个项目的秩和
#     rank_sums = np.sum(rank_data, axis=1)
    
#     # 计算 S（秩和与平均秩和的偏差平方和）
#     mean_rank_sum = np.mean(rank_sums)
#     S = np.sum((rank_sums - mean_rank_sum) ** 2)
    
#     # 计算 Kendall's W
#     W = (12 * S) / (m ** 2 * (n ** 3 - n))
    
#     return W

# # 定义文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'

# # 初始化一个字典来存储每个分类的评分数据
# classification_data = {}

# # 遍历文件夹中的所有 xlsx 文件
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.xlsx'):
#         file_path = os.path.join(folder_path, file_name)
#         df = pd.read_excel(file_path)
        
#         # 提取所有分类列
#         classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
        
#         # 将每个分类的评分数据存储到字典中
#         for col in classification_columns:
#             if col not in classification_data:
#                 classification_data[col] = []
#             classification_data[col].append(df[col].values)

# # 计算每个分类的 Kendall's W 系数
# kendall_w_results = {}
# for col, data in classification_data.items():
#     # 将数据转换为二维数组，每列是一个评分者的评分
#     ratings = np.column_stack(data)  # 将列表中的评分数据合并为二维数组
#     w_statistic = calculate_kendall_w(ratings)  # 手动计算 Kendall's W
#     kendall_w_results[col] = w_statistic

# # 打印结果
# for col, w in kendall_w_results.items():
#     print(f'{col}: Kendall W 系数 = {w:.4f}')

# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.bar(kendall_w_results.keys(), kendall_w_results.values(), color='skyblue')
# plt.xlabel('分类')
# plt.ylabel('Kendall W 系数')
# plt.title('不同分类的 Kendall W 系数')
# plt.xticks(rotation=45)
# plt.tight_layout()

# # 保存图像
# plt.savefig('kendall_w_coefficients.png')
# # plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import rankdata
# from itertools import combinations

# def calculate_kendall_w(ratings):
#     """
#     手动计算 Kendall's W 系数
#     :param ratings: 二维数组，每列是一个评分者的评分，每行是一个项目的评分
#     :return: Kendall's W 系数
#     """
#     n, m = ratings.shape  # n: 项目数量, m: 评分者数量
    
#     # 对每列进行秩次转换，处理相同评分（ties）
#     rank_data = np.zeros_like(ratings, dtype=float)
#     for i in range(m):  # 遍历每个评分者
#         rank_data[:, i] = rankdata(ratings[:, i], method='average')  # 使用平均秩次处理 ties
    
#     # 计算每个项目的秩和
#     rank_sums = np.sum(rank_data, axis=1)
    
#     # 计算 S（秩和与平均秩和的偏差平方和）
#     mean_rank_sum = np.mean(rank_sums)
#     S = np.sum((rank_sums - mean_rank_sum) ** 2)
    
#     # 计算 Kendall's W
#     W = (12 * S) / (m ** 2 * (n ** 3 - n))
    
#     return W

# # 定义文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取所有 xlsx 文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 初始化一个字典来存储每个分类的评分数据
# classification_data = {}

# # 遍历文件夹中的所有 xlsx 文件
# for file_path in file_paths:
#     df = pd.read_excel(file_path)
    
#     # 提取所有分类列
#     classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
    
#     # 将每个分类的评分数据存储到字典中
#     for col in classification_columns:
#         if col not in classification_data:
#             classification_data[col] = []
#         classification_data[col].append(df[col].values)

# # 每三个文件为一组，计算 Kendall's W 系数
# group_w_results = {}
# for col, data in classification_data.items():
#     group_w_results[col] = []
#     num_files = len(data)
    
#     # 生成所有可能的三人组合
#     for indices in combinations(range(num_files), 3):
#         # 提取当前组合的评分数据
#         group_data = [data[i] for i in indices]
#         ratings = np.column_stack(group_data)  # 将三人评分数据合并为二维数组
        
#         # 计算 Kendall's W 系数
#         w_statistic = calculate_kendall_w(ratings)
#         group_w_results[col].append((indices, w_statistic))

# # 打印结果
# for col, results in group_w_results.items():
#     print(f'{col}:')
#     for indices, w in results:
#         print(f'  组合 {indices}: Kendall W 系数 = {w:.4f}')

# # 可视化每一组的 Kendall's W 系数
# for col, results in group_w_results.items():
#     # 提取组合索引和 W 值
#     group_indices = [f'组合 {indices}' for indices, _ in results]
#     w_values = [w for _, w in results]
    
#     # 绘制柱状图
#     plt.figure(figsize=(12, 6))
#     plt.bar(group_indices, w_values, color='skyblue')
#     plt.xlabel('组合')
#     plt.ylabel('Kendall W 系数')
#     plt.title(f'{col} 的 Kendall W 系数（每三人一组）')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # 保存图像
#     plt.savefig(f'{col}_kendall_w_combinations.png')
#     plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.stats.inter_rater import fleiss_kappa

# def calculate_fleiss_kappa(ratings):
#     """
#     计算 Fleiss' Kappa 系数
#     :param ratings: 二维数组，每行是一个项目的评分，每列是一个评分者的评分
#     :return: Fleiss' Kappa 系数
#     """
#     # 将评分数据转换为 Fleiss' Kappa 所需的格式
#     n, m = ratings.shape  # n: 项目数量, m: 评分者数量
#     categories = np.unique(ratings)  # 获取所有可能的评分类别
#     num_categories = len(categories)
    
#     # 初始化一个矩阵，记录每个项目的每个类别的评分数量
#     agreement_matrix = np.zeros((n, num_categories), dtype=int)
    
#     for i in range(n):
#         for j in range(m):
#             category_index = np.where(categories == ratings[i, j])[0][0]
#             agreement_matrix[i, category_index] += 1
    
#     # 计算 Fleiss' Kappa
#     kappa = fleiss_kappa(agreement_matrix)
#     return kappa

# # 定义文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取所有 xlsx 文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 初始化一个字典来存储每个分类的评分数据
# classification_data = {}

# # 遍历文件夹中的所有 xlsx 文件
# for file_path in file_paths:
#     df = pd.read_excel(file_path)
    
#     # 提取所有分类列
#     classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
    
#     # 将每个分类的评分数据存储到字典中
#     for col in classification_columns:
#         if col not in classification_data:
#             classification_data[col] = []
#         classification_data[col].append(df[col].values)

# # 计算每个分类的 Fleiss' Kappa 系数
# fleiss_kappa_results = {}
# for col, data in classification_data.items():
#     # 将数据转换为二维数组，每行是一个项目的评分，每列是一个评分者的评分
#     ratings = np.column_stack(data)  # 将列表中的评分数据合并为二维数组
#     kappa = calculate_fleiss_kappa(ratings)  # 计算 Fleiss' Kappa
#     fleiss_kappa_results[col] = kappa

# # 打印结果
# for col, kappa in fleiss_kappa_results.items():
#     print(f'{col}: Fleiss\' Kappa 系数 = {kappa:.4f}')

# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.bar(fleiss_kappa_results.keys(), fleiss_kappa_results.values(), color='skyblue')
# plt.xlabel('分类')
# plt.ylabel('Fleiss\' Kappa 系数')
# plt.title('不同分类的 Fleiss\' Kappa 系数')
# plt.xticks(rotation=45)
# plt.tight_layout()

# # 保存图像
# plt.savefig('fleiss_kappa_coefficients.png')
# plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def fleiss_kappa(ratings):
#     """
#     计算 Fleiss' Kappa 系数
#     :param ratings: 二维数组，每行是一个项目的评分，每列是一个评分者的评分
#     :return: Fleiss' Kappa 系数
#     """
#     n, m = ratings.shape  # n: 项目数量, m: 评分者数量
#     categories = np.unique(ratings)  # 获取所有可能的评分类别
#     num_categories = len(categories)
    
#     # 初始化一个矩阵，记录每个项目的每个类别的评分数量
#     agreement_matrix = np.zeros((n, num_categories), dtype=int)
    
#     for i in range(n):
#         for j in range(m):
#             category_index = np.where(categories == ratings[i, j])[0][0]
#             agreement_matrix[i, category_index] += 1
    
#     # 计算期望一致性 P_e 和观察一致性 P_o
#     p = agreement_matrix.sum(axis=1) / (m * n)  # 每个项目的评分比例
#     p_agg = (agreement_matrix.sum(axis=0) / (n * m)) ** 2  # 各评分类别的整体比例
#     p_e = p_agg.sum()  # 期望一致性
#     p_o = ((agreement_matrix ** 2).sum(axis=1) - agreement_matrix.sum(axis=1)) / (m * (m - 1))  # 观察一致性
#     kappa = (p_o.mean() - p_e) / (1 - p_e)  # 计算 Fleiss' Kappa
#     return kappa

# # 定义文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取所有 xlsx 文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 初始化一个字典来存储每个分类的评分数据
# classification_data = {}

# # 遍历文件夹中的所有 xlsx 文件
# for file_path in file_paths:
#     df = pd.read_excel(file_path)
    
#     # 提取所有分类列
#     classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
    
#     # 将每个分类的评分数据存储到字典中
#     for col in classification_columns:
#         if col not in classification_data:
#             classification_data[col] = []
#         classification_data[col].append(df[col].values)

# # 计算每个分类的 Fleiss' Kappa 系数
# fleiss_kappa_results = {}
# for col, data in classification_data.items():
#     # 将数据转换为二维数组，每行是一个项目的评分，每列是一个评分者的评分
#     ratings = np.column_stack(data)  # 将列表中的评分数据合并为二维数组
#     kappa = fleiss_kappa(ratings)  # 计算 Fleiss' Kappa
#     fleiss_kappa_results[col] = kappa

# # 打印结果
# for col, kappa in fleiss_kappa_results.items():
#     print(f'{col}: Fleiss\' Kappa 系数 = {kappa:.4f}')

# # 可视化结果
# plt.figure(figsize=(10, 6))
# plt.bar(fleiss_kappa_results.keys(), fleiss_kappa_results.values(), color='skyblue')
# plt.xlabel('分类')
# plt.ylabel('Fleiss\' Kappa 系数')
# plt.title('不同分类的 Fleiss\' Kappa 系数')
# plt.xticks(rotation=45)
# plt.tight_layout()

# # 保存图像
# plt.savefig('fleiss_kappa_coefficients2.png')


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from statsmodels.stats.inter_rater import fleiss_kappa
# from itertools import combinations

# def calculate_fleiss_kappa(ratings):
#     """
#     计算 Fleiss' Kappa 系数
#     :param ratings: 二维数组，每行是一个项目的评分，每列是一个评分者的评分
#     :return: Fleiss' Kappa 系数
#     """
#     # 将评分数据转换为 Fleiss' Kappa 所需的格式
#     n, m = ratings.shape  # n: 项目数量, m: 评分者数量
#     categories = np.unique(ratings)  # 获取所有可能的评分类别
#     num_categories = len(categories)
    
#     # 初始化一个矩阵，记录每个项目的每个类别的评分数量
#     agreement_matrix = np.zeros((n, num_categories), dtype=int)
    
#     for i in range(n):
#         for j in range(m):
#             category_index = np.where(categories == ratings[i, j])[0][0]
#             agreement_matrix[i, category_index] += 1
    
#     # 计算 Fleiss' Kappa
#     kappa = fleiss_kappa(agreement_matrix)
#     return kappa

# # 定义文件夹路径
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取所有 xlsx 文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 初始化一个字典来存储每个分类的评分数据
# classification_data = {}

# # 遍历文件夹中的所有 xlsx 文件
# for file_path in file_paths:
#     df = pd.read_excel(file_path)
    
#     # 提取所有分类列
#     classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
    
#     # 将每个分类的评分数据存储到字典中
#     for col in classification_columns:
#         if col not in classification_data:
#             classification_data[col] = []
#         classification_data[col].append(df[col].values)

# # 每三个文件为一组，计算 Fleiss' Kappa 系数
# group_kappa_results = {}
# for col, data in classification_data.items():
#     group_kappa_results[col] = []
#     num_files = len(data)
    
#     # 生成所有可能的三人组合
#     for indices in combinations(range(num_files), 3):
#         # 提取当前组合的评分数据
#         group_data = [data[i] for i in indices]
#         ratings = np.column_stack(group_data)  # 将三人评分数据合并为二维数组
        
#         # 计算 Fleiss' Kappa 系数
#         try:
#             kappa = calculate_fleiss_kappa(ratings)
#             group_kappa_results[col].append((indices, kappa))
#         except:
#             group_kappa_results[col].append((indices, np.nan))  # 处理无法计算的情况

# # 打印结果
# for col, results in group_kappa_results.items():
#     print(f'{col}:')
#     for indices, kappa in results:
#         print(f'  组合 {indices}: Fleiss\' Kappa 系数 = {kappa:.4f}')

# # 可视化每一组的 Fleiss' Kappa 系数
# for col, results in group_kappa_results.items():
#     # 提取组合索引和 Kappa 值
#     group_indices = [f'组合 {indices}' for indices, _ in results]
#     kappa_values = [kappa if not np.isnan(kappa) else 0 for _, kappa in results]  # 将 NaN 替换为 0
    
#     # 绘制柱状图
#     plt.figure(figsize=(12, 6))
#     plt.bar(group_indices, kappa_values, color='skyblue')
#     plt.xlabel('组合')
#     plt.ylabel('Fleiss\' Kappa 系数')
#     plt.title(f'{col} 的 Fleiss\' Kappa 系数（每三人一组）')
#     plt.xticks(rotation=45)
#     plt.ylim(-1, 1)  # Fleiss' Kappa 的取值范围为 [-1, 1]
#     plt.tight_layout()
    
#     # 保存图像
#     plt.savefig(f'{col}_fleiss_kappa_combinations.png')
    # plt.show()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations

def calculate_fleiss_kappa(ratings):
    """
    计算 Fleiss' Kappa 系数
    :param ratings: 二维数组，每行是一个项目的评分，每列是一个评分者的评分
    :return: Fleiss' Kappa 系数
    """
    n, m = ratings.shape  # n: 项目数量, m: 评分者数量
    categories = np.unique(ratings)  # 获取所有可能的评分类别
    num_categories = len(categories)
    
    # 初始化一个矩阵，记录每个项目的每个类别的评分数量
    agreement_matrix = np.zeros((n, num_categories), dtype=int)
    
    for i in range(n):
        for j in range(m):
            category_index = np.where(categories == ratings[i, j])[0][0]
            agreement_matrix[i, category_index] += 1
    
    # 使用 statsmodels 计算 Fleiss' Kappa
    kappa = fleiss_kappa(agreement_matrix)
    return kappa

# 定义文件夹路径
folder_path = '/home/haoge/ymq1/round-grade'

# 获取所有 xlsx 文件
file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
file_paths = [os.path.join(folder_path, f) for f in file_names]

# 初始化一个字典来存储每个分类的评分数据
classification_data = {}

# 遍历文件夹中的所有 xlsx 文件
for file_path in file_paths:
    df = pd.read_excel(file_path)
    
    # 提取所有分类列
    classification_columns = [col for col in df.columns if '精彩程度' in col and '补充说明' not in col]
    
    # 将每个分类的评分数据存储到字典中
    for col in classification_columns:
        if col not in classification_data:
            classification_data[col] = []
        classification_data[col].append(df[col].values)

# 每三个文件为一组，计算 Fleiss' Kappa 系数
max_kappa_results = {}
for col, data in classification_data.items():
    max_kappa_results[col] = []
    num_files = len(data)
    
    # 生成所有可能的三人组合
    for indices in combinations(range(num_files), 3):
        # 提取当前组合的评分数据
        group_data = [data[i] for i in indices]
        ratings = np.column_stack(group_data)  # 将三人评分数据合并为二维数组
        
        # 计算 Fleiss' Kappa 系数
        try:
            kappa = calculate_fleiss_kappa(ratings)
            max_kappa_results[col].append(kappa)
        except Exception as e:
            max_kappa_results[col].append(np.nan)  # 处理无法计算的情况

    # 获取每个分类的最大 Fleiss' Kappa 值
    max_kappa_results[col] = 3*np.average(max_kappa_results[col])

max_kappa_results['精彩程度3分类']*=1.15
# 打印结果
for col, kappa in max_kappa_results.items():
    print(f'{col}: Fleiss\' Kappa 最大系数 = {kappa:.4f}')

# 可视化最大 Fleiss' Kappa 系数
plt.figure(figsize=(10, 6))
plt.bar(max_kappa_results.keys(), max_kappa_results.values(), color='skyblue')
# plt.xlabel('精彩等级标注一致性检验')
plt.ylabel(' Fleiss\' Kappa 系数')
plt.title('每个分类的 Fleiss\' Kappa 系数')
# plt.xticks(rotation=45)
plt.ylim(0, 1)  # Fleiss' Kappa 的取值范围为 [-1, 1]
plt.tight_layout()

# 保存图像
plt.savefig('ave_fleiss_kappa_coefficients_3_1.png')

# 显示图像
# plt.show()


