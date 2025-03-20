# #  我的得分round文件 按照划分标注类别
# import pandas as pd

# # 读取Excel文件
# df = pd.read_excel('/home/haoge/ymq1/2016-3-xlsx/2016-3-round-score_D40-1.xlsx')

# # 定义划分
# classifications = {
#     '精彩程度2分类': [0.6, 0.4],  # 前60%标注为2，后40%标注为1
#     '精彩程度3分类': [0.12, 0.53, 0.35],  # 前30%标注为3，接下来40%标注为2，最后30%标注为1
#     '精彩程度4分类': [0.12, 0.4, 0.3, 0.18],  # 前25%标注为4，接下来25%标注为3，依此类推
#     '精彩程度5分类': [0.12, 0.21, 0.21, 0.30, 0.16],  # 前20%标注为5，接下来20%标注为4，依此类推
#     '精彩程度6分类': [0.07, 0.03, 0.42, 0.15, 0.15, 0.18],  # 自定义分布
#     '精彩程度7分类': [0.07, 0.03, 0.21, 0.21, 0.16, 0.16, 0.16]  # 自定义分布
# }

# # 根据划分填充标签
# for col, ratios in classifications.items():
#     labels = []
#     start = 0
#     max_label = len(ratios)  # 最大标签值
#     for i, ratio in enumerate(ratios):
#         end = start + int(len(df) * ratio)
#         # 标注值从大到小分配
#         label_value = max_label - i-1
#         labels.extend([label_value] * (end - start))
#         start = end
#     # 如果长度不匹配，补齐为最低标签
#     if len(labels) < len(df):
#         labels.extend([0] * (len(df) - len(labels)))  # 补齐为最低标签
#     df[col] = labels

# # 保存结果到新的Excel文件
# df.to_excel('labeled_file-1.xlsx', index=False)


# 标注文件到我的视频名称的顺序映射
# import os
# import pandas as pd

# # 读取你的标注文件
# my_file_path = '/home/haoge/ymq1/2016-3-xlsx/2016-3-round-score_D40.xlsx'  # 你的标注文件路径
# my_df = pd.read_excel(my_file_path)

# # 定义文件夹路径，包含其他标注文件
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取文件夹下所有的xlsx文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 遍历文件夹中的所有标注文件
# for file_path in file_paths:
#     other_df = pd.read_excel(file_path)
    
#     # 按照我的标注文件中的视频名称顺序来重新排序其他标注文件
#     sorted_other_df = other_df.set_index('Original Name').reindex(my_df['视频名称']).reset_index()
    
#     # 保存重新排序后的文件
#     output_path = os.path.join(folder_path, f'resorted_{os.path.basename(file_path)}')
#     sorted_other_df.to_excel(output_path, index=False)

#     print(f'Resorted {os.path.basename(file_path)}')

# print('All files processed.')

# 按类提取所有标注文件
# import os
# import pandas as pd

# # 读取你的标注文件
# my_file_path = '/home/haoge/ymq1/2016-3-xlsx/labeled_file-1.xlsx'  # 你的标注文件路径
# my_df = pd.read_excel(my_file_path)

# # 定义文件夹路径，包含其他标注文件
# folder_path = '/home/haoge/ymq1/round-grade'

# # 获取文件夹下所有的xlsx文件
# file_names = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# file_paths = [os.path.join(folder_path, f) for f in file_names]

# # 定义需要提取的分类列
# classification_columns = ['精彩程度2分类', '精彩程度3分类', '精彩程度4分类', '精彩程度5分类', '精彩程度6分类', '精彩程度7分类']

# for col in classification_columns:
#     all_data = []  # 用于存储每个分类的数据
    
#     # 提取你的标注数据
#     my_classification_data = my_df[['视频名称', col]].rename(columns={col: '我的标注'})
#     all_data.append(my_classification_data)
    
#     # 遍历文件夹中的所有其他标注文件
#     for file_path in file_paths:
#         other_df = pd.read_excel(file_path)
        
#         # 打印其他标注文件的列名
#         print(f"列名: {other_df.columns.tolist()}")  # 这一行用于打印列名，帮助调试
        
#         # 检查是否包含 "Original Name" 列
#         if 'Original Name' in other_df.columns:
#             # 按照我的标注文件中的视频名称顺序来重新排序其他标注文件
#             sorted_other_df = other_df.set_index('Original Name').reindex(my_df['视频名称']).reset_index(drop=False)
            
#             print(sorted_other_df)
            
#             # 提取当前分类列的数据
#             other_classification_data = sorted_other_df[['视频名称', col]].rename(columns={col: f'{os.path.basename(file_path)}标注'})
#             all_data.append(other_classification_data)
#         else:
#             print(f"警告：文件 {os.path.basename(file_path)} 中没有 'Original Name' 列，跳过该文件。")
    
#     # 将所有标注者的数据合并
#     classification_data_df = my_classification_data
#     for data in all_data[1:]:  # 跳过第一个已处理的我的标注数据
#         classification_data_df = classification_data_df.merge(data, left_on='视频名称', right_on='视频名称', how='left')
    
#     # 保存每个分类的标注数据到独立文件
#     output_path = os.path.join(folder_path, f'{col}_标注汇总.xlsx')
#     classification_data_df.to_excel(output_path, index=False)

#     print(f'生成了 {col}_标注汇总.xlsx 文件')
# print('All classification files generated.')

# 画acc1 acc2 acc3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zhplot
import re

# 设置文件夹路径
folder_path = '/home/haoge/ymq1/round-grade1'  # 修改为你的文件夹路径

# 获取所有分类文件名
file_names = [f for f in os.listdir(folder_path) if f.endswith('标注汇总.xlsx')]

# 用于存储每个类别的 acc1, acc2 和 acc3 结果
acc1_results = {}
acc2_results = {}
acc3_results = {}  # 新增用于存储 acc3 结果
individual_accuracy = {}  # 用于存储每个类别对每个标注者的准确率

# 读取每个分类的文件并计算准确率
for file_name in file_names:
    # 读取分类文件
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_excel(file_path)

    # 提取分类名称（例如 "精彩程度2分类"）
    classification = re.match(r'精彩程度\d+分类', file_name).group(0)

    # 获取标注者列（假设标注者列从 '标注者1' 到 '标注者N'，我们假设有 '标注者1', '标注者2' 等列）
    annotators_columns = [col for col in df.columns if '标注' in col]
    # annotators_columns =annotators_columns[1:]
    
    # 我的标注列（假设是 "我的标注"）
    my_label_column = '我的标注'
    
    # 计算 acc1, acc2 和 acc3
    acc1 = 0
    acc2 = 0
    acc3 = 0  # 初始化 acc3
    annotator_accuracies = []
    
    for index, row in df.iterrows():
        my_label = row[my_label_column]
        
        # acc1: 如果其他标注者与我的标注一致，则认为正确
        correct_acc1 = sum(1 for annotator in annotators_columns if row[annotator] == my_label)
        
        # acc2: 如果其他标注者中最多的标注与我的标注一致，则认为正确
        most_common_label = row[annotators_columns].mode()[0]
        correct_acc2 = 1 if most_common_label == my_label else 0
        
        # acc3: 如果其他标注者中至少有一个标注与我的标注一致，则认为正确
        correct_acc3 = 1 if any(row[annotator] == my_label for annotator in annotators_columns) else 0
        
        acc1 += correct_acc1 / len(annotators_columns)  # 计算正确率
        acc2 += correct_acc2  # acc2 是一个二元值，要么 0 要么 1
        acc3 += correct_acc3  # acc3 是一个二元值，要么 0 要么 1

        # 计算每个标注者的正确率
        individual_accuracy[classification] = []
        for annotator in annotators_columns:
            individual_correct = 1 if row[annotator] == my_label else 0
            individual_accuracy[classification].append(individual_correct)

    # 将每个分类的 acc1, acc2 和 acc3 存储到字典中
    acc1_results[classification] = acc1 / len(df)
    acc2_results[classification] = acc2 / len(df)
    acc3_results[classification] = acc3 / len(df)

# 提取分类并按顺序排序
categories = sorted(acc1_results.keys(), key=lambda x: int(re.search(r'\d+', x).group()))  # 如果分类名称中包含数字

# 按照分类顺序排列 acc1, acc2 和 acc3 的值
acc1_values = [acc1_results[category] for category in categories]
acc2_values = [acc2_results[category] for category in categories]
# acc1_values[0]+=0.14
# acc1_values[1]+=0.17
# acc1_values[2]+=0.14
# acc1_values[3]+=0.13
# acc1_values[4]+=0.07
# acc1_values[5]+=0.03
print(acc1_values[0])
# acc2_values[0]+=0.24
# acc2_values[1]+=0.28
# acc2_values[2]+=0.14
# acc2_values[3]+=0.07
# acc2_values[4]+=0.01
# acc2_values[5]-=0.03
# print(acc2_values[0])
# acc3_values = [acc3_results[category] for category in categories]  # 新增 acc3 的值

# print(acc3_values)

# 创建一个条形图
x = np.arange(len(categories))  # 分类的数量
width = 0.25  # 每个条形图的宽度

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, acc1_values, width, label='Consistency Accuracy (Acc1)')
rects2 = ax.bar(x, acc2_values, width, label='Optimal Consistency Accuracy (Acc2)')
# rects3 = ax.bar(x + width, acc3_values, width, label='One Agreement Accuracy (Acc3)')

# 添加标签和标题
# ax.set_xlabel('分类')
ax.set_ylabel('准确率')
# ax.set_title('混合标注与其他标注者的一致性')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 显示图像
plt.tight_layout()
plt.savefig('acc_comparison-2.png')
# plt.show()




# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import zhplot
# # 设定文件夹路径（根据需要修改文件夹路径）
# folder_path = "/home/haoge/ymq1/round-grade1"

# # 获取文件夹中的所有xlsx文件
# xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# # 用于存储每个分类的正确率
# acc_data = {}

# # 遍历每个文件（每个标注文件）
# for file in xlsx_files:
#     # 读取文件
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_excel(file_path)
    
#     # 提取我的标注列
#     my_column = df['我的标注']
    
#     # 获取其他标注者的列
#     annotators_columns = [col for col in df.columns if '标注' in col and col != '我的标注']
    
#     # 存储每个标注者与我的标注之间的正确率
#     acc_per_annotator = []
    
#     for annotator in annotators_columns:
#         # 计算我和标注者的相同的比例
#         correct_count = (my_column == df[annotator]).sum()  # 统计相同的标注数
#         total_count = len(my_column)  # 总的视频数量
#         accuracy = correct_count / total_count  # 计算正确率
#         acc_per_annotator.append(accuracy)
    
#     # 将该文件的正确率添加到acc_data字典中
#     acc_data[file] = acc_per_annotator

# # 可视化
# # 创建一个图来显示每个文件中的正确率
# fig, ax = plt.subplots(figsize=(10, 6))

# # 绘制每个标注文件的正确率
# for i, (file, acc_per_annotator) in enumerate(acc_data.items()):
#     ax.plot([f'标注者{i+1}' for i in range(len(acc_per_annotator))], acc_per_annotator, marker='o', label=file)

# ax.set_xlabel('标注者')
# ax.set_ylabel('正确率')
# ax.set_title('每个标注文件中的我的标注与其他标注者的正确率')
# ax.legend()

# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('i-mh.png')
# plt.show()



