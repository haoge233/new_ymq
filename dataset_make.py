import os
import random
import shutil
import pandas as pd

# 定义原视频文件夹路径和新视频文件夹路径
original_folder_path = '/home/haoge/ymq1/2016-3-lround_30_D40'
new_folder_path = '/home/haoge/ymq1/dataset'

# 创建新文件夹（如果不存在的话）
os.makedirs(new_folder_path, exist_ok=True)

# 获取文件夹下所有的视频文件
video_files = [f for f in os.listdir(original_folder_path) if f.endswith('.mp4')]

# 打乱文件顺序
random.shuffle(video_files)

# 创建记录文件名映射关系的列表
name_mapping = []

# 进行重命名并移动到新文件夹
for index, original_name in enumerate(video_files):
    # 生成新的文件名
    new_name = f"video_{index+1}.mp4"
    
    # 构造原文件和新文件的路径
    original_path = os.path.join(original_folder_path, original_name)
    new_path = os.path.join(new_folder_path, new_name)
    
    # 移动并重命名文件到新文件夹
    shutil.copy(original_path, new_path)  # 使用 copy 保留原文件
    
    # 记录原始文件名和新文件名的映射关系
    name_mapping.append({'Original Name': original_name, 'New Name': new_name})

# 创建一个 DataFrame 并保存到 Excel 文件
df = pd.DataFrame(name_mapping)
df.to_excel('file_renaming_mapping.xlsx', index=False)

print("Files have been renamed and moved to the new folder. The mapping has been saved to 'file_renaming_mapping.xlsx'.")
