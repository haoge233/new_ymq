import os
import shutil

# 原始图片所在的文件夹路径
source_folder = '/home/haoge/ymq1/20160501_3_frames_output'  # 替换为你的图片文件夹路径

# 目标文件夹路径
destination_folder = '/home/haoge/ymq1/tiqupho_2016_3'  # 替换为你的目标文件夹路径

# 文件列表，包含要提取的帧的编号
# frame_list = ['15800', '45456', '9743', '28995', '37913']
# frame_list =['16468', '10379', '29784', '12192', '25277'] #2016-1
frame_list =['5663', '12161', '33796', '41027', '37386'] #2016-3


# 如果目标文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历文件夹中的所有文件
for frame_name in frame_list:
    # 构建源文件的完整路径
    source_file = os.path.join(source_folder, f"frame_{frame_name}.jpg")

    # 检查文件是否存在
    if os.path.exists(source_file):
        # 构建目标文件的完整路径
        destination_file = os.path.join(destination_folder, f"keyframe_{frame_name}.jpg")
        
        # 拷贝文件到目标文件夹
        shutil.copy(source_file, destination_file)
        print(f"Copied: {source_file} to {destination_file}")
    else:
        print(f"File not found: {source_file}")

print("Frame extraction completed.")
