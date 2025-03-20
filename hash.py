import os
import cv2
import imagehash
from collections import Counter
from PIL import Image
import re

# 自定义排序函数，确保按数字顺序排序
def natural_sort_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]

# 计算图像的感知哈希值（dhash）
def dhash(image, hash_size=8):
    # 转为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]  # 计算相邻像素的差异
    return imagehash.hex_to_hash(''.join(str(int(b)) for b in diff.flatten()))

# 计算两个哈希值之间的汉明距离（即它们有多少位不同）
def hamming_distance(hash1, hash2):
    return bin(int(str(hash1), 16) ^ int(str(hash2), 16)).count('1')

def filter_frames_by_most_frequent_hash(keyframes_folder, output_folder, max_diff=5, initial_window_size=10, consecutive_threshold=10):
    # 获取文件夹中的所有关键帧图片
    keyframes = sorted([img for img in os.listdir(keyframes_folder) if img.endswith(('.png', '.jpg', '.jpeg'))], key=natural_sort_key)

    # 计算所有帧的哈希值
    hashes = []
    for frame_name in keyframes:
        frame_path = os.path.join(keyframes_folder, frame_name)
        frame = cv2.imread(frame_path)
        hash_value = dhash(frame)
        hashes.append(hash_value)
    
    # 统计每个哈希值出现的频率
    hash_counts = Counter(hashes)
    print("Most frequent hashes:", hash_counts.most_common(5))  # 打印出现频率最高的5个哈希值

    # 找到出现频率最高的哈希值作为参考哈希
    most_common_hash = hash_counts.most_common(1)[0][0]
    print(f"Using most frequent hash: {most_common_hash}")
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化回合计数
    round_count = 1
    current_round_folder = os.path.join(output_folder, f"round_{round_count}")
    if not os.path.exists(current_round_folder):
        os.makedirs(current_round_folder)

    # 用于记录当前回合的帧数
    frame_count = 0

    i = 0  # 当前帧的位置
    diff_greater_than_threshold = 0
    print(len(keyframes))
    while i < len(keyframes):
        # 定义当前窗口的右边界
        window_end = min(i + initial_window_size, len(keyframes))

        save_frames = []

        # 检查窗口内每张图像与参考图像的差异
        for j in range(i, window_end):
            current_frame_name = keyframes[j]
            frame_path = os.path.join(keyframes_folder, current_frame_name)
            frame = cv2.imread(frame_path)
            current_hash = hashes[j]
            diff = hamming_distance(current_hash, most_common_hash)

            # 如果差异小于阈值，保存当前帧
            if diff <= max_diff:
                # 保存从左边界到当前图像的所有图像
                save_frames = keyframes[i:j+1]  # 从窗口左边界到当前图像
                break  # 找到差异小于阈值的帧后，停止继续检查
            else:
                diff_greater_than_threshold += 1

        # 如果窗口内有超过10张差异大于阈值的图像，则跳过该窗口内的所有图像
        if save_frames and diff_greater_than_threshold > consecutive_threshold:
            print(f"More than {consecutive_threshold} frames with difference greater than threshold, skipping this window.")
            # 更新窗口左边界为当前图像
            i = keyframes.index(save_frames[-1]) 
            round_count += 1  # 更新回合计数
            diff_greater_than_threshold = 0
            current_round_folder = os.path.join(output_folder, f"round_{round_count}")
            if not os.path.exists(current_round_folder):
                os.makedirs(current_round_folder)
        elif save_frames:
            # 如果找到了差异小于阈值的帧，保存整个窗口从左边界到该图像的图像
            diff_greater_than_threshold = 0
            for frame_name in save_frames:
                frame_path = os.path.join(keyframes_folder, frame_name)
                frame = cv2.imread(frame_path)
                frame_count += 1
                save_path = os.path.join(current_round_folder, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Saved: {frame_name}")
            # 处理完本回合，滑动窗口左边界
            i = keyframes.index(save_frames[-1]) + 1  # 滑动窗口的左边界
        else:
            # 如果没有找到差异小于阈值的图像，直接滑动左边界，检查下一个窗口
            print("No frames saved, moving window forward.")
            i = 1 + window_end  # 直接将左边界滑动到下一个图像

    print(f"Total rounds: {round_count}")
    print(f"Total frames processed: {frame_count}")

# 生成不同参数的组合，并分别运行测试
def run_tests(keyframes_folder, base_output_folder, max_diff_values, window_sizes, consecutive_threshold=10):
    for max_diff in max_diff_values:
        for window_size in window_sizes:
            # 为每个组合创建一个新的文件夹
            output_folder = os.path.join(base_output_folder, f"max_diff_{max_diff}_window_size_{window_size}")
            print(f"Running test for max_diff={max_diff}, window_size={window_size}...")
            filter_frames_by_most_frequent_hash(keyframes_folder, output_folder, max_diff=max_diff, initial_window_size=window_size, consecutive_threshold=window_size)

# 示例使用
keyframes_folder = 'keyframes_output'  # 提取的关键帧图片文件夹路径
base_output_folder = 'test_results'  # 存放测试结果的基础文件夹

# 测试参数
max_diff_values = [7,8]  # max_diff的可能值
window_sizes = range(300, 301, 10)  # window_size从10到150，间隔10

# 运行测试
run_tests(keyframes_folder, base_output_folder, max_diff_values, window_sizes)
