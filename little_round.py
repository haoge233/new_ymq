import json
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from moviepy.editor import VideoFileClip

from matplotlib.pyplot import MultipleLocator

# 读取JSON文件的函数
def load_d_values(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

# 查找回合的起始帧和结束帧
def get_rounds_from_json(d_values):
    rounds = []
    current_round_start = None
    previous_frame = None

    for frame, value in d_values.items():
        if value == 0:  # 识别回合的开始（值为0的帧）
            if current_round_start is not None:
                rounds.append((current_round_start, previous_frame))  # 上一回合结束
            current_round_start = frame  # 当前回合的起始帧
        previous_frame = frame  # 更新上一帧为当前帧

    # 添加最后一个回合
    if current_round_start is not None and previous_frame is not None:
        rounds.append((current_round_start, previous_frame))

    return rounds

# # 查找极小值点，并返回其索引和对应的值
# def find_local_minima(y_values, min_distance=30):
#     minima_idx, _ = find_peaks(-np.array(y_values))  # 找到极小值点的索引
#     minima_values = [y_values[i] for i in minima_idx]  # 获取对应的极小值

#     # 过滤相距小于min_distance的极小值
#     filtered_minima_idx = []
#     filtered_minima_values = []
#     for i in range(len(minima_idx)):
#         if minima_values[i]<40:
#             if not filtered_minima_idx:  # 如果是第一个点，直接添加
#                 filtered_minima_idx.append(minima_idx[i])
#                 filtered_minima_values.append(minima_values[i])
#             else:
#                 if minima_idx[i] - filtered_minima_idx[-1] < min_distance:
#                     # 保留更小的极小值
#                     if minima_values[i] < filtered_minima_values[-1]:
#                         filtered_minima_idx[-1] = minima_idx[i]
#                         filtered_minima_values[-1] = minima_values[i]
#                 else:
#                     filtered_minima_idx.append(minima_idx[i])
#                     filtered_minima_values.append(minima_values[i])

#     return filtered_minima_idx, filtered_minima_values

def find_local_minima(y_values, frames, min_distance=30):
    # 使用find_peaks找到局部极大值，`-y_values`是为了找极小值
    minima_idx, _ = find_peaks(-np.array(y_values))  # 找到极小值点的索引
    minima_values = [y_values[i] for i in minima_idx]  # 获取对应的极小值

    # 获取对应的帧号
    minima_frames = [frames[i][0] for i in minima_idx]  # 获取对应的帧号

    # 过滤相距小于min_distance的极小值
    filtered_minima_frames = []
    filtered_minima_values = []
    for i in range(len(minima_idx)):
        if minima_values[i] < 40:  # 可以根据需要设置阈值
            if not filtered_minima_frames:  # 如果是第一个点，直接添加
                filtered_minima_frames.append(minima_frames[i])
                filtered_minima_values.append(minima_values[i])
            else:
                # 如果当前极小值与前一个极小值之间的间距小于min_distance
                if int(minima_frames[i]) - int(filtered_minima_frames[-1]) < min_distance:
                    # 保留更小的极小值
                    if minima_values[i] < filtered_minima_values[-1]:
                        filtered_minima_frames[-1] = minima_frames[i]
                        filtered_minima_values[-1] = minima_values[i]
                else:
                    # 如果间距足够大，直接添加当前极小值
                    filtered_minima_frames.append(minima_frames[i])
                    filtered_minima_values.append(minima_values[i])

    return filtered_minima_frames, filtered_minima_values  # 返回帧号和极小值


# 计算每个片段的D值最大最小差
def calculate_d_range(d_values, start_frame, end_frame):
    frames = list(d_values.items())[list(d_values.keys()).index(start_frame):list(d_values.keys()).index(end_frame)+1]
    print(frames)
    min_d = float('inf')
    max_d = -float('inf')
    for _, d_vals in frames:
        d_vals=d_vals[0]
        print(d_vals)
        min_d = min(min_d, d_vals)
        max_d = max(max_d, d_vals)
    return max_d - min_d

# 根据极值点分割视频
def split_video_using_extrema(video_path, extrema_points, d_values,last_index, sframes,min_d_diff=30, min_frames_last_segment=30, save_dir="split_videos/"):
    # 加载视频
    video = VideoFileClip(video_path)

    # 确保保存文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    segments = []
    # extrema_points = [str(int(i) + int(sframes)) for i in extrema_points]
    print("last")
    print(last_index)
    extrema_points = [sframes] + extrema_points + [last_index]  # 添加视频的起始帧和结束帧
    print(extrema_points)

    for i in range(1, len(extrema_points)):
        start_frame = extrema_points[i - 1]
        end_frame = extrema_points[i]
        print(start_frame)
        print(end_frame)
        # 计算D值的最大最小差
        d_diff = calculate_d_range(d_values, start_frame, end_frame)
        start_frame=int(start_frame)
        end_frame=int(end_frame)
        # 对于最后一个片段，确保其帧数大于30
        if i == len(extrema_points) - 1:
            last_segment_frames = end_frame - start_frame
            # print("last")
            # print(last_segment_frames)
            if last_segment_frames < min_frames_last_segment:
                print(f"最后一个片段的帧数小于30 ({last_segment_frames}帧)，不保存该片段")
                continue
        # 如果差值大于指定的最小差值，则保存该片段
        if d_diff > min_d_diff:
            segment = video.subclip(start_frame / 25, end_frame / 25)  # 假设视频的帧率是30fps
            segment_filename = os.path.join(save_dir, f"segment_{start_frame}_{end_frame}.mp4")
            segment.write_videofile(segment_filename, codec="libx264")
            segments.append(segment_filename)



    print(f"总共分割了 {len(segments)} 个片段")
    return segments

# 主函数
def main():
    # JSON文件路径
    json_file_path = '/home/haoge/ymq1/20160430-json/frame_similarity_results.json'  # 替换为你的JSON文件路径
    json_file_path1 = '/home/haoge/ymq1/20160430-json/frame_distances.json'  # 替换为你的JSON文件路径
    video_path = '/home/haoge/solopose/SoloShuttlePose/videos 20160430/20160430/output.mp4'  # 替换为你的视频文件路径
    save_dir = "20160430-lround_30_D40"  # 保存图像的目录
    kmeans_idx=1 

    # 读取D值数据
    d_values = load_d_values(json_file_path)
    d_values1 = load_d_values(json_file_path1)
    # 获取回合的起始和结束帧
    rounds = get_rounds_from_json(d_values1)
    print(rounds)


    # 计算每个回合的D值，查找极小值
    for round_idx, (start_frame, end_frame) in enumerate(rounds):
        # 获取start_frame和end_frame之间的帧以及最后一个元素的索引
        start_index = list(d_values.keys()).index(start_frame)
        end_index = list(d_values.keys()).index(end_frame)
        print(start_frame)
        print(end_frame)
        # 获取切片范围内的帧
        frames = list(d_values.items())[start_index:end_index + 1]

        for keypoint_idx in range(5):  # 假设每一帧有5个D值
            if keypoint_idx==kmeans_idx:
                d_values_per_keypoint = [d_vals[keypoint_idx] for _, d_vals in frames]
                # print(d_values_per_keypoint)
                minima_idx, minima_values = find_local_minima(d_values_per_keypoint,frames,min_distance=30)
                print(f"回合 {round_idx + 1}, Keypoint {keypoint_idx + 1}, 极小值点: {minima_idx}")

                # 分割视频并保存片段
                print("end")
                print(end_index)
                split_video_using_extrema(video_path, minima_idx, d_values,end_frame,start_frame, min_d_diff=20, min_frames_last_segment=40, save_dir=save_dir)
                break

if __name__ == '__main__':
    main()