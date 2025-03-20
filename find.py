# # ---------------------------视频关键帧提取--------
# import cv2
# import numpy as np
# import os

# # 提取关键帧的函数
# def extract_keyframes(video_path, output_dir, threshold=1000000):
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print("Error: Unable to open video.")
#         return
    
#     # 获取视频的帧率
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Video FPS: {fps}")
    
#     # 创建保存关键帧的文件夹
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     ret, prev_frame = cap.read()
#     if not ret:
#         print("Error: Unable to read the first frame.")
#         return

#     # 将第一帧转换为灰度图
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
#     frame_count = 1
#     keyframe_count = 0
#     keyframe_paths = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         # 将当前帧转换为灰度图
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # 计算当前帧和前一帧之间的差异
#         frame_diff = cv2.absdiff(prev_gray, gray_frame)
#         diff_score = np.sum(frame_diff)
        
#         # 如果差异超过阈值，认为是关键帧
#         if diff_score > threshold:
#             keyframe_count += 1
#             keyframe_filename = os.path.join(output_dir, f"keyframe_{keyframe_count:04d}.jpg")
#             cv2.imwrite(keyframe_filename, frame)
#             keyframe_paths.append(keyframe_filename)
#             print(f"Keyframe {keyframe_count} saved: {keyframe_filename}")
        
#         # 更新前一帧为当前帧
#         prev_gray = gray_frame
    
#     cap.release()
#     print(f"\nTotal keyframes extracted: {keyframe_count}")
#     return keyframe_paths

# # 使用示例
# video_path = '/home/haoge/solopose/SoloShuttlePose/videos/20160501-1.mp4'  # 输入视频文件路径
# output_dir = 'keyframes_output_2016-1'  # 输出目录

# keyframes = extract_keyframes(video_path, output_dir)



import cv2
import os

# 视频文件路径
video_path = '/home/haoge/solopose/SoloShuttlePose/videos/20160501-3.mp4'
# 设置保存每一帧的文件夹
output_folder = '20160501_3_frames_output'

# 创建保存文件夹，如果不存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 逐帧读取并保存
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 如果没有读取到帧，结束循环
    
    # 设置输出图像文件名，按照帧数命名
    frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    
    # 保存帧
    cv2.imwrite(frame_filename, frame)
    
    # 更新帧数
    frame_number += 1

    # 打印进度
    print(f"Saving frame {frame_number}/{total_frames}")

# 释放视频捕获对象
cap.release()
print("All frames have been saved.")

# 读取视频帧数
# import cv2

# # 视频文件路径
# video_path = '/home/haoge/ymq1/20160501-1.mp4'

# # 打开视频文件
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Error: Unable to open video file.")
# else:
#     # 获取视频的总帧数
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Total number of frames: {total_frames}")
#     print(f"Frame rate (FPS): {fps}")

# # 释放资源
# cap.release()




