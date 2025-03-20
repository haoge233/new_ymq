import cv2
import json
import numpy as np
import os
import math
import pandas as pd
from openpyxl import Workbook
import glob

def calculate_center_of_mass(rectangle):
    # 计算重心：((x1 + x2) / 2, (y1 + 2 * y2) / 3)
    x1, y1 = rectangle[1]
    x2, y2 = rectangle[3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + 2 * y2) / 3
    return (center_x, center_y)

def calculate_speed(center1, center2, fps):
    # 计算速度：欧几里得距离
    dist = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    return dist * fps  # 速度为距离/时间，即（像素/秒）

def process_round_video(round_dir, frame_range, json_file, fps, xlsx_writer):
    start_frame, end_frame = int(frame_range[0]), int(frame_range[1])
    
    # 读取外接矩形的JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取该回合的帧范围对应的矩形框数据
    round_rectangles = {frame: data.get(str(frame), None) for frame in range(start_frame, end_frame+1)}

    # 处理每一帧并计算速度
    total_speed = 0
    num_frames = len(range(start_frame, end_frame+1))
    speeds = []
    
    # 可视化保存路径
    output_folder = "/home/haoge/ymq1/2016-3-siam-v"
    os.makedirs(output_folder, exist_ok=True)

    # 创建帧数的排序列表，确保按顺序处理帧
    sorted_frames = sorted(round_rectangles.keys())

    # 初始化前一帧的重心
    prev_top_center = None
    prev_bottom_center = None

    # 处理每一帧
    for i, frame_num in enumerate(sorted_frames):
        if round_rectangles.get(frame_num) is None:
            continue
        rect_data = round_rectangles[frame_num]
        
        # 计算两运动员的重心
        top_center = calculate_center_of_mass(rect_data['top'])
        bottom_center = calculate_center_of_mass(rect_data['bottom'])
        
        # 计算前后帧的速度
        if prev_top_center is not None and prev_bottom_center is not None:  # 确保前一帧的重心存在
            # 计算速度
            top_speed = calculate_speed(prev_top_center, top_center, fps)
            bottom_speed = calculate_speed(prev_bottom_center, bottom_center, fps)
            
            avg_speed = (top_speed + bottom_speed) / 2
            speeds.append(avg_speed)
            total_speed += avg_speed

# 可视化部分
            # # 可视化：读取对应的图像
            # img_path = os.path.join(v_dir, f"{frame_num}_annotated.jpg")
            # img = cv2.imread(img_path)
            
            # # 绘制重心：绿色表示上半身，红色表示下半身
            # cv2.circle(img, (int(top_center[0]), int(top_center[1])), 5, (0, 255, 0), -1)  # 上半身
            # cv2.circle(img, (int(bottom_center[0]), int(bottom_center[1])), 5, (0, 0, 255), -1)  # 下半身
            
            # # 绘制速度信息
            # speed_text = f"Avg Speed: {avg_speed:.2f} px/s"
            # cv2.putText(img, speed_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # # 保存图像
            # cv2.imwrite(os.path.join(output_folder, f"frame_{frame_num}_with_centers.jpg"), img)
# 可视化部分

        # 更新前一帧的重心
        prev_top_center = top_center
        prev_bottom_center = bottom_center

    # 计算该回合的平均速度
    avg_speed = total_speed / num_frames if num_frames > 0 else 0

      # 记录回合数据到xlsx（添加回合信息）
    xlsx_writer.append([f"{start_frame}-{end_frame}", total_speed, avg_speed, num_frames])

    print(f"回合 {start_frame}-{end_frame} 平均速度：{avg_speed} pixels/s")

    # # 输出到xlsx
    # wb = Workbook()
    # ws = wb.active
    # ws.append(["Frame", "Top Speed", "Bottom Speed", "Average Speed"])
    # for i, speed in enumerate(speeds):
    #     ws.append([sorted_frames[i], speed, speed, speed])  # 使用排序后的帧号
    
    # ws.append(["", "", "Average Speed", avg_speed])
    # xlsx_output = os.path.join(round_dir, "speed_data.xlsx")
    # wb.save(xlsx_output)
    # print(f"回合 {frame_range[0]} - {frame_range[1]} 平均速度：{avg_speed} pixels/s")


def process_all_rounds(base_path, json_file, fps):
    # 读取所有回合的视频文件，过滤和排序
    video_files = sorted(glob.glob(os.path.join(base_path, '*.mp4')))  # 获取所有mp4文件
    round_videos = []
    
    for video in video_files:
        video_name = os.path.basename(video)
        # 提取视频名称中的帧范围，格式：20160501-3_1863-2364.mp4
        if '_' in video_name and '.mp4' in video_name:
            frame_range_str = video_name.split('_')[-1].replace('.mp4', '')
            start_frame, end_frame = map(int, frame_range_str.split('-'))
            round_videos.append((video, start_frame, end_frame))

    # 根据帧范围排序
    round_videos.sort(key=lambda x: (x[1], x[2]))  # 按照起始帧（start_frame）和结束帧（end_frame）排序
    # print(round_videos)

    # 创建XLSX写入器
    wb = Workbook()
    ws = wb.active
    ws.append(["Frame Range", "Total Speed (px/s)", "Average Speed (px/s)", "Total Frames"])
    xlsx_output = os.path.join('/home/haoge/ymq1', "speed_data.xlsx")

    # 处理每个回合的视频文件
    for video_path, start_frame, end_frame in round_videos:
        round_name = f"{start_frame}-{end_frame}"
        print(f"处理回合 {round_name} ({video_path})")
        
        # # 根据视频文件名推断回合帧范围
        # round_dir = os.path.join(base_path, round_name)  # 假设回合文件夹名称与帧范围一致
        # os.makedirs(round_dir, exist_ok=True)  # 创建回合文件夹
        process_round_video(round_dir, (start_frame, end_frame), json_file, fps,ws)
    # 保存所有回合的数据到XLSX文件
    wb.save(xlsx_output)
    print(f"所有回合的速度数据已保存到 {xlsx_output}")
if __name__ == "__main__":
    round_dir = "/home/haoge/solopose/SoloShuttlePose/res 2016-3/videos/20160501-3"
    v_dir="/home/haoge/ymq1/2016-3-rec-output_images-1"
    json_file = "/home/haoge/ymq1/rectangles.json"
    fps = 1  # 假设帧率为30fps，实际可根据视频调整
    process_all_rounds(round_dir, json_file, fps)
