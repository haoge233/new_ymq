# import json
# import cv2
# import os

# # 加载原始的JSON文件
# with open("/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json", "r") as f:
#     poses_data = json.load(f)

# # 保存结果的文件夹
# output_folder = "2016-3-rec-output_images"
# os.makedirs(output_folder, exist_ok=True)

# # 创建一个新字典来保存最小外接矩形信息
# rectangles = {}

# # 处理每个图像
# for filename in os.listdir("/home/haoge/ymq1/score_output_folder_2016-3"):
#     print(filename)
#     if filename.endswith(".jpg"):
#         frame_name = filename.split("_")[0]
#         print(frame_name)
        
#         # 获取该帧对应的姿态数据
#         if frame_name in poses_data:
#             pose = poses_data[frame_name]
#             top_points = pose["top"]
#             # print(top_points)
#             bottom_points = pose["bottom"]
            
#             # 计算top和bottom的最小外接矩形
#             top_x_min = min([point[0] for point in top_points])
#             top_x_max = max([point[0] for point in top_points])
#             top_y_min = min([point[1] for point in top_points])
#             top_y_max = max([point[1] for point in top_points])
            
#             bottom_x_min = min([point[0] for point in bottom_points])
#             bottom_x_max = max([point[0] for point in bottom_points])
#             bottom_y_min = min([point[1] for point in bottom_points])
#             bottom_y_max = max([point[1] for point in bottom_points])
            
#             # 保存最小外接矩形数据
#             rectangles[frame_name] = {
#                 "top_rect": [top_x_min, top_y_min, top_x_max, top_y_max],
#                 "bottom_rect": [bottom_x_min, bottom_y_min, bottom_x_max, bottom_y_max]
#             }
            
#             # 读取图像
#             img = cv2.imread(os.path.join("/home/haoge/ymq1/score_output_folder_2016-3", filename))
            
#             # 绘制矩形
#             cv2.rectangle(img, (int(top_x_min), int(top_y_min)), (int(top_x_max), int(top_y_max)), (0, 255, 0), 2)
#             cv2.rectangle(img, (int(bottom_x_min), int(bottom_y_min)), (int(bottom_x_max), int(bottom_y_max)), (0, 0, 255), 2)
            
#             # 保存图像
#             output_image_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_image_path, img)

# # 将最小外接矩形的坐标保存到新的JSON文件
# with open("rectangles.json", "w") as f:
#     json.dump(rectangles, f, indent=4)

# print("处理完成！")



import os
import json
import cv2
import numpy as np
import glob
import re

# 设置输入输出路径
frame_folder = "/home/haoge/ymq1/score_output_folder_2016-3"  # 视频帧图像文件夹
json_file_path = "/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json"  # 姿态数据 JSON 文件
output_frame_folder = "2016-3-rec-output_images-1"  # 输出图像文件夹
output_json_path = "rectangles.json"  # 输出矩形框数据 JSON 文件

# 创建输出文件夹
os.makedirs(output_frame_folder, exist_ok=True)

# 读取 JSON 文件，加载姿态数据
with open(json_file_path, 'r') as f:
    pose_data = json.load(f)

# 自定义排序函数：提取数字部分并按数字排序
def natural_sort_key(filename):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]

# 计算最小外接矩形
def get_min_area_rect(points):
    # 确保点集非空
    if len(points) > 1:
        # 转换为 Numpy 数组
        points = np.array(points).astype(np.int32)
        # 使用 cv2.minAreaRect 来获取最小外接矩形
        points.astype(np.int32) 
        print(points.dtype)
        rect = cv2.minAreaRect(points)
        # 获取矩形的四个角点
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # 转换为整数
        return box
    else:
        return None

# 保存矩形框数据到 JSON 文件
rectangles = {}

# 获取帧图像文件并按数字排序
img_files = sorted(glob.glob(os.path.join(frame_folder, '*.jp*')), key=natural_sort_key)

for img_file in img_files:
    # 获取当前帧的编号（从文件名中提取）
    frame_number = os.path.splitext(os.path.basename(img_file))[0].split("_")[0]
    # frame_number =img_file.split("_")[0]
    print(frame_number)
    
    # 检查 JSON 数据中是否有该帧的姿态数据
    if frame_number in pose_data:
        # 读取图像
        img = cv2.imread(img_file)
        
        # 获取该帧的姿态数据
        athlete_data = pose_data[frame_number]
        
        # 获取两个运动员的姿态点
        top_points = athlete_data.get("top", [])
        bottom_points = athlete_data.get("bottom", [])
        
        # 如果有两个运动员的姿态点数据
        if len(top_points) > 0 and len(bottom_points) > 0:
            # 计算两个运动员的最小外接矩形
            top_box = get_min_area_rect(top_points)
            bottom_box = get_min_area_rect(bottom_points)
            
            # 在图像上绘制矩形框
            cv2.polylines(img, [top_box], isClosed=True, color=(0, 255, 0), thickness=3)  # 绿色
            cv2.polylines(img, [bottom_box], isClosed=True, color=(0, 0, 255), thickness=3)  # 红色

            # 保存标注后的图像
            output_frame_path = os.path.join(output_frame_folder, f"{frame_number}_annotated.jpg")
            cv2.imwrite(output_frame_path, img)

            # 保存矩形框坐标数据到字典
            rectangles[frame_number] = {
                'top': top_box.tolist(),
                'bottom': bottom_box.tolist()
            }

# 将矩形框数据保存到新的 JSON 文件
with open(output_json_path, 'w') as f:
    json.dump(rectangles, f, indent=4)

print("矩形框已保存到 JSON 文件中，并已生成标注图像。")
