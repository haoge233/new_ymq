import cv2
import numpy as np
import json

# 读取原始的 JSON 文件
with open('/home/haoge/solopose/SoloShuttlePose/res 2016-1/players/player_kp/20160501-1.json', 'r') as f:
    json_data = json.load(f)

# 目标矩形大小
width = 1101
height = 2848
rect_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

# 透视变换的源矩阵
# 输入图像的四个角点：梯形区域的四个顶点
trapezoid_points = np.array([
    [642, 481],
    [1309, 481],
    [1526, 1027],
    [425, 1027]
], dtype=np.float32)

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(trapezoid_points, rect_points)

# 变换函数
def transform_points(points, matrix):
    points = np.array([points], dtype=np.float32)
    # print(points)
    transformed_points = cv2.perspectiveTransform(points, matrix)
    return transformed_points[0]

# 处理 JSON 数据中的每一组
for key, value in json_data.items():
    if value["top"] is not None and value["bottom"] is not None:
        # 提取 top 和 bottom 坐标
        top_points = np.array(value["top"], dtype=np.float32)
        bottom_points = np.array(value["bottom"], dtype=np.float32)
        
        # 进行透视变换
        # print(top_points)
        transformed_top = transform_points(top_points, matrix)
        transformed_bottom = transform_points(bottom_points, matrix)
        
        # 保存变换后的坐标
        value["top"] = transformed_top.tolist()
        value["bottom"] = transformed_bottom.tolist()

# 将变换后的数据保存到新的 JSON 文件
with open('transformed_coordinates.json', 'w') as f:
    json.dump(json_data, f, indent=4)

print("透视变换完成，变换后的坐标已保存到 'transformed_coordinates.json'")
