import math
import numpy as np

# 计算两个向量之间的角度（返回角度，单位：度）
def calculate_angle(ax, ay, bx, by):
    # 计算向量AB与横轴的夹角
    dot_product = ax * bx + ay * by
    magnitude_a = math.sqrt(ax ** 2 + ay ** 2)
    magnitude_b = math.sqrt(bx ** 2 + by ** 2)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_deg

# 计算弧长
def calculate_arc_length(radius, angle_deg):
    # 将角度转换为弧度
    angle_rad = math.radians(angle_deg)
    # 弧长公式 L = r * θ
    arc_length = radius * angle_rad
    return arc_length

def sector_area(radius, angle_deg):
    # 角度转换为弧度
    angle_rad = math.radians(angle_deg)
    # 扇形面积公式： A = 1/2 * r^2 * θ
    area = 0.5 * radius**2 * angle_rad
    return area

# 输入数据
A = (429, 1020)  # 横轴的第一个点
B = (500, 1020)  # 横轴的第二个点
points = [(448, 971), (531, 763), (590, 610), (634, 495)]  # 4个点

# 计算横轴与每个线段的角度
angles = []
for p in points:
    angle = calculate_angle(A[0] - B[0], A[1] - B[1], A[0] - p[0], A[1] - p[1])
    angles.append(angle)

# 计算弧长
arc_lengths = []
for p in points:
    # 计算半径，即横轴的第一个点与给定点之间的距离
    radius = math.sqrt((A[0] - p[0]) ** 2 + (A[1] - p[1]) ** 2)
    # 计算弧长
    arc_length = sector_area(radius, angles[points.index(p)])
    arc_lengths.append(arc_length)

# 输出结果
for i in range(len(points)):
    print(f"线段 {A} -> {points[i]} 与横轴的角度: {angles[i]:.2f}°")
    print(f"弧长 (圆心在 {A}，半径为 {math.sqrt((A[0] - points[i][0]) ** 2 + (A[1] - points[i][1]) ** 2)}): {arc_lengths[i]:.2f}单位")
