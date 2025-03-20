# import cv2
# import numpy as np

# # 读取图像
# image = cv2.imread('/home/haoge/ymq1/20160501-1_10903.png')

# # 1. 预处理：转换为灰度图像并高斯模糊
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 2. 边缘检测：使用Canny边缘检测
# edges = cv2.Canny(blurred, 50, 150)

# # 3. 寻找轮廓
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 4. 筛选矩形轮廓
# for contour in contours:
#     # 使用逼近多边形算法
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     # 如果找到四个顶点，说明是矩形
#     if len(approx) == 4:
#         # 获取矩形的四个顶点
#         points = approx.reshape(4, 2)
        
#         # 绘制矩形（可以选择绘制在原图上）
#         cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

#         # 可以对检测到的矩形进行透视变换，将球场区域映射到一个平面上
#         # 假设points为四个角，按照顺序排序后可以进行透视变换
#         pts1 = np.float32(points)
#         # 目标位置设定为矩形的标准位置，例如一个矩形网格
#         width = 640
#         height = 480
#         pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

#         # 获取透视变换矩阵
#         matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
#         # 透视变换
#         warped_image = cv2.warpPerspective(image, matrix, (width, height))

#         # 保存变换后的图像
#         cv2.imwrite('warped_badminton_court.jpg', warped_image)

# # 保存带有矩形框的原图
# cv2.imwrite('detected_court.jpg', image)


# import cv2
# import numpy as np

# # 读取图像
# image = cv2.imread('/home/haoge/ymq1/20160501-1_10903.png')

# # 1. 预处理：转换为灰度图像并高斯模糊
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # 2. 边缘检测：使用Canny边缘检测
# edges = cv2.Canny(blurred, 50, 150)

# # 3. 寻找轮廓
# contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 初始化列表，用来存储每个矩形的面积和轮廓
# rectangles = []

# # 4. 筛选矩形轮廓
# for contour in contours:
#     # 使用逼近多边形算法
#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     # 如果找到四个顶点，说明是矩形
#     if len(approx) == 4:
#         # 计算矩形的面积
#         area = cv2.contourArea(contour)
#         rectangles.append((area, approx))

# # 5. 按照面积降序排序
# rectangles = sorted(rectangles, key=lambda x: x[0], reverse=True)

# # 6. 绘制前五个最大的矩形，使用不同的颜色
# colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # 预定义颜色
# for i in range(min(5, len(rectangles))):  # 确保不超过5个矩形
#     area, approx = rectangles[i]
#     cv2.drawContours(image, [approx], -1, colors[i], 2)

# # 7. 保存包含前五个矩形框的图像
# cv2.imwrite('detected_top_5_courts.jpg', image)


# import cv2
# import numpy as np

# # 1. 定义等腰梯形的四个顶点（上边和下边）
# trapezoid_points = np.array([[642, 484], [1305, 484],  [1533, 1034],[426, 1034]], dtype=np.float32)

# # 2. 目标底边的坐标（映射到底边的两端点）
# # 我们希望将上边左顶点映射到底边的左端点
# bottom_left = [426, 1034]   # 底边左端点
# bottom_right = [1533, 1034]  # 底边右端点

# # 使用梯形的三个顶点来计算仿射变换
# # 这里使用上边的左顶点、右顶点和下边左顶点（不包括下边右顶点）
# source_points = np.array([trapezoid_points[0], trapezoid_points[1], trapezoid_points[3]], dtype=np.float32)
# target_points = np.array([bottom_left, bottom_right, bottom_left], dtype=np.float32)

# # 3. 计算仿射变换矩阵
# matrix = cv2.getAffineTransform(source_points, target_points)

# # 4. 映射任意点到目标底边
# # 假设你要映射的点是(1000, 600)
# point_to_map = np.array( [
#             1387,
#             682
#         ], dtype=np.float32)
# mapped_point = cv2.transform(np.array([[point_to_map]]), matrix)

# # 打印映射结果
# print(f'Original point: {point_to_map}')
# print(f'Mapped point: {mapped_point[0][0]}')

# # 可视化（可选）
# # 画出原始梯形和变换后的底边
# image = np.zeros((1500, 2000, 3), dtype=np.uint8)
# cv2.polylines(image, [trapezoid_points.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
# cv2.polylines(image, [np.array([bottom_left, bottom_right], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

# # 画出映射的点
# cv2.circle(image, tuple(point_to_map.astype(int)), 10, (255, 0, 0), -1)
# cv2.circle(image, tuple(mapped_point[0][0].astype(int)), 10, (0, 0, 255), -1)
# # 显示图像
# cv2.imwrite("Trapezoid Mapping.jpg", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# import cv2
# import numpy as np

# # 1. 定义等腰梯形的四个顶点（上边和下边）
# trapezoid_points = np.array([[
#         642,
#         476
#     ],
#     [
#         1308,
#         476
#     ],

#     [
#         424,
#         1024
#     ],
#     [
#         1527,
#         1024
#     ]], dtype=np.float32)
# trapezoid_points1 = np.array([[
#         642,
#         476
#     ],
#     [
#         1308,
#         476
#     ],

#     [
#         1527,
#         1024
#     ],    
#     [
#         424,
#         1024
#     ],], dtype=np.float32)

# # 2. 目标底边的坐标（映射到底边的两端点）
# # 我们希望将梯形的四个顶点映射到目标矩形
# bottom_left = [424, 1024]   # 底边左端点
# bottom_right = [1527, 1024]  # 底边右端点
# target_points = np.array([bottom_left, bottom_right, bottom_left, bottom_right], dtype=np.float32)

# # 3. 计算透视变换矩阵
# # 使用透视变换函数 getPerspectiveTransform 来计算变换矩阵
# matrix = cv2.getPerspectiveTransform(trapezoid_points, target_points)

# # 4. 映射任意点到目标底边
# # 假设你要映射的点是(1000, 600)
# point_to_map = np.array([
#             975,
#             912
#         ], dtype=np.float32)
# mapped_point = cv2.perspectiveTransform(np.array([[point_to_map]]), matrix)

# # 打印映射结果
# print(f'Original point: {point_to_map}')
# print(f'Mapped point: {mapped_point[0][0]}')

# # 可视化（可选）
# # 画出原始梯形和变换后的底边
# image = np.zeros((1500, 2000, 3), dtype=np.uint8)
# cv2.polylines(image, [trapezoid_points1.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
# cv2.polylines(image, [np.array([bottom_left, bottom_right], dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

# # 画出映射的点
# cv2.circle(image, tuple(point_to_map.astype(int)), 10, (255, 0, 0), -1)
# cv2.circle(image, tuple(mapped_point[0][0].astype(int)), 10, (0, 0, 255), -1)

# # 显示图像
# cv2.imwrite("Trapezoid Mapping.jpg", image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

import cv2
import numpy as np

# 目标矩阵大小 (宽度, 高度)
width = 1101
height = 2848

# 输入图像的四个角点：梯形区域的四个顶点
# trapezoid_points = np.array([       [
#             642,
#             481
#         ],
#         [
#             1309,
#             481
#         ],

#                 [
#             1526,
#             1027
#         ],
#                 [
#             425,
#             1027
#         ]
#         ], dtype=np.float32)


trapezoid_points = np.array([       [
            642,
            484
        ],
        [
            1305,
            484
        ],
        [
            1533,
            1034
        ],
                [
            426,
            1034
        ]
        ], dtype=np.float32)

# 目标矩形的四个顶点
rect_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# # 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(trapezoid_points, rect_points)

# # 假设我们有一个需要变换的点
# # 例如，梯形中的一个点，(x, y)
# # point = np.array([[448,971],[531,763],[590,610],[634,495]], dtype=np.float32)  # 使用一个点：右上角
# point = np.array([[953,319],[947,560],[731,621],[736,661]], dtype=np.float32)
# # 将点从梯形坐标系转换到目标矩形坐标系
# # 需要使用 perspectiveTransform 来变换坐标
# transformed_point = cv2.perspectiveTransform(np.array([point], dtype=np.float32), matrix)

# # 输出结果
# print(f"原始点: {point}")
# print(f"变换后的点: {transformed_point}")

def court_trans(point,matrixs=matrix):
    transformed_point = cv2.perspectiveTransform(np.array([point], dtype=np.float32), matrixs)
    return transformed_point




