# import numpy as np

# # 模拟一个包含 101 个图像的差异（布尔数组）
# # 假设0表示小于阈值的图像，1表示大于阈值的图像
# test_array = np.random.choice([True, False], size=101, p=[0.8, 0.2])  # 随机生成True/False数组

# # 打印生成的布尔数组
# print("Test Array:", test_array)

# # 测试功能：按照要求筛选出每个回合的图像
# def filter_rounds(test_array, consecutive_threshold=10):
#     rounds = []  # 用来保存每一回合
#     round_count = 1  # 初始回合计数
#     i = 0  # 当前帧位置
#     diff_greater_than_threshold = 0
#     while i < len(test_array):
#         # 定义当前窗口的右边界
#         window_end = min(i + 10, len(test_array))  # 10为窗口大小
#         # print(window_end)

#         # 用于记录窗口内连续大于阈值的帧数
#         save_frames = []

#         # 检查窗口内每个图像与参考图像的差异
#         for j in range(i, window_end):
#             print(j)
#             if not test_array[j]:  # 差异小于阈值，保存当前帧
#                 save_frames = list(range(i, j+1))  # 保存从左边界到当前图像
#                 break
#             else:
#                 diff_greater_than_threshold += 1

#         # 如果窗口内有超过10张差异大于阈值的图像，则跳过该窗口
#         if save_frames and diff_greater_than_threshold >= consecutive_threshold:
#             print(f"More than {consecutive_threshold} frames with difference greater than threshold, skipping this window.")
#             i = save_frames[-1]   # 更新窗口左边界为当前图像
#             diff_greater_than_threshold =0
#             round_count += 1  # 更新回合计数
#         elif save_frames:
#             # 保存从左边界到差异小于阈值的帧
#             rounds.append(save_frames)
#             print(f"Round {round_count} saved frames: {save_frames}")
#             i = save_frames[-1] + 1  # 更新窗口左边界
#             diff_greater_than_threshold =0
#         else:
#             # 如果没有找到符合条件的帧，直接滑动窗口左边界
#             i = window_end+1

#     return rounds

# # 测试用例：筛选回合
# rounds = filter_rounds(test_array)
# print(f"Total rounds: {len(rounds)}")



# import json

# # JSON 文件路径
# json_file_path = '/home/haoge/solopose/SoloShuttlePose/res/players/player_kp/output_video1.json'

# # 打开并读取 JSON 文件
# with open(json_file_path, 'r') as file:
#     data = json.load(file)

# # 存储帧间隔的结果
# frame_intervals = []
# start_frame = None

# # 遍历字典中的每一项
# for frame, coordinates in data.items():
#     bottom_data = coordinates.get("bottom", None)
#     bottom_data = coordinates.get("bottom", None)

#     # 检查是否有数据，若没有数据则认为该帧为空
#     if bottom_data is None or bottom_data is None:
#         if start_frame is not None:  # 记录有数据的帧间隔
#             frame_intervals.append((start_frame, frame))
#         start_frame = None  # 重置开始帧
#     else:
#         if start_frame is None:  # 如果是第一次有数据
#             start_frame = frame

# # 输出帧间隔和相应的开始帧
# for interval in frame_intervals:
#     print(f"数据采集间隔：从帧 {interval[0]} 到帧 {interval[1]}")



# import cv2
# import numpy as np

# # Bottom keypoints coordinates
# keypoints = [
#     [760.7611694335938, 576.4559936523438],
#     [762.1963500976562, 572.1373901367188],
#     [796.6414794921875, 564.9397583007812],
#     [765.0667724609375, 576.4559936523438],
#     [793.7710571289062, 569.25830078125],
#     [767.937255859375, 611.0044555664062],
#     [818.1696166992188, 602.3673095703125],
#     [756.45556640625, 665.7061767578125],
#     [851.1795043945312, 654.1900024414062],
#     [724.8809204101562, 706.0126953125],
#     [854.0499267578125, 698.8151245117188],
#     [822.475341796875, 706.0126953125],
#     [854.0499267578125, 703.1336669921875],
#     [786.5950317382812, 780.8677978515625],
#     [839.6978149414062, 750.6378173828125],
#     [821.0400390625, 851.4042358398438],
#     [856.9203491210938, 824.0532836914062]
# ]

# # Load the image
# image_path = "/home/haoge/ymq1/keyframes_output/keyframe_10201.jpg"  # Replace with your image path
# output_path = "/home/haoge/ymq1/output.png"  # Path to save the result
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Unable to load image.")
# else:
#     # Annotate keypoints
#     for idx, (x, y) in enumerate(keypoints):
#         # Draw a small circle at each keypoint
#         cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
#         # Annotate point index near the keypoint
#         cv2.putText(image, str(idx), (int(x) + 10, int(y) - 10), 
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
#                     color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

#     # Save the annotated image
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved as {output_path}")

# import numpy as np
# from sklearn.cluster import KMeans

# # 示例数据
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0], [10, 3]])

# # 使用KMeans聚类
# kmeans = KMeans(n_clusters=2, random_state=42).fit(X)

# # 聚类中心
# print("Cluster Centers:")
# print(kmeans.cluster_centers_)

# # 数据点标签
# print("\nData Point Labels:")
# print(kmeans.labels_)

# print(np.bincount(kmeans.labels_))

# # 验证对应关系
# for i, label in enumerate(kmeans.labels_):
#     print(f"Data Point {X[i]} belongs to Cluster {label}, Center: {kmeans.cluster_centers_[label]}")

# import numpy as np
# import json
# import numpy as np

# class PoseDistanceCalculator:
#     def __init__(self, json_data):
#         self.json_data = json_data

#     def extract_features(self, keypoints):
#         """
#         提取姿态的特征（12个角度特征）
#         """
#         angle_definitions = [
#             (5, 6, 11), (5, 11, 7), (6, 5, 12), (6, 12, 8),
#             (7, 5, 9), (8, 6, 10), (11, 5, 12), (11, 12, 13),
#             (12, 6, 11), (12, 11, 14), (13, 11, 15), (14, 12, 16)
#         ]
#         angles = []
#         for a, b, c in angle_definitions:
#             angle = self.calculate_angle(
#                 keypoints[a],
#                 keypoints[b],
#                 keypoints[c]
#             )
#             angles.append(angle)
#         return np.array(angles)

#     def calculate_angle(self, p1, p2, p3):
#         """
#         计算三点之间的夹角
#         """
#         v1 = np.array(p2) - np.array(p1)
#         v2 = np.array(p3) - np.array(p2)
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
#         return np.degrees(angle)

#     def compute_distances(self, frame_ids):
#         """
#         计算选定帧之间的特征距离
#         """
#         # 提取选定帧的关键点
#         keypoints_list = []
#         for frame_id in frame_ids:
#             if frame_id in self.json_data:
#                 bottom_keypoints = np.array(self.json_data[frame_id]['bottom'])
#                 if bottom_keypoints.size > 1:
#                     keypoints_list.append(bottom_keypoints)
#                 else:
#                     print(f"Frame {frame_id} has no valid keypoints.")
#                     return None
#             else:
#                 print(f"Frame {frame_id} not found in JSON data.")
#                 return None

#         # 提取特征
#         features = [self.extract_features(kp) for kp in keypoints_list]

#         # 计算第一帧与其他帧的欧几里得距离
#         base_feature = features[0]
#         distances = []
#         for i, feature in enumerate(features[1:], start=2):  # 从第二帧开始
#             distance = np.linalg.norm(base_feature - feature)
#             distances.append((frame_ids[0], frame_ids[i - 1], distance))
#         return distances


# # 假设我们有一个JSON文件
# json_file_path = '/home/haoge/solopose/SoloShuttlePose/res vedio1/players/player_kp/output_video1.json'

# # 读取JSON数据
# with open(json_file_path, 'r') as f:
#     json_data = json.load(f)

# # 实例化类
# pose_calculator = PoseDistanceCalculator(json_data)

# # 选定三帧的 Frame IDs
# frame_ids = ['9136', '10222', '11976']  # 替换为实际的 Frame ID

# # 计算距离
# distances = pose_calculator.compute_distances(frame_ids)

# # 输出结果
# if distances:
#     for frame_a, frame_b, distance in distances:
#         print(f"Distance between {frame_a} and {frame_b}: {distance}")



#  -----------------------------静态相似度计算公式
# import numpy as np
# import math

# def calculate_score(class_scores, class_similarities, point_similarities, a=1/2, b=1/2):
#     """
#     计算给定点的分数。

#     参数:
#     class_scores (list of int/float): 每个类的类中心点得分。
#     class_similarities (list of list of int/float): 每个类的类中心点的相似度。
#     point_similarities (list of int/float): 给定点的相似度。
#     a (int/float): 类内得分的权重系数。
#     b (int/float): 类外得分的权重系数。

#     返回:
#     float: 给定点的总分。
#     """
#     # 找到最大相似度及其对应的类索引
#     max_similarity = max(point_similarities)
#     class_index = np.argmax(point_similarities)
#     print(class_index)



#     # 计算类外得分
#     out_class_score = 0
#     flag=1
#     for i, class_score in enumerate(class_scores):
#         if i != class_index:
#             similarity_difference = point_similarities[i] - class_similarities[class_index][i]  # 假设相似度在0到1之间
#             # print(similarity_difference)
#             angle=similarity_difference/100
#             # angle = similarity_difference*abs(similarity_difference) / (class_similarities[class_index][i]*max(abs(similarity_difference),20))
#             print(angle)
#             # angle = max(min(angle, math.pi / 2), -math.pi / 2)
#             # similarity_difference = max(0, similarity_difference)  # 确保差异不为负
#             # out_class_score += 1/200*(point_similarities[i] + class_similarities[class_index][i])*(similarity_difference / class_similarities[class_index][i]) * (class_score - class_scores[class_index]) * b
#             # out_class_score += math.sin(angle) * (class_score - class_scores[class_index]) * b
#             out_class_score += angle * (class_score - class_scores[class_index]) * b
#             print("out")
#             print(math.sin(angle))
#             # print(math.sin(angle) * (class_score - class_scores[class_index]) * b)
#             print(angle * (class_score - class_scores[class_index]) * b)
#             print(out_class_score)

#     # 计算总分
#     if out_class_score < -5 :
#         flag=-1
#         # 计算类内得分
#     in_class_score = class_scores[class_index]+flag*(100 / max_similarity-1) * class_scores[class_index] * a
#     print("in")
#     print(in_class_score)
#     total_score = in_class_score   + out_class_score 
#     return total_score

# # 示例数据
# class_scores = [10, 20,30,40,50]
# class_similarities = [[100, 50,30,30,10], [50, 100,40,20,20],[30,40,100,50,10],[10,20,50,100,30],[10,20,10,30,100]]  # 这里只是示例，实际中可能不需要这个矩阵，因为我们已经有了point_similarities
# point_similarities =[70, 80,40,20,20]  # 给定点的相似度

# # 计算分数
# a = 1/2
# b = 1
# score = calculate_score(class_scores, class_similarities, point_similarities, a, b)
# print(f"给定点的总分: {score}")

# # 生成随机相似度的点进行计算
# # np.random.seed(0)  # 设置随机种子以便复现结果
# random_point_similarities = np.random.randint (0, 100, size=len(class_scores))
# print(f"随机点的相似度: {random_point_similarities}")
# random_score = calculate_score(class_scores, class_similarities, random_point_similarities, a, b)
# print(f"随机点的总分: {random_score}")



import json
import numpy as np
import math
import cv2


connections = [
    (11, 5),
    (12, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (4, 11),
    (4, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
    # 注意：如果原编号中有缺失或重复，需要相应地调整这里的索引
]

def callen(keypoints_list):
    # 计算线段长度的函数
    def calculate_distance(coord1, coord2):
        return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    # 存储线段长度的列表
    distances = []
    
    # 遍历连接表格并计算线段长度
    coordinates=keypoints_list[0]
    print("cor")
    print(coordinates)
    for (node1, node2) in connections:
        # 注意：这里假设坐标列表中的索引与关节点的原编号一一对应
        # 如果实际情况不是这样，请相应地调整索引方式
        distance = calculate_distance(coordinates[node1], coordinates[node2]) # 减1是因为列表索引从0开始
        distances.append(distance)

            # 存储线段长度的列表
    distances1 = []
    
    # 遍历连接表格并计算线段长度
    coordinates1=keypoints_list[1]
    print("cor1")
    print(coordinates1)
    for (node1, node2) in connections:
        # 注意：这里假设坐标列表中的索引与关节点的原编号一一对应
        # 如果实际情况不是这样，请相应地调整索引方式
        distance1 = calculate_distance(coordinates1[node1], coordinates1[node2]) # 减1是因为列表索引从0开始
        distances1.append(distance1)
    
    # 打印线段长度
    for i, distance in enumerate(distances, start=1):
        print(f"线段{i}的长度: {distance}")

    for i, distance in enumerate(distances1, start=1):
        print(f"线段{i}的长度: {distance}")

    return distances,distances1,coordinates,coordinates1









def compute_distances(frame_ids,json_data):
        """
        计算选定帧之间的特征距离
        """
        # 提取选定帧的关键点
        keypoints_list = []
        for frame_id in frame_ids:
            if frame_id in json_data:
                bottom_keypoints = np.array(json_data[frame_id]['bottom'])
                print(bottom_keypoints)
                bottom_keypoints[4]=(bottom_keypoints[11]+bottom_keypoints[12])*1/2
                print(bottom_keypoints)
                if bottom_keypoints.size > 1:
                    keypoints_list.append(bottom_keypoints)
                else:
                    print(f"Frame {frame_id} has no valid keypoints.")
            else:
                print(f"Frame {frame_id} not found in JSON data.")
                return None

        # 提取特征
        return keypoints_list


# 假设我们有一个JSON文件
json_file_path = '/home/haoge/solopose/SoloShuttlePose/res vedio1/players/player_kp/output_video1.json'

# 读取JSON数据
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# 计算帧 19186 与所有模板帧的相似度:
#   与模板帧 9136 相似度 D = 20.942026325734393
#   与模板帧 11976 相似度 D = 40.47571925653505
#   帧 19186 与所有模板的相似度: [20.942026325734393, 40.47571925653505]

# 选定三帧的 Frame IDs
frame_ids = ['9136', '19186','11976']  # 替换为实际的 Frame ID
# 7230-9136:动作相似度汇总距离D: 24.43774950419479
# 7230-11976:30.34190260487305
# 计算距离 
distances = compute_distances(frame_ids,json_data)
template_lengths,target_lengths,coordinates,coordinates1=callen(distances)
# print(template_lengths)
# print(type(template_lengths))
# # 输出结果
# if distances:
#     for frame_a, frame_b, distance in distances:
#         print(f"Distance between {frame_a} and {frame_b}: {distance}")







# 计算变换后的坐标增量 M = (Δx, Δy)
def calculate_transforms(template_lengths, target_lengths):
    transforms = []
    for i in range(len(template_lengths)):
        # 计算比例西塔（θ）
        if(target_lengths[i]==0):
            theta = 0
        else:
            theta = template_lengths[i] / target_lengths[i] -1
            
            # 获取当前线段的坐标
            # x_start, y_start, x_end, y_end = segments_coords[i]

            x_end = coordinates1[connections[i][1]][0]

            x_start=coordinates1[connections[i][0]][0]

            y_end = coordinates1[connections[i][1]][1]

            y_start = coordinates1[connections[i][0]][1]

            print(theta,x_end,x_start,y_end,y_start)
        
        # 计算Δx和Δy
        delta_x = (x_end - x_start) * theta
        delta_y = (y_end - y_start) * theta
        
        # 存储变换结果
        transforms.append((delta_x, delta_y))
    
    return transforms

# 调用函数并打印结果
transforms = calculate_transforms(template_lengths, target_lengths)
for i, (delta_x, delta_y) in enumerate(transforms, start=1):
    print(f"M{i} = ({delta_x}, {delta_y})")





# 根据表2和变换增量计算每个点的坐标
def calculate_coords(table, transforms):
    coords = coordinates1.copy()
    for point, path in table.items():
        # current_pos = initial_coords[5]  # 从根节点5开始
        current_pos = coords[point]
        path_segments = path.split('-')[1:]
        for segment in path_segments:
            current_pos[0] +=transforms[int(segment)-5][0]
            current_pos[1] += transforms[int(segment)-5][1]
            # current_pos += (current_pos[0] + transforms[int(segment)-6][0],
            #                current_pos[1] + transforms[int(segment)-6][1])
        coords[point] = current_pos
    return coords

# 使用上述函数计算坐标
calculated_coords = calculate_coords({
    11: '4-11',
    12: '4-12',
    5: '4-11-5',
    6: '4-12-6',
    7: '4-11-5-7',
    9: '4-11-5-7-9',
    8: '4-12-6-8',
    10: '4-12-6-10',  # 假设有一个从6到10的路径段增量，尽管表2中没有直接给出
    13: '4-11-13',
    14: '4-12-14',
    15: '4-11-13-15',  # 假设有一个从13到15的路径段增量
    16: '4-12-14-16',  # 假设有一个从14到16的路径段增量
},  transforms)

# 打印计算结果
print(calculated_coords)
print(coordinates1)


# 效果展示  
# image_path = "/home/haoge/ymq1/keyframes_output/keyframe_10223.jpg"  # Replace with your image path
# output_path = "/home/haoge/ymq1/output.png"  # Path to save the result
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Unable to load image.")
# else:
#     # Annotate keypoints
#     for idx, (x, y) in enumerate(calculated_coords):
#         # Draw a small circle at each keypoint
#         cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
#         # Annotate point index near the keypoint
#         cv2.putText(image, str(idx), (int(x) + 10, int(y) - 10), 
#                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
#                     color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

#     # Save the annotated image
#     cv2.imwrite(output_path, image)
#     print(f"Annotated image saved as {output_path}")




# 假设这是目标骨架（a）的关节点坐标，每个点都是一个三维坐标
a_joints = calculated_coords

# 假设这是源骨架（t）的关节点坐标
t_joints = coordinates

# 选择基准点，这里选择第一个点（脊椎点j1）
j1_a = a_joints[4]
j1_t = t_joints[4]

# 计算平移向量
translation_vector = j1_t - j1_a
print(translation_vector)

# 应用平移向量到源骨架的所有关节点
a_joints_translated = a_joints + translation_vector

# 输出平移后的源骨架关节点坐标
print("Translated Joints of Skeleton (a):")
print(a_joints_translated)


image_path = "/home/haoge/ymq1/keyframes_output/keyframe_9137.jpg"  # Replace with your image path
output_path = "/home/haoge/ymq1/output.png"  # Path to save the result
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
else:
    # Annotate keypoints
    for idx, (x, y) in enumerate(a_joints_translated):
        # Draw a small circle at each keypoint
        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        # Annotate point index near the keypoint
        cv2.putText(image, str(idx), (int(x) + 10, int(y) - 10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                    color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as {output_path}")




a_joints = a_joints_translated

a_joints = a_joints[5:]

t_joints = t_joints[5:]



def calculate_w(template_lengths):
    W = []
    template_sumlen = sum(template_lengths)
    for i in range(len(connections)):
        # 计算比例西塔（θ）
        w= template_lengths[i] / template_sumlen
        
        # 存储变换结果
        W.append(w)
    
    return W



W=calculate_w(template_lengths)
print(W)

# 计算对应关节点的距离并加权求和得到汇总距离D
distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和

print(f"动作相似度汇总距离D: {D}")

# 使用 Sigmoid 公式计算相似度
def sigmoid_similarity(D):
    return 2 / (1 + math.exp(D / 23.0))
 
# 计算相似度
similarity = sigmoid_similarity(D)
 
# 输出相似度结果
print(f"两个动作的相似度为: {similarity:.4f}")
