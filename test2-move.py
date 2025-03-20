

import json
import numpy as np
import math
import cv2
import os
import re
from court import court_trans
from draw_connections import draw_diff


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

# ----------------------------------------------------------------

def movea2t(calculated_coords,coordinates):


    # 假设这是目标骨架（a）的关节点坐标，每个点都是一个三维坐标
    a_joints = calculated_coords

    # 假设这是源骨架（t）的关节点坐标
    t_joints = coordinates

    # 选择基准点，这里选择第一个点（脊椎点j1）
    j1_a = a_joints[4]
    j1_t = t_joints[4]

    # 计算平移向量
    translation_vector = j1_t - j1_a
    # print(translation_vector)

    # 应用平移向量到源骨架的所有关节点
    a_joints_translated = a_joints + translation_vector

    # # 输出平移后的源骨架关节点坐标
    # print("Translated Joints of Skeleton (a):")
    # print(a_joints_translated)

    return a_joints_translated

def compute_distances(frame_ids,json_data):
        """
        计算选定帧之间的特征距离
        """
        # 提取选定帧的关键点
        keypoints_list = []
        for frame_id in frame_ids:
            if str(frame_id) in json_data:
                bottom_keypoints = np.array(json_data[str(frame_id)]['bottom'])
                # print(bottom_keypoints)
                if bottom_keypoints.size > 1:
                    bottom_keypoints[4]=(bottom_keypoints[11]+bottom_keypoints[12])*1/2
                    # print(bottom_keypoints)
                    trans=court_trans(bottom_keypoints)
                    trans=trans[0]
                    # print(trans)
                    trans_bottom_keypoints=movea2t(bottom_keypoints,trans)
                    keypoints_list.append(trans_bottom_keypoints)
                    # keypoints_list.append(bottom_keypoints)
                    # print(len(keypoints_list))
                # else:
                    # print(f"Frame {frame_id} has no valid keypoints.")
            else:
                print(f"Frame {frame_id} not found in JSON data.")
                return None

        # 提取特征
        return keypoints_list
def calculate_distance(coord1, coord2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

def callen(template_coordinates, target_coordinates):
    """计算模板帧和目标帧之间的线段长度"""
    def calculate_line_lengths(coordinates):
        """计算关节点之间的线段长度"""
        distances = []
        for (node1, node2) in connections:
            distance = calculate_distance(coordinates[node1], coordinates[node2])
            distances.append(distance)
        return distances

    # 计算模板帧和待计算帧的线段长度
    template_lengths = calculate_line_lengths(template_coordinates)
    target_lengths = calculate_line_lengths(target_coordinates)

    return template_lengths, target_lengths


# 计算变换后的坐标增量 M = (Δx, Δy)
def calculate_transforms(template_lengths, target_lengths, coordinates1):
    transforms = []
    for i in range(len(template_lengths)):
        if(target_lengths[i]==0):
            theta = 0
                    # 计算Δx和Δy
            delta_x = 0
            delta_y =0
        else:
            # 计算比例西塔（θ）
            theta = template_lengths[i] / target_lengths[i] -1
            
            # 获取当前线段的坐标
            # x_start, y_start, x_end, y_end = segments_coords[i]

            x_end = coordinates1[connections[i][1]][0]

            x_start=coordinates1[connections[i][0]][0]

            y_end = coordinates1[connections[i][1]][1]

            y_start = coordinates1[connections[i][0]][1]

            # print(theta,x_end,x_start,y_end,y_start)
        
            # 计算Δx和Δy
            delta_x = (x_end - x_start) * theta
            delta_y = (y_end - y_start) * theta
        
        # 存储变换结果
        transforms.append((delta_x, delta_y))
    
    return transforms



# 根据表2和变换增量计算每个点的坐标
def calculate_coords(table, transforms, coordinates1):
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



tables={
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
}


# def movea2t(calculated_coords,coordinates):


#     # 假设这是目标骨架（a）的关节点坐标，每个点都是一个三维坐标
#     a_joints = calculated_coords

#     # 假设这是源骨架（t）的关节点坐标
#     t_joints = coordinates

#     # 选择基准点，这里选择第一个点（脊椎点j1）
#     j1_a = a_joints[4]
#     j1_t = t_joints[4]

#     # 计算平移向量
#     translation_vector = j1_t - j1_a
#     # print(translation_vector)

#     # 应用平移向量到源骨架的所有关节点
#     a_joints_translated = a_joints + translation_vector

#     # # 输出平移后的源骨架关节点坐标
#     # print("Translated Joints of Skeleton (a):")
#     # print(a_joints_translated)

#     return a_joints_translated,a_joints,t_joints






def calculate_w(template_lengths):
    W = []
    template_sumlen = sum(template_lengths)
    for i in range(len(connections)):
        # 计算比例西塔（θ）
        w= template_lengths[i] / template_sumlen
        
        # 存储变换结果
        W.append(w)
    
    return W


# 读取回合分割文件，获取帧范围
def load_rounds(round_file_folder):
    """
    读取回合分割文件，返回一个字典，key 为回合编号，value 为帧序列范围（元组）。
    """
    valid_video_pattern = re.compile(r'output+_(\d+)-(\d+)\.mp4')
    rounds = {}
    for filename in os.listdir(round_file_folder):
        match = valid_video_pattern.match(filename)
    
        if match:
            # 假设文件名格式为：20160501-1_12636-12913.mp4
            frame_range = filename.split('_')[1].split('.')[0]
            start_frame, end_frame = map(int, frame_range.split('-'))
            rounds[(start_frame, end_frame)] = (start_frame, end_frame)
    return rounds

# 获取当前帧的前一帧
def get_previous_frame(current_frame, all_frames, round_frames, json_data,max_frame_distance=100):
    """
    查找当前帧的前一帧。如果当前帧是回合的第一帧，则返回 None。
    否则，在回合范围内查找距离当前帧不超过 max_frame_distance 的最接近帧。
    """
    if current_frame == round_frames[0]:  # 当前帧是回合的第一帧
        return None

    # 查找距离当前帧不超过 max_frame_distance 的最接近的帧
    previous_frame = None
    min_distance = float('inf')
    
    # for frame in range(current_frame-5, max(current_frame - max_frame_distance, round_frames[0])-1, -5):#top-5
    for frame in range(current_frame-1, max(current_frame - max_frame_distance, round_frames[0])-1, -1):
        if str(frame) in all_frames and compute_distances([frame],json_data):
            # 如果该帧已经计算过，返回该帧
            previous_frame = frame
            break
        

    return previous_frame


# def get_similarity(json_file_path,frame_ids,output_file,min_similarity_file):
    
#     # 读取 JSON 数据
#     with open(json_file_path, 'r') as f:
#         json_data = json.load(f)

    


#     # 获取所有帧的关键点数据
#     all_frames_keypoints = list(json_data.keys())  # 获取所有帧ID



#     # 存储每一帧及其相似度
#     frame_similarity = {}
#     min_similarity_idx = []  # 存储每帧最大相似度的模板索引


#     # 对每一帧计算与所有模板帧的相似度
#     for frame_id in all_frames_keypoints:
#         # 获取当前帧的关键点数据
#         distances = compute_distances([frame_id], json_data)
#         if distances:
#             similarity_list = []  # 用于存储与各模板帧的相似度
#             print(f"计算帧 {frame_id} 与所有模板帧的相似度:")
#             # 对每个模板帧计算与当前帧的相似度
#             for idx, keypoints in enumerate(template_keypoints):
#                 # 在此处，template_keypoints[idx][0] 是模板帧，distances[0] 是待计算帧
#                 template_lengths, target_lengths = callen(keypoints, distances[0])  # 模板帧与待计算帧进行对比
#                 transforms = calculate_transforms(template_lengths, target_lengths,distances[0])
#                 calculated_coords = calculate_coords(tables,transforms, distances[0])
#                 # a_joints_translated,a_joints,t_joints=movea2t(calculated_coords,keypoints)
#                 a_joints = calculated_coords

#                 a_joints = a_joints[5:]

#                 t_joints = keypoints[5:]
#                 W=calculate_w(template_lengths)
#                 # 计算对应关节点的距离并加权求和得到汇总距离D
#                 distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
#                 D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和
#                 similarity_list.append(D)
#                 print(f"  与模板帧 {frame_ids[idx]} 相似度 D = {D}")
#             if similarity_list is None:
#                 continue
#             # 保存当前帧的相似度列表
#             frame_similarity[frame_id] = similarity_list
#             min_idx = np.argmin(similarity_list)
#             min_similarity_idx.append((frame_id, min_idx))  # 记录最大相似度的索引
#             print(f"  帧 {frame_id} 与所有模板的相似度: {similarity_list}")
#             print(f"  最大相似度的模板索引: {min_idx}\n")


#     # 自定义序列化函数，将 numpy 数据类型转换为标准 Python 数据类型
#     def convert_to_serializable(obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()  # 将 numpy 数组转换为列表
#         else:
#             raise TypeError(f"Type {type(obj)} not serializable")

#     # 保存相似度结果到文件
#     # output_file = 'frame_similarity_results.json'
#     with open(output_file, 'w') as f:
#         json.dump(frame_similarity, f, indent=4, default=convert_to_serializable)

#     # 保存最大相似度索引到文件
#     # min_similarity_file = 'min_similarity_indices.json'
#     with open(min_similarity_file, 'w') as f:
#         json.dump(min_similarity_idx, f, indent=4, default=convert_to_serializable)


# 设定路径
round_file_folder = '/home/haoge/solopose/SoloShuttlePose/res 20160430-2/videos/output'
json_file_path = '/home/haoge/solopose/SoloShuttlePose/res 20160430-2/players/player_kp/output.json'
json_file_path1 = '/home/haoge/ymq1/20160430-json/frame_similarity_results.json'

# output_folder = 'score_output_folder_2016-1'

with open(json_file_path1, 'r') as f:
        json_data1 = json.load(f)

with open(json_file_path, 'r') as f:
        json_data = json.load(f)


# current_keypoints = compute_distances([18598], json_data)
# # print(current_keypoints)
# current_keypoints1 = compute_distances([18599], json_data)
# draw_diff(current_keypoints[0],current_keypoints1[0])
# print(current_keypoints)
# current_keypoints1=current_keypoints1[0]
# current_keypoints=current_keypoints[0]
# template_lengths,_=callen(current_keypoints,current_keypoints1)
# print(template_lengths)
# a_joints = current_keypoints1[5:]

# t_joints = current_keypoints[5:]
# W=calculate_w(template_lengths)
# print(W)
# # 计算对应关节点的距离并加权求和得到汇总距离D
# distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
# D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和
# print(D)


# 正则表达式匹配有效的视频文件名，格式为 output_video1_7245-7565.mp4
valid_video_pattern = re.compile(r'output+_(\d+)-(\d+)\.mp4')



rounds = load_rounds(round_file_folder)
print(rounds)

# 获取所有帧的关键点数据
all_frames_keypoints = list(json_data1.keys())  # 获取所有帧ID

# print(all_frames_keypoints)

# 存储每一帧与前一帧的距离
frame_distances = {}

# 存储最大相似度的索引
min_similarity_idx = []

similarity_list=[]
# 只对符合正则表达式的文件进行排序
# print(rounds)
sorted_rounds = sorted(rounds)  # 根据起始值排序
# sorted_rounds_dict = dict(sorted_rounds)
print(sorted_rounds)

for round_range in sorted_rounds:
    round_start, round_end = round_range
    print(f"开始计算回合帧范围：{round_start} 到 {round_end}")

    previous_frame = None

    # 对每一帧计算与前一帧的骨骼点距离
    for frame_id in range(round_start, round_end+1):
        print(frame_id)
        if str(frame_id) not in all_frames_keypoints:
            continue  # 跳过没有关键点数据的帧
        # print(frame_id)
        # 获取当前帧的关键点数据
        current_keypoints = compute_distances([frame_id], json_data)
        if len(current_keypoints)==0:
            print(f"跳过帧 {frame_id}，无法计算距离")
            continue  # 如果 distances 为 None，则跳过该帧
        # print(current_keypoints)
        current_keypoints=current_keypoints[0]
        
        # 如果是回合的第一帧，距离置为 0
        if frame_id == round_start: #这里要合并到下面
            frame_distances[frame_id] = 0
            previous_frame = frame_id  # 当前帧为第一帧，作为前一帧
            continue

        # 查找前一帧
        prev_frame = get_previous_frame(frame_id, all_frames_keypoints, (round_start, round_end),json_data)
        

        if prev_frame is None:
            # 如果找不到前一帧，表示当前帧无法计算
            print(f"无法找到前一帧，跳过帧 {frame_id}")
            continue

        # 获取前一帧的关键点数据
        prev_keypoints =  compute_distances([prev_frame], json_data)
        prev_keypoints=prev_keypoints[0]

        # 计算当前帧与前一帧的距离
        # distances = compute_distances([prev_keypoints,current_keypoints], json_data)

        template_lengths, target_lengths = callen(prev_keypoints,current_keypoints)  # 模板帧与待计算帧进行对比
        transforms = calculate_transforms(template_lengths, target_lengths,current_keypoints)
        calculated_coords = calculate_coords(tables,transforms, current_keypoints)
        # a_joints_translated,a_joints,t_joints=movea2t(calculated_coords,keypoints)
        a_joints = calculated_coords

        a_joints = a_joints[5:]

        t_joints = prev_keypoints[5:]
        W=calculate_w(template_lengths)
        # 计算对应关节点的距离并加权求和得到汇总距离D
        distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
        D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和

        frame_distances[frame_id] = D

        print(f"帧 {frame_id} 与前一帧 {prev_frame} 的骨骼点距离: {D}")

        # 更新前一帧为当前帧
        previous_frame = frame_id

# 输出距离结果到文件
output_file = '/home/haoge/ymq1/20160430-json/frame_distances.json'
with open(output_file, 'w') as f:
    json.dump(frame_distances, f, indent=4)

print(f"帧之间的距离已计算完成并保存到 {output_file}")

