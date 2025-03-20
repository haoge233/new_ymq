

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

# ----------------------------------------------------------------


def compute_distances(frame_ids,json_data):
        """
        计算选定帧之间的特征距离
        """
        # 提取选定帧的关键点
        keypoints_list = []
        for frame_id in frame_ids:
            if frame_id in json_data:
                bottom_keypoints = np.array(json_data[frame_id]['bottom'])
                # print(bottom_keypoints)
                if bottom_keypoints.size > 1:
                    bottom_keypoints[4]=(bottom_keypoints[11]+bottom_keypoints[12])*1/2
                    print(bottom_keypoints)
                    keypoints_list.append(bottom_keypoints)
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

    return a_joints_translated,a_joints,t_joints






def calculate_w(template_lengths):
    W = []
    template_sumlen = sum(template_lengths)
    for i in range(len(connections)):
        # 计算比例西塔（θ）
        w= template_lengths[i] / template_sumlen
        
        # 存储变换结果
        W.append(w)
    
    return W



def get_similarity(json_file_path,frame_ids,output_file,min_similarity_file):
    
    # 读取 JSON 数据
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # 选定的模板帧ID列表，可以根据需要添加更多模板帧
    # frame_ids =['7375', '9135', '10156', '11976', '9078']   #hierarchical
    # frame_ids =['10298', '9590', '16409', '19089', '16524']   #gmm
    # frame_ids =['11976', '11852', '9137', '12232', '16554']   #kmeans
    # frame_ids =['7233', '13217', '16381', '7484', '10255', '9028', '10165', '9138', '9204', '9589', '15411', '11201', '11869', '12334', '12439', '15396', '19152'] # dbscan
    # frame_ids =['12260', '8005', '11831', '10272', '9588']

    # frame_ids = ['9136', '10157','10201','10223','11976']  # 可以加入更多帧ID，如：['9136', '11976', '15000', '20000']


    # frame_ids =['15470', '25154', '10379', '21929', '12192'] #hierarchical  2016

    # 计算模板帧的关键点数据    
    template_keypoints = compute_distances(frame_ids, json_data)


    # 获取所有帧的关键点数据
    all_frames_keypoints = list(json_data.keys())  # 获取所有帧ID



    # 存储每一帧及其相似度
    frame_similarity = {}
    min_similarity_idx = []  # 存储每帧最大相似度的模板索引


    # 对每一帧计算与所有模板帧的相似度
    for frame_id in all_frames_keypoints:
        # 获取当前帧的关键点数据
        distances = compute_distances([frame_id], json_data)
        if distances:
            similarity_list = []  # 用于存储与各模板帧的相似度
            print(f"计算帧 {frame_id} 与所有模板帧的相似度:")
            # 对每个模板帧计算与当前帧的相似度
            for idx, keypoints in enumerate(template_keypoints):
                # 在此处，template_keypoints[idx][0] 是模板帧，distances[0] 是待计算帧
                template_lengths, target_lengths = callen(keypoints, distances[0])  # 模板帧与待计算帧进行对比
                transforms = calculate_transforms(template_lengths, target_lengths,distances[0])
                calculated_coords = calculate_coords(tables,transforms, distances[0])
                a_joints_translated,a_joints,t_joints=movea2t(calculated_coords,keypoints)
                a_joints = a_joints_translated

                a_joints = a_joints[5:]

                t_joints = t_joints[5:]
                W=calculate_w(template_lengths)
                # 计算对应关节点的距离并加权求和得到汇总距离D
                distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
                D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和
                similarity_list.append(D)
                print(f"  与模板帧 {frame_ids[idx]} 相似度 D = {D}")
            if similarity_list is None:
                continue
            # 保存当前帧的相似度列表
            frame_similarity[frame_id] = similarity_list
            min_idx = np.argmin(similarity_list)
            min_similarity_idx.append((frame_id, min_idx))  # 记录最大相似度的索引
            print(f"  帧 {frame_id} 与所有模板的相似度: {similarity_list}")
            print(f"  最大相似度的模板索引: {min_idx}\n")


    # 自定义序列化函数，将 numpy 数据类型转换为标准 Python 数据类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # 将 numpy 数组转换为列表
        else:
            raise TypeError(f"Type {type(obj)} not serializable")

    # 保存相似度结果到文件
    # output_file = 'frame_similarity_results.json'
    with open(output_file, 'w') as f:
        json.dump(frame_similarity, f, indent=4, default=convert_to_serializable)

    # 保存最大相似度索引到文件
    # min_similarity_file = 'min_similarity_indices.json'
    with open(min_similarity_file, 'w') as f:
        json.dump(min_similarity_idx, f, indent=4, default=convert_to_serializable)



json_file_path = '/home/haoge/solopose/SoloShuttlePose/res 2016-3/players/player_kp/20160501-3.json'
# 读取 JSON 数据
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# 选定的模板帧ID列表，可以根据需要添加更多模板帧
# frame_ids =['7375', '9135', '10156', '11976', '9078']   #hierarchical
# frame_ids =['10298', '9590', '16409', '19089', '16524']   #gmm
# frame_ids =['11976', '11852', '9137', '12232', '16554']   #kmeans
# frame_ids =['7233', '13217', '16381', '7484', '10255', '9028', '10165', '9138', '9204', '9589', '15411', '11201', '11869', '12334', '12439', '15396', '19152'] # dbscan
# frame_ids =['12260', '8005', '11831', '10272', '9588']

# frame_ids = ['9136', '10157','10201','10223','11976']  # 可以加入更多帧ID，如：['9136', '11976', '15000', '20000']


# frame_ids =['15470', '25154', '10379', '21929', '12192'] #hierarchical  2016

# frame_ids =['16468', '10379', '29784', '12192', '25277'] #hierarchical  2016-1
# frame_ids =['15800', '45456', '9743', '28995', '37913'] #hierarchical  2016-1-key

frame_ids =['5663', '12161', '33796', '41027', '37386'] #2016-3
# frame_ids =['38966', '116634', '115691', '102919', '13170'] #20160430

# 计算模板帧的关键点数据    
template_keypoints = compute_distances(frame_ids, json_data)


# 获取所有帧的关键点数据
all_frames_keypoints = list(json_data.keys())  # 获取所有帧ID



# 存储每一帧及其相似度
frame_similarity = {}
min_similarity_idx = []  # 存储每帧最大相似度的模板索引


# 对每一帧计算与所有模板帧的相似度
for frame_id in all_frames_keypoints:
    # 获取当前帧的关键点数据
    distances = compute_distances([frame_id], json_data)
    if distances:
        similarity_list = []  # 用于存储与各模板帧的相似度
        print(f"计算帧 {frame_id} 与所有模板帧的相似度:")
        # 对每个模板帧计算与当前帧的相似度
        for idx, keypoints in enumerate(template_keypoints):
            # 在此处，template_keypoints[idx][0] 是模板帧，distances[0] 是待计算帧
            template_lengths, target_lengths = callen(keypoints, distances[0])  # 模板帧与待计算帧进行对比
            transforms = calculate_transforms(template_lengths, target_lengths,distances[0])
            calculated_coords = calculate_coords(tables,transforms, distances[0])
            a_joints_translated,a_joints,t_joints=movea2t(calculated_coords,keypoints)
            a_joints = a_joints_translated

            a_joints = a_joints[5:]

            t_joints = t_joints[5:]
            W=calculate_w(template_lengths)
            # 计算对应关节点的距离并加权求和得到汇总距离D
            distances_t_a = np.linalg.norm(t_joints - a_joints, axis=1)  # 计算每对关节点的欧几里得距离
            D = np.dot(W, distances_t_a)  # 使用权重对距离进行加权求和
            similarity_list.append(D)
            print(f"  与模板帧 {frame_ids[idx]} 相似度 D = {D}")
        if similarity_list is None:
            continue
        # 保存当前帧的相似度列表
        frame_similarity[frame_id] = similarity_list
        min_idx = np.argmin(similarity_list)
        min_similarity_idx.append((frame_id, min_idx))  # 记录最大相似度的索引
        print(f"  帧 {frame_id} 与所有模板的相似度: {similarity_list}")
        print(f"  最大相似度的模板索引: {min_idx}\n")


# 自定义序列化函数，将 numpy 数据类型转换为标准 Python 数据类型
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 numpy 数组转换为列表
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

# 保存相似度结果到文件
output_file = '/home/haoge/ymq1/20160510-3-json/frame_similarity_results.json'
with open(output_file, 'w') as f:
    json.dump(frame_similarity, f, indent=4, default=convert_to_serializable)

# 保存最大相似度索引到文件
min_similarity_file = '/home/haoge/ymq1/20160510-3-json/min_similarity_indices.json'
with open(min_similarity_file, 'w') as f:
    json.dump(min_similarity_idx, f, indent=4, default=convert_to_serializable)