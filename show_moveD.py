# # import json
# # import matplotlib.pyplot as plt


# # json_file_path = '/home/haoge/ymq1/frame_distances.json'




# # with open(json_file_path, 'r') as f:
# #         json_data = json.load(f)

# # # 1. 解析数据
# # timestamps = list(json_data.keys())  # 时间戳（字符串）
# # values = list(json_data.values())   # 对应的数值

# # # 2. 找出回合的起始点：值为 0 时即为回合开始
# # rounds = []
# # current_round = []

# # for ts, value in zip(timestamps, values):
# #     if value == -10:  # 回合开始
# #         if current_round:
# #             rounds.append(current_round)  # 保存上一个回合
# #         current_round = [(ts, value)]  # 新回合，重置
# #     else:
# #         current_round.append((ts, value))  # 当前回合继续添加数据

# # # 别忘了添加最后一个回合
# # if current_round:
# #     rounds.append(current_round)

# # # 3. 可视化每个回合
# # for i, round_data in enumerate(rounds):
# #     # 提取时间戳和数值
# #     round_timestamps = [x[0] for x in round_data]
# #     round_values = [x[1] for x in round_data]
    
# #     # 绘制图形
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(round_timestamps, round_values, marker='o')
# #     plt.title(f'Round {i+1}')
# #     plt.xlabel('Timestamp')
# #     plt.ylabel('Value')
# #     plt.grid(True)
    
# #     # 保存图像，使用回合编号命名文件
# #     # plt.savefig(f'/home/haoge/ymq1/move_round_D/round_{i+1}.png')
# #     plt.savefig(f'round_{i+1}.png')
# #     plt.close()  # 关闭当前图形，避免内存问题








# # move and pose -1
import json
import matplotlib
matplotlib.use('Agg')  # 设置为无图形界面的后端
import matplotlib.pyplot as plt
import numpy as np

# 读取D值的JSON文件
def load_d_values(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def get_frame_index(d_values, frame_number):
    # 通过帧号查找对应的索引
    if frame_number in d_values:
        index = list(d_values.keys()).index(frame_number)
        return index
    else:
        print(f"帧号 {frame_number} 不存在")
        return None

# 绘制D值变化曲线并保存为单独的图像
def plot_and_save_d_values(d_values,d_values1, num_frames=1000, save_dir="d_values_plots/"):
    startf = get_frame_index(d_values, "23657")
    endf = get_frame_index(d_values, "23827")
    print(startf)
    print(endf)
    
    # 获取前num_frames帧的数据
    frames = list(d_values.items())[startf:endf+1]
    frames1=list(d_values1.items())[startf:endf+1]
    print("frames")
    print(frames[:1])
    startf1 = get_frame_index(d_values1, "23657")
    endf1 = get_frame_index(d_values1, "23827")
    print(startf1)
    print(endf1)

    frames1=list(d_values1.items())[startf1:endf1+1]
    print("frames1")
    print(frames1)

    # 每个 D 值将会有 5 条曲线
    labels = [f"Keypoint {i+1}" for i in range(5)]

    # 初始化一个列表来存储5个D值的位置
    D_values_per_keypoint = [[] for _ in range(5)]

    # 将每一帧的5个D值按索引分类到5条曲线
    for _, d_vals in frames:
        for idx in range(5):
            D_values_per_keypoint[idx].append(d_vals[idx])

    # 为每条曲线分别绘制并保存为图像
    for idx, D_values in enumerate(D_values_per_keypoint):
        # 创建一个新的图像
        plt.figure(figsize=(10, 6))

        # 在跳过的帧处插入NaN来断开曲线
        x_values = []
        y_values = []
        last_frame = None

        for i, (frame, value) in enumerate(frames):
            frame_idx = int(frame)  # 将帧号转换为整数
            # if i > 0 and int(frames[i-1][0]) != frame_idx - 1:  # 如果帧号跳跃
            #     x_values.append(None)  # 在横坐标上插入缺失的帧
            #     y_values.append(None)  # 在纵坐标上插入缺失的D值
            x_values.append(frame_idx)  # 横坐标是帧号
            y_values.append(D_values[i])  # 对应的D值

        # 绘制曲线
        plt.plot(x_values, y_values, marker='o', label="pose_D", color='b')
        plt.title(f'D Values for Keypoint  (Frames {2900}-{3060})')
        plt.xlabel('Frame Index')
        plt.ylabel('D Value')
        plt.legend(loc='best')

        x_values1 = []
        y_values1 = []

        for i, (frame, value) in enumerate(frames1):
            frame_idx = int(frame)  # 将帧号转换为整数
            # if i > 0 and int(frames[i-1][0]) != frame_idx - 1:  # 如果帧号跳跃
            #     x_values.append(None)  # 在横坐标上插入缺失的帧
            #     y_values.append(None)  # 在纵坐标上插入缺失的D值
            x_values1.append(frame_idx)  # 横坐标是帧号
            y_values1.append(value)  # 对应的D值
        plt.plot(x_values1, y_values1, marker='o',label="move_D")
        plt.legend(loc='best')

        # 保存每一条曲线的图像
        plt.savefig(f'{save_dir}keypoint_{idx + 1}_d_values.png')
        plt.close()  # 关闭当前图像，避免图像重叠

# 主函数
def main():
    # JSON文件路径
    json_file_path = '/home/haoge/ymq1/20160501-1_json/frame_similarity_results.json'  # 替换为你的JSON文件路径
    json_file_path1 = '/home/haoge/ymq1/20160501-1_json/frame_distances_5.json'  # 替换为你的JSON文件路径
    save_dir = "d_values_move_plots_2016-3_11-16_5/"  # 保存图像的目录

    # 读取D值数据
    d_values = load_d_values(json_file_path)
    d_values1 = load_d_values(json_file_path1)

    # 创建保存图像的目录
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制D值变化曲线并保存为单独的图像
    plot_and_save_d_values(d_values,d_values1, num_frames=100, save_dir=save_dir)

if __name__ == '__main__':
    main()


# 画静态回合曲线图 和静态动态曲线图
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# from matplotlib.pyplot import MultipleLocator

# # 读取JSON文件的函数
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 查找回合的起始帧和结束帧
# def get_rounds_from_json(d_values):
#     rounds = []
#     current_round_start = None
#     previous_frame = None

#     for frame, value in d_values.items():
#         if value == 0:  # 识别回合的开始（值为0的帧）
#             if current_round_start is not None:
#                 # 当前回合的结束帧是前一个帧
#                 rounds.append((current_round_start, previous_frame))  # 上一回合结束
#             current_round_start = frame  # 当前回合的起始帧
#         previous_frame = frame  # 更新上一帧为当前帧

#     # 添加最后一个回合
#     if current_round_start is not None and previous_frame is not None:
#         rounds.append((current_round_start, previous_frame))

#     return rounds

# # 绘制D值变化曲线并保存为单独的图像
# def plot_and_save_d_values(d_values, d_values1, save_dir="d_values_plots/"):
#     rounds = get_rounds_from_json(d_values1)  # 获取回合的起始和结束帧
#     # print(rounds)
    
#     for i, (start_frame, end_frame) in enumerate(rounds):
#         print(f"处理回合 {i+1}: 从帧 {start_frame} 到 帧 {end_frame}")
        
#         # 获取该回合的数据
#         frames = list(d_values.items())[list(d_values.keys()).index(start_frame):list(d_values.keys()).index(end_frame)+1]
#         # print(frames)
#         frames1 = list(d_values1.items())[list(d_values1.keys()).index(start_frame):list(d_values1.keys()).index(end_frame)+1]
#         # print(frames1)
#         # # 每个 D 值将会有 5 条曲线
#         labels = [f"Keypoint {i+1}" for i in range(5)]

#         # 初始化一个列表来存储5个D值的位置
#         D_values_per_keypoint = [[] for _ in range(5)]

#         # 将每一帧的5个D值按索引分类到5条曲线
#         for _, d_vals in frames:
#             for idx in range(5):
#                 D_values_per_keypoint[idx].append(d_vals[idx])
            
#         # 为每条曲线分别绘制并保存为图像
#         for idx, D_values in enumerate(D_values_per_keypoint):
#             # 创建一个新的图像
#             plt.figure(figsize=(40, 6))

#             # 在跳过的帧处插入NaN来断开曲线
#             x_values = []
#             y_values = []
#             for i, (frame, value) in enumerate(frames):
#                 frame_idx = int(frame)  # 将帧号转换为整数
#                 x_values.append(frame_idx)  # 横坐标是帧号
#                 y_values.append(D_values[i])  # 对应的D值

#             # 绘制曲线
#             plt.plot(x_values, y_values, marker='o', label="pose_D", color='b')
#             plt.title(f'D Values for Keypoint {idx + 1} (Frames {start_frame}-{end_frame})')
#             plt.xlabel('Frame Index')
#             plt.ylabel('D Value')
#             plt.legend(loc='best')
#                 # 调整横坐标刻度

#             x_major_locator=MultipleLocator(25)
#             #把x轴的刻度间隔设置为1，并存在变量里
#             # y_major_locator=MultipleLocator(10)
#             #把y轴的刻度间隔设置为10，并存在变量里
#             ax=plt.gca()
#             #ax为两条坐标轴的实例
#             ax.xaxis.set_major_locator(x_major_locator)
#             #把x轴的主刻度设置为1的倍数
#             # ax.yaxis.set_major_locator(y_major_locator)

            

#             # 绘制move_D
#             # x_values1 = []
#             # y_values1 = []
#             # for i, (frame, value) in enumerate(frames1):
#             #     frame_idx = int(frame)  # 将帧号转换为整数
#             #     x_values1.append(frame_idx)  # 横坐标是帧号
#             #     y_values1.append(value)  # 对应的D值

#             # plt.plot(x_values1, y_values1, marker='o', label="move_D")
#             # plt.legend(loc='best')
#             # 根据回合的起始帧和结束帧创建保存路径
#             round_save_dir = os.path.join(save_dir, f"{start_frame}_{end_frame}")
#             if not os.path.exists(round_save_dir):
#                 os.makedirs(round_save_dir)

#             # 保存每一条曲线的图像
#             plt.savefig(f'{round_save_dir}/keypoint_{idx + 1}_d_values_round_{i + 1}.png')
#             plt.close()  # 关闭当前图像，避免图像重叠

# # 主函数
# def main():
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/frame_similarity_results.json'  # 替换为你的JSON文件路径
#     json_file_path1 = '/home/haoge/ymq1/frame_distances.json'  # 替换为你的JSON文件路径
#     save_dir = "d_values_pose_plots_40_1_2016-3/"  # 保存图像的目录

#     # 读取D值数据
#     d_values = load_d_values(json_file_path)
#     d_values1 = load_d_values(json_file_path1)

#     # 创建保存图像的目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 绘制D值变化曲线并保存为单独的图像
#     plot_and_save_d_values(d_values, d_values1, save_dir=save_dir)

# if __name__ == '__main__':
#     main()

# # 画静态图找极小值点 分割回合
# import json
# import matplotlib
# matplotlib.use('Agg')  # 设置为无图形界面的后端
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from scipy.signal import find_peaks

# from matplotlib.pyplot import MultipleLocator

# # 读取JSON文件的函数
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 查找回合的起始帧和结束帧
# def get_rounds_from_json(d_values):
#     rounds = []
#     current_round_start = None
#     previous_frame = None

#     for frame, value in d_values.items():
#         if value == 0:  # 识别回合的开始（值为0的帧）
#             if current_round_start is not None:
#                 # 当前回合的结束帧是前一个帧
#                 rounds.append((current_round_start, previous_frame))  # 上一回合结束
#             current_round_start = frame  # 当前回合的起始帧
#         previous_frame = frame  # 更新上一帧为当前帧

#     # 添加最后一个回合
#     if current_round_start is not None and previous_frame is not None:
#         rounds.append((current_round_start, previous_frame))

#     return rounds

# # 查找极小值点，并返回其索引和对应的值
# def find_local_minima(y_values, min_distance=10):
#     # 使用find_peaks找到局部极大值，`-y_values`是为了找极小值
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
#                 # 如果当前极小值与前一个极小值之间的间距小于min_distance
#                 if minima_idx[i] - filtered_minima_idx[-1] < min_distance:
#                     # 保留更小的极小值
#                     if minima_values[i] < filtered_minima_values[-1]:
#                         filtered_minima_idx[-1] = minima_idx[i]
#                         filtered_minima_values[-1] = minima_values[i]
#                 else:
#                     # 如果间距足够大，直接添加当前极小值
#                     filtered_minima_idx.append(minima_idx[i])
#                     filtered_minima_values.append(minima_values[i])

#     return filtered_minima_idx, filtered_minima_values

# # 绘制D值变化曲线并保存为单独的图像
# def plot_and_save_d_values(d_values, d_values1, save_dir="d_values_plots/"):
#     rounds = get_rounds_from_json(d_values1)  # 获取回合的起始和结束帧

#     for i, (start_frame, end_frame) in enumerate(rounds):
#         print(f"处理回合 {i+1}: 从帧 {start_frame} 到 帧 {end_frame}")
        
#         # 获取该回合的数据
#         frames = list(d_values.items())[list(d_values.keys()).index(start_frame):list(d_values.keys()).index(end_frame)+1]

#         # 初始化一个列表来存储5个D值的位置
#         D_values_per_keypoint = [[] for _ in range(5)]

#         # 将每一帧的5个D值按索引分类到5条曲线
#         for _, d_vals in frames:
#             for idx in range(5):
#                 D_values_per_keypoint[idx].append(d_vals[idx])

#         # 为每条曲线分别绘制并保存为图像
#         for idx, D_values in enumerate(D_values_per_keypoint):
#             # 创建一个新的图像
#             plt.figure(figsize=(10, 6))

#             # 在跳过的帧处插入NaN来断开曲线
#             x_values = []
#             y_values = []
#             for i, (frame, value) in enumerate(frames):
#                 frame_idx = int(frame)  # 将帧号转换为整数
#                 x_values.append(frame_idx)  # 横坐标是帧号
#                 y_values.append(D_values[i])  # 对应的D值

#             # 查找并过滤极小值点
#             minima_idx, minima_values = find_local_minima(y_values, min_distance=35)

#             # 打印极小值点的坐标
#             for idx1, value in zip(minima_idx, minima_values):
#                 print(f"Minima at index {x_values[idx1]} with value {value}")

#             # 绘制曲线
#             plt.plot(x_values, y_values, marker='o', label="pose_D", color='b')
#             plt.scatter([x_values[i] for i in minima_idx], minima_values, color='red', label="Local Minima", zorder=5)
#             plt.title(f'D Values for Keypoint {idx + 1} (Frames {start_frame}-{end_frame})')
#             plt.xlabel('Frame Index')
#             plt.ylabel('D Value')
#             plt.legend(loc='best')

#             # 调整横坐标刻度
#             # x_major_locator = MultipleLocator(25)
#             # ax = plt.gca()
#             # ax.xaxis.set_major_locator(x_major_locator)

#             # 根据回合的起始帧和结束帧创建保存路径
#             round_save_dir = os.path.join(save_dir, f"{start_frame}_{end_frame}")
#             if not os.path.exists(round_save_dir):
#                 os.makedirs(round_save_dir)

#             # 保存每一条曲线的图像
#             plt.savefig(f'{round_save_dir}/keypoint_{idx + 1}_d_values_round_{i + 1}.png')
#             plt.close()  # 关闭当前图像，避免图像重叠

# # 主函数
# def main():
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/20160510-3-json//frame_similarity_results.json'  # 替换为你的JSON文件路径
#     json_file_path1 = '/home/haoge/ymq1/20160510-3-json/frame_distances.json'  # 替换为你的JSON文件路径
#     save_dir = "d_values_pose_plots_test_1-40_2016-3_25"  # 保存图像的目录

#     # 读取D值数据
#     d_values = load_d_values(json_file_path)
#     d_values1 = load_d_values(json_file_path1)

#     # 创建保存图像的目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 绘制D值变化曲线并保存为单独的图像
#     plot_and_save_d_values(d_values, d_values1, save_dir=save_dir)

# if __name__ == '__main__':
#     main()

# 画动态图
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from scipy.signal import find_peaks

# from matplotlib.pyplot import MultipleLocator

# # 读取JSON文件的函数
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 查找回合的起始帧和结束帧
# def get_rounds_from_json(d_values):
#     rounds = []
#     current_round_start = None
#     previous_frame = None

#     for frame, value in d_values.items():
#         if value == 0:  # 识别回合的开始（值为0的帧）
#             if current_round_start is not None:
#                 # 当前回合的结束帧是前一个帧
#                 rounds.append((current_round_start, previous_frame))  # 上一回合结束
#             current_round_start = frame  # 当前回合的起始帧
#         previous_frame = frame  # 更新上一帧为当前帧

#     # 添加最后一个回合
#     if current_round_start is not None and previous_frame is not None:
#         rounds.append((current_round_start, previous_frame))

#     return rounds

# # 查找极小值点，并返回其索引和对应的值
# def find_local_minima(y_values, min_distance=10):
#     # 使用find_peaks找到局部极大值，`-y_values`是为了找极小值
#     minima_idx, _ = find_peaks(-np.array(y_values))  # 找到极小值点的索引
#     minima_values = [y_values[i] for i in minima_idx]  # 获取对应的极小值

#     # 过滤相距小于min_distance的极小值
#     filtered_minima_idx = []
#     filtered_minima_values = []
#     for i in range(len(minima_idx)):
#         if minima_values[i]<2000:
#             if not filtered_minima_idx:  # 如果是第一个点，直接添加
#                 filtered_minima_idx.append(minima_idx[i])
#                 filtered_minima_values.append(minima_values[i])
#             else:
#                 # 如果当前极小值与前一个极小值之间的间距小于min_distance
#                 if minima_idx[i] - filtered_minima_idx[-1] < min_distance:
#                     # 保留更小的极小值
#                     if minima_values[i] < filtered_minima_values[-1]:
#                         filtered_minima_idx[-1] = minima_idx[i]
#                         filtered_minima_values[-1] = minima_values[i]
#                 else:
#                     # 如果间距足够大，直接添加当前极小值
#                     filtered_minima_idx.append(minima_idx[i])
#                     filtered_minima_values.append(minima_values[i])

#     return filtered_minima_idx, filtered_minima_values

# # 绘制D值变化曲线并保存为单独的图像
# def plot_and_save_d_values(d_values, d_values1, save_dir="d_values_plots/"):
#     rounds = get_rounds_from_json(d_values1)  # 获取回合的起始和结束帧

#     for i, (start_frame, end_frame) in enumerate(rounds):
#         print(f"处理回合 {i+1}: 从帧 {start_frame} 到 帧 {end_frame}")
        
#         # 获取该回合的数据
#         frames = list(d_values.items())[list(d_values.keys()).index(start_frame):list(d_values.keys()).index(end_frame)+1]
#         frames1 = list(d_values1.items())[list(d_values1.keys()).index(start_frame):list(d_values1.keys()).index(end_frame)+1]

#         plt.figure(figsize=(40, 6))

#         # 绘制move_D
#         x_values1 = []
#         y_values1 = []
#         for i, (frame, value) in enumerate(frames1):
#             frame_idx = int(frame)  # 将帧号转换为整数
#             x_values1.append(frame_idx)  # 横坐标是帧号
#             y_values1.append(value)  # 对应的D值

#         # 查找并过滤极小值点
#         minima_idx, minima_values = find_local_minima(y_values1, min_distance=30)

#         # 打印极小值点的坐标
#         for idx1, value in zip(minima_idx, minima_values):
#             print(f"Minima at index {x_values1[idx1]} with value {value}")

#         # 绘制曲线
#         # 绘制move_D
#         x_values1 = []
#         y_values1 = []
#         for i, (frame, value) in enumerate(frames1):
#             frame_idx = int(frame)  # 将帧号转换为整数
#             x_values1.append(frame_idx)  # 横坐标是帧号
#             y_values1.append(value)  # 对应的D值

#         plt.plot(x_values1, y_values1, marker='o', label="move_D")
#         plt.scatter([x_values1[i] for i in minima_idx], minima_values, color='red', label="Local Minima", zorder=5)
#         plt.legend(loc='best')

#         # 调整横坐标刻度
#         x_major_locator = MultipleLocator(25)
#         ax = plt.gca()
#         ax.xaxis.set_major_locator(x_major_locator)

#         # 根据回合的起始帧和结束帧创建保存路径
#         round_save_dir = os.path.join(save_dir, f"{start_frame}_{end_frame}")
#         if not os.path.exists(round_save_dir):
#             os.makedirs(round_save_dir)

#         # 保存每一条曲线的图像
#         plt.savefig(f'{round_save_dir}/keypoint_{1}_d_values_round_{i + 1}.png')
#         plt.close()  # 关闭当前图像，避免图像重叠
# # 主函数
# def main():
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/frame_similarity_results.json'  # 替换为你的JSON文件路径
#     json_file_path1 = '/home/haoge/ymq1/frame_distances-5.json'  # 替换为你的JSON文件路径
#     save_dir = "d_values_move5_plots_test_2_2016-3"  # 保存图像的目录

#     # 读取D值数据
#     d_values = load_d_values(json_file_path)
#     d_values1 = load_d_values(json_file_path1)

#     # 创建保存图像的目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 绘制D值变化曲线并保存为单独的图像
#     plot_and_save_d_values(d_values, d_values1, save_dir=save_dir)

# if __name__ == '__main__':
#     main()