# import json
# import matplotlib.pyplot as plt

# # 读取D值的JSON文件
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data


# def get_frame_index(d_values, frame_number):
#     # 通过帧号查找对应的索引
#     if frame_number in d_values:
#         index = list(d_values.keys()).index(frame_number)
#         return index
#     else:
#         print(f"帧号 {frame_number} 不存在")
#         return None

# # 绘制D值变化曲线并保存为单独的图像
# def plot_and_save_d_values(d_values, num_frames=1000, save_dir="d_values_plots/"):

#     startf=get_frame_index(d_values,"10903")
#     endf=get_frame_index(d_values,"15700")
#     print(startf)
#     print(endf)
#     # 获取前num_frames帧的数据
#     # print(d_values[startf])
#     # print(d_values[endf])
#     frames = list(d_values.items())[startf:endf+1]
#     print(frames)


#     # 每个 D 值将会有 5 条曲线
#     labels = [f"Keypoint {i+1}" for i in range(5)]

#     # 初始化一个列表来存储5个D值的位置
#     D_values_per_keypoint = [[] for _ in range(5)]

#     # 将每一帧的5个D值按索引分类到5条曲线
#     for _, d_vals in frames:
#         for idx in range(5):
#             D_values_per_keypoint[idx].append(d_vals[idx])

#     # 为每条曲线分别绘制并保存为图像
#     for idx, D_values in enumerate(D_values_per_keypoint):
#         # 创建一个新的图像
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(len(D_values)), D_values, marker='o', label=labels[idx], color='b')
#         plt.title(f'D Values for Keypoint {idx + 1} (First {num_frames} Frames)')
#         plt.xlabel('Frame Index')
#         plt.ylabel('D Value')
#         plt.legend(loc='best')
        
#         # 保存每一条曲线的图像
#         plt.savefig(f'{save_dir}keypoint_{idx + 1}_d_values.png')
#         plt.close()  # 关闭当前图像，避免图像重叠

# # 主函数
# def main():
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/frame_similarity_results.json'  # 替换为你的JSON文件路径
#     save_dir = "d_values_plots_2016-1/"  # 保存图像的目录

#     # 读取D值数据
#     d_values = load_d_values(json_file_path)

#     # 创建保存图像的目录
#     import os
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 绘制D值变化曲线并保存为单独的图像
#     plot_and_save_d_values(d_values, num_frames=100, save_dir=save_dir)

# if __name__ == '__main__':
#     main()

import json
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
def plot_and_save_d_values(d_values, num_frames=1000, save_dir="d_values_plots/"):
    startf = get_frame_index(d_values, "10089")
    endf = get_frame_index(d_values, "10481")
    print(startf)
    print(endf)
    
    # 获取前num_frames帧的数据
    frames = list(d_values.items())[startf:endf+1]
    print(frames)

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
        print(x_values)
        print(y_values)
        # 绘制曲线
        plt.plot(x_values, y_values, marker='o', label=labels[idx], color='b')
        plt.title(f'D Values for Keypoint {idx + 1} (Frames {startf}-{endf})')
        plt.xlabel('Frame Index')
        plt.ylabel('D Value')
        plt.legend(loc='best')

        # 保存每一条曲线的图像
        # plt.savefig(f'{save_dir}keypoint_{idx + 1}_d_values.png')
        plt.close()  # 关闭当前图像，避免图像重叠

# 主函数
def main():
    # JSON文件路径
    json_file_path = '/home/haoge/ymq1/frame_similarity_results.json'  # 替换为你的JSON文件路径
    save_dir = "d_values_plots_2016-1/"  # 保存图像的目录

    # 读取D值数据
    d_values = load_d_values(json_file_path)

    # 创建保存图像的目录
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 绘制D值变化曲线并保存为单独的图像
    plot_and_save_d_values(d_values, num_frames=100, save_dir=save_dir)

if __name__ == '__main__':
    main()



# import json
# import matplotlib.pyplot as plt




# # 读取D值的JSON文件
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 绘制D值变化曲线
# def plot_d_values_curve(d_values, num_frames=1000, save_path="d_values_curve.png"):
#     # 获取前num_frames帧的数据
#     frames = list(d_values.items())[108:491]
#     print(frames)

#     # 每个 D 值将会有 5 条曲线
#     # 设置颜色和标签
#     colors = ['r', 'g', 'b', 'c', 'm']  # 5个颜色
#     labels = [f"Keypoint {i+1}" for i in range(5)]

#     # 设置图形大小
#     plt.figure(figsize=(10, 6))
    
#     # 初始化一个列表来存储5个D值的位置
#     D_values_per_keypoint = [[] for _ in range(5)]

#     # 将每一帧的5个D值按索引分类到5条曲线
#     for _, d_vals in frames:
#         for idx in range(5):
#             D_values_per_keypoint[idx].append(d_vals[idx])

#     # 绘制每条曲线
#     for idx, D_values in enumerate(D_values_per_keypoint):
#         plt.plot(range(len(D_values)), D_values, marker='o', color=colors[idx], label=labels[idx])

#     # 设置标题和标签
#     plt.title('D Values Changes for First 1000 Frames')
#     plt.xlabel('Frame Index')
#     plt.ylabel('D Value')
    
#     # 设置图例
#     plt.legend(loc='best')
    
#     # 保存图像
#     plt.savefig(save_path)

# # 主函数
# def main():
#     W=[541,1663,690,434,82]
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/frame_similarity_results.json'  # 替换为你的JSON文件路径
    
#     # 读取D值数据
#     d_values = load_d_values(json_file_path)
    
#     # 绘制D值变化曲线
#     plot_d_values_curve(d_values, num_frames=100, save_path="d_values_curve_2.png")

# if __name__ == '__main__':
#     main()



# 加权的组合D值展示
# import json
# import matplotlib.pyplot as plt
# import os
# import numpy as np
# from itertools import combinations

# # 读取D值的JSON文件
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     return data

# # 绘制D值变化曲线并保存为单独的图像
# def plot_and_save_d_values(d_values, weights, num_frames=1000, save_dir="d_values_plots/"):
#     frames = list(d_values.items())[15:289]
#     # 获取前num_frames帧的数据
#     # frames = list(d_values.items())[:num_frames]
#     print(frames)

#     # 获取每个D值的名称
#     num_d_values = len(frames[0][1])  # 假设所有帧的D值数量一致
#     D_keys = [f"D{i+1}" for i in range(num_d_values)]  # D1, D2, D3, D4, D5
#     all_combos = []  # 存储所有组合

#     # 遍历所有组合长度，从2到num_d_values
#     for length in range(2, num_d_values+1):
#         all_combos.extend(combinations(D_keys, length))  # 获取所有长度为length的组合

#     # 提取所有帧的第二个D值
#     second_D_values = []  # 存储所有帧中的第二个D值（即D2）
#     for _, d_vals in frames:
#         second_D_values.append(d_vals[1])  # 获取每帧的第二个D值

#     # 遍历所有组合
#     for combo in all_combos:
#         # 计算当前组合的加权平均
#         weights_for_combo = [weights[D] for D in combo]  # 获取当前组合的权重
#         D_values_for_combo = [[] for _ in range(len(combo))]  # 存储每个组合的D值

#         # 提取当前组合的D值
#         for _, d_vals in frames:
#             for idx, D in enumerate(combo):
#                 D_values_for_combo[idx].append(d_vals[int(D[1])-1])  # 获取每个D值对应的数值

#         # 计算加权平均
#         weighted_avg = np.zeros(len(D_values_for_combo[0]))  # 初始化加权平均数组
#         for idx, D_values in enumerate(D_values_for_combo):
#             weighted_avg += weights_for_combo[idx] * np.array(D_values)

#         weighted_avg /= sum(weights_for_combo)  # 归一化

#         # 创建新的图像并绘制曲线
#         plt.figure(figsize=(10, 6))
        
#         # 绘制第二个D值的曲线（即D2的曲线）
#         plt.plot(second_D_values, label=f'Second D Value (D2)', color='b')  # 第二个D值曲线

#         # 绘制加权平均曲线
#         plt.plot(weighted_avg, label=f'Weighted Avg ({", ".join(combo)})', color='r', linewidth=2)  # 加权平均曲线

#         plt.title(f'Second D Value (D2) and Weighted Average Curve')
#         plt.xlabel('Frame Index')
#         plt.ylabel('D Value')
#         plt.legend(loc='best')

#         # 保存图像
#         plt.savefig(f'{save_dir}{"_".join(combo)}_weighted_avg.png')
#         plt.close()

# # 主函数
# def main():
#     # JSON文件路径
#     json_file_path = '/home/haoge/ymq1/hierarchical/frame_similarity_results.json'  # 替换为你的JSON文件路径
#     save_dir = "d_values_plots_com/"  # 保存图像的目录
#     weights = {"D1": 541, "D2": 1663, "D3": 690, "D4": 434, "D5": 82}  # 每个D值的权重

#     # 读取D值数据
#     d_values = load_d_values(json_file_path)

#     # 创建保存图像的目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 绘制D值变化曲线并保存为单独的图像
#     plot_and_save_d_values(d_values, weights, num_frames=300, save_dir=save_dir)

# if __name__ == '__main__':
#     main()
