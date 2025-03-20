# import numpy as np
# import json
# import matplotlib.pyplot as plt

# # 读取 JSON 文件并获取 D 值
# def load_d_values(json_file_path):
#     with open(json_file_path, 'r') as f:
#         data = json.load(f)
#     d_values = []
#     for key in data:
#         d_values.extend(data[key])  # 将每个帧的 D 值合并到一个列表中
#     return np.array(d_values)

# # 定义 Sigmoid 映射函数
# def sigmoid_similarity(D, E):
#     """
#     计算相似度 S(t, a) 基于 Sigmoid 函数
#     D: 距离或差异度量
#     E: 平滑参数，范围 (0, 1)
#     """
#     return 1 / (1 + np.exp(D / E))

# # 可视化 Sigmoid 映射结果
# def plot_sigmoid_mapping(D_values, E_values, save_path):
#     plt.figure(figsize=(10, 6))
    
#     # 用于保存每个 E 值下的标准差，来评估均匀性
#     std_devs = []

#     for E in E_values:
#         S_values = sigmoid_similarity(D_values, E)
        
#         # 计算映射结果的标准差，标准差越小说明分布越均匀
#         std_dev = np.std(S_values)
#         std_devs.append(std_dev)
        
#         # 绘制每个 E 值下的 Sigmoid 映射结果
#         plt.plot(D_values, S_values, label=f'E = {E}, Std Dev = {std_dev:.4f}')

#     # 绘制图表
#     plt.title('Sigmoid Mapping for Different E Values')
#     plt.xlabel('D (Distance or Difference)')
#     plt.ylabel('S(t, a) (Mapped Similarity)')
#     plt.legend()
#     plt.grid(True)

#     # 保存图像而不是显示
#     plt.savefig(save_path)

#     # 输出每个 E 值的标准差
#     for E, std_dev in zip(E_values, std_devs):
#         print(f"Standard Deviation for E={E}: {std_dev:.4f}")
    
#     # 选择标准差最小的 E 值
#     best_E = E_values[np.argmin(std_devs)]
#     print(f"Best E value for uniform mapping: {best_E}")
#     return best_E

# # 主函数
# def main():
#     # 读取 D 值数据
#     json_file_path = '/home/haoge/ymq1/hierarchical/frame_similarity_results.json'  # 替换为你的 JSON 文件路径
#     D_values = load_d_values(json_file_path)

#     # 设定不同的 E 值进行比较，从 1 到 50
#     E_values = np.arange(1, 51)  # 从 1 到 50，步长为 1

#     # 指定保存路径
#     save_path = 'sigmoid_mapping.png'  # 图像保存路径

#     # 可视化 Sigmoid 映射并选择最佳 E
#     best_E = plot_sigmoid_mapping(D_values, E_values, save_path)

#     # 使用最佳 E 值对 D 值进行映射
#     S_values_best_E = sigmoid_similarity(D_values, best_E)
    
#     # 映射后的结果
#     print(f"Mapped Similarities using Best E (Best E = {best_E}):")
#     print(S_values_best_E)

# if __name__ == '__main__':
#     main()



import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

# 读取 JSON 文件并获取 D 值
def load_d_values(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    d_values = []
    for key in data:
        # print(calculate_score(data[key][3]))
        # for a in data[key]:
        #     d_values.append(calculate_score(a))
        # d_values.append(calculate_score(data[key][3]))  # 将每个帧的 D 值合并到一个列表中
        d_values.append(data[key][3])
    return np.array(d_values)

def load_d_values1(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    d_values = []
    for key in data:
        # print(calculate_score(data[key][3]))
        # for a in data[key]:
        #     d_values.append(calculate_score(a))
        d_values.append(calculate_score(data[key][3]))  # 将每个帧的 D 值合并到一个列表中
        # d_values.append(data[key][3])
    return np.array(d_values)


# 得分计算公式
def calculate_score(x):
    theta_0 = -5
    theta_1 = 1
    return 100 / (1 + np.exp(-(theta_0 + theta_1 * x/10)))

# 绘制 D 值的分布图并拟合正态分布
def plot_d_distribution(D_values, save_path):
    plt.figure(figsize=(10, 6))
    
    # 绘制 D 值的直方图
    plt.hist(D_values, bins=30, edgecolor='black', color='blue', alpha=0.7, density=False)
    
    # 拟合正态分布
    mu, std = stats.norm.fit(D_values)
    
    # 绘制拟合的正态分布曲线
    xmin, xmax = plt.xlim()  # 获取 x 轴的范围
    x = np.linspace(xmin, xmax, 100)  # 生成从 xmin 到 xmax 的 100 个点
    p = stats.norm.pdf(x, mu, std)  # 根据拟合的均值和标准差计算正态分布的概率密度函数
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal fit: $\mu={mu:.2f}$, $\sigma={std:.2f}$')
    
    # 设置标题和标签
    plt.title('Distribution of D Values with Normal Fit')
    plt.xlabel('D (Distance or Difference)')
    plt.ylabel('Frequency')
    
    # 显示网格线
    plt.grid(True)
    
    # 显示图例
    plt.legend()
    
    # 保存图像
    plt.savefig(save_path)
    plt.close()  # 关闭图像以释放内存
    print(f"Distribution plot saved to {save_path}")

# 主函数
def main():
    # 读取 D 值数据
    json_file_path = '/home/haoge/ymq1/20160501-1_json/frame_similarity_results.json' # 替换为你的 JSON 文件路径
    D_values = load_d_values(json_file_path)
    print(np.var(D_values))

    D_values1 = load_d_values1(json_file_path)
    print(np.var(D_values1))

    
    # 指定保存路径
    save_path = 'd_distribution_with_normal_fit.png'  # 图像保存路径
    
    # 绘制 D 值的分布图并保存
    # plot_d_distribution(D_values, save_path)

if __name__ == '__main__':
    main()
