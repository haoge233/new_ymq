# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, confusion_matrix
# from scipy.optimize import minimize
# import seaborn as sns

# # 读取数据集
# file_path = "/home/haoge/ymq1/2016-1.xlsx"
# df = pd.read_excel(file_path)


# # 转换数据类型
# df['回合总分'] = pd.to_numeric(df['回合总分'], errors='coerce')
# # df['是否回合被分割'] = pd.to_numeric(df['是否回合被分割'], errors='coerce')
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})
# df['回合尾多余帧'] = pd.to_numeric(df['回合尾多余帧'], errors='coerce')
# df['回合时长'] = pd.to_numeric(df['回合时长'], errors='coerce')
# df['精彩指数'] = pd.to_numeric(df['精彩指数'], errors='coerce')
# # print(df['精彩指数'])
# # df['回合时长'] = pd.to_numeric(df['回合时长'], errors='coerce')

# # 检查转换后的数据类型
# print(df.dtypes)


# # 计算有效时长 (去除多余帧的影响)
# df['有效时长'] = df['回合时长'] - df['回合尾多余帧'] / 25

# # 标记是否精彩：1为精彩，0为不精彩
# df['是否精彩'] = (df['精彩指数'] >= 1).astype(int)

# # 查看数据基本情况
# print("数据集样例：")
# print(df.head())

# # 查看标签分布
# print("\n标签分布：")
# print(df['是否精彩'].value_counts())

# # 定义精彩指数计算公式
# def calculate_score(params, df):
#     """
#     计算精彩指数并预测标签
#     params: [w, alpha] -> f(是否分割), g(有效时长)
#     """
#     w = params[0]  # 是否分割的加权值
#     alpha = params[1]  # 有效时长的指数
#     # print(df['回合总分'])
    
#     # 计算精彩指数
#     numerator = df['回合总分'] + w * df['是否回合被分割']
#     denominator = df['有效时长'] ** alpha
#     score = numerator / denominator
#     # print(df['是否回合被分割'])
    
#     # 预测标签：如果精彩指数 >= 1，则标记为1（精彩）
#     predicted = (score >= 1).astype(int)
#     return predicted, score

# # 定义目标函数：最小化预测误差
# def objective(params, df):
#     predicted, _ = calculate_score(params, df)
#     return 1 - accuracy_score(df['是否精彩'], predicted)  # 准确率的反向

# # 初始化参数 [w, alpha]
# initial_params = [1.0, 1.0]  # w=1, alpha=1

# # 优化参数
# result = minimize(objective, initial_params, args=(df,), bounds=[(0, 10), (0.1, 5)])
# best_w, best_alpha = result.x

# # print(df['回合总分'])

# # 输出最优参数
# print(f"\n最优的f参数 (是否分割影响): {best_w:.2f}")
# print(f"最优的g参数 (有效时长影响指数): {best_alpha:.2f}")

# # 计算最终结果
# df['预测精彩'], df['精彩指数'] = calculate_score([best_w, best_alpha], df)

# # 计算准确率
# accuracy = accuracy_score(df['是否精彩'], df['预测精彩'])
# print(f"\n模型的准确率: {accuracy:.2%}")

# # 混淆矩阵
# cm = confusion_matrix(df['是否精彩'], df['预测精彩'])
# print("混淆矩阵:\n", cm)

# print(df['精彩指数'])
# # 可视化：精彩指数分布
# if df['精彩指数'].isnull().all() or len(df['精彩指数'].dropna()) < 2:
#     print("错误: '精彩指数' 列为空或数据量不足，无法绘制图表。")
# else:
#     # 绘图代码
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data=df, x='精彩指数', bins=50, kde=True, hue='是否精彩', palette="coolwarm", legend=True)
#     plt.axvline(x=1, color='r', linestyle='--', label="Exciting Threshold")
#     plt.title("Excitement Index Distribution and Threshold")
#     plt.xlabel("Excitement Index")
#     plt.ylabel("Count")
#     plt.legend()
#     # plt.show()
#     # plt.savefig('good.png')
#     # 保存图表为图片文件
#     output_path = "good.png"  # 文件名及路径
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 控制分辨率，bbox_inches 去除多余空白

#     plt.close()  # 关闭当前图表，释放内存
# # plt.show()

# # 可视化：实际标签与预测标签
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["good", "bad"], yticklabels=["bad", "good"])
# plt.title("real vs pre")
# plt.xlabel("pre")
# plt.ylabel("real")
# output_path1 = "labels.png"  # 文件名及路径
# plt.savefig(output_path1, dpi=300, bbox_inches='tight')  # dpi 控制分辨率，bbox_inches 去除多余空白

# plt.close()  # 关闭当前图表，释放内存
# # plt.show()


# import pandas as pd
# import numpy as np

# def custom_sigmoid(x, theta_0=5, theta_1=-6):
#     return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5

# # 读取你的数据集（替换为实际文件路径）
# file_path = "/home/haoge/ymq1/2016-1.xlsx"  # 替换为你的实际文件路径
# df = pd.read_excel(file_path)
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})
# # 计算 "有效时长"
# df['有效时长'] = df['回合时长'] - (df['回合尾多余帧'] / 25)

# # 计算 "有效得分"
# df['有效得分'] = (df['回合总分']+df['是否回合被分割']*2000 )/( custom_sigmoid(df['有效时长'])*df['有效时长'])

# # 计算 "平均有效得分"
# # average_effective_score = df['有效得分'].mean()

# # 输出平均有效得分
# print(df['有效得分'])

# # 将更新后的 DataFrame 保存为新的 Excel 文件
# output_file_path = "2016-res_1.xlsx"  # 新的文件路径
# df.to_excel(output_file_path, index=False)

# # 输出保存路径
# print(f"数据已保存到: {output_file_path}")


# import pandas as pd

# # 读取你的数据集（替换为实际文件路径）
# file_path = "/home/haoge/ymq1/2016-res_1.xlsx"  # 替换为你的实际文件路径
# df = pd.read_excel(file_path)



# # 按 "有效得分" 排序（降序：从高到低）
# df_sorted = df.sort_values(by='有效得分', ascending=False)

# # 保存排序后的结果到新的 Excel 文件
# output_file_path = "/home/haoge/ymq1/2016-res-sort_2.xlsx"  # 新的文件路径
# df_sorted.to_excel(output_file_path, index=False)

# # 输出保存路径
# print(f"排序后的数据已保存到: {output_file_path}")


#cal_time-main 

# import pandas as pd
# import numpy as np

# def custom_sigmoid(x, theta_0=5, theta_1=-6):
#     return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5

# # 读取你的数据集（替换为实际文件路径）
# file_path = "/home/haoge/ymq1/2016-1-mtest_with_scores_theta0_16.xlsx"  # 替换为你的实际文件路径
# df = pd.read_excel(file_path)
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})

# # 计算 "有效时长"
# df['有效时长'] = df['回合时长'] - (df['回合尾多余帧'] / 25)

# # 定义网格搜索的参数范围
# theta_0_values = np.linspace(3, 7, 9)  # 你可以调整这个范围和步长
# theta_1_values = np.linspace(-7, -4, 16)  # 你可以调整这个范围和步长

# # 进行网格搜索
# for theta_0 in theta_0_values:
#     for theta_1 in theta_1_values:
#         # 计算有效得分
#         # df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 1000) / (custom_sigmoid(df['有效时长'], theta_0, theta_1) * df['有效时长'])
#         df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 500) 
        
#         # 按 "有效得分" 排序（降序：从高到低）
#         df_sorted = df.sort_values(by='有效得分', ascending=False)

#         # 保存排序后的结果到新的 Excel 文件
#         output_file_path = f"/home/haoge/ymq1/grid_time_16_1/move_grid_search_result_theta0_{theta_0}_theta1_{theta_1}.xlsx"
#         df_sorted.to_excel(output_file_path, index=False)

#         # 输出保存路径
#         print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

import os
import pandas as pd

# 用于存储文件名和其对应的zero_ratio
res = []

# 定义读取excel文件并处理的函数
def process_excel_file(file_path):
    try:
        # 读取excel文件
        df = pd.read_excel(file_path)

        # 检查是否有“精彩指数”这一列
        if '精彩指数' in df.columns:
            print(f"文件: {file_path}")
            # 获取前20行“精彩指数”列的数据
            s_column = df['精彩指数'].head(20)
            
            # 去掉末尾的0
            while len(s_column) > 0 and s_column.iloc[-1] == 0:
                s_column = s_column[:-1]  # 删除最后一个0

            # 输出前20行数据
            print("\n前20行精彩指数：")
            print(s_column)
            
            # 计算前20行中为0的比例
            if len(s_column) > 0:  # 确保s_column不为空
                zero_count = (s_column == 0).sum()
                zero_ratio = zero_count / len(s_column)
            else:
                zero_ratio = 0  # 如果去掉末尾0后s_column为空，设置比例为0

            print(f"\n前20行中为0的比例: {zero_ratio:.2%}")
            
            # 保存文件路径和对应的zero_ratio
            res.append((file_path, zero_ratio))
        else:
            print(f"文件 {file_path} 中没有 '精彩指数' 列")
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")

# 遍历文件夹下的所有符合命名规则的文件
def process_files_in_directory(directory_path):
    # 获取目录下所有文件
    for filename in os.listdir(directory_path):
        # 检查文件是否符合命名规则
        if filename.endswith('.xlsx') and 'x_grid_search_result_theta' in filename:
            # 构建文件的完整路径
            file_path = os.path.join(directory_path, filename)
            # 处理每个excel文件
            process_excel_file(file_path)

# 主函数
def main():
    # 设置文件夹路径
    directory_path = '/home/haoge/ymq1/grid_x_3_sig_pow1'  # 请替换为你的文件夹路径

    # 处理文件夹中的所有符合规则的excel文件
    process_files_in_directory(directory_path)

    # 获取前10个最小的zero_ratio文件
    if res:
        # 按照zero_ratio排序并取前10个
        sorted_res = sorted(res, key=lambda x: x[1])[:30]  # 获取前10个
        print(f"前10个最小的 zero_ratio:")
        for i, (file, zero_ratio) in enumerate(sorted_res, 1):
            print(f"{i}. 文件: {file}, zero_ratio: {zero_ratio:.2%}")
    else:
        print("没有有效的文件进行处理。")

if __name__ == "__main__":
    main()





# # pose
# import pandas as pd
# import numpy as np

# def custom_sigmoid(x, theta_0=5, theta_1=-6):
#     return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5

# # 读取你的数据集（替换为实际文件路径）
# file_path = "/home/haoge/ymq1/2016-3-pose-t.xlsx"  # 替换为你的实际文件路径
# df = pd.read_excel(file_path)
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})

# # 计算 "有效时长"
# df['有效时长'] = df['回合时长'] 

# # 定义网格搜索的参数范围
# theta_0_values = np.linspace(3, 7, 9)  # 你可以调整这个范围和步长
# theta_1_values = np.linspace(-7, -4, 16)  # 你可以调整这个范围和步长
# theta_0=3.0
# theta_1=-4.4

# # 计算有效得分
# df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 500) / (custom_sigmoid(df['有效时长'], theta_0, theta_1) * df['有效时长'])
# # df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 500) 

# # 按 "有效得分" 排序（降序：从高到低）
# df_sorted = df.sort_values(by='有效得分', ascending=False)

# # 保存排序后的结果到新的 Excel 文件
# output_file_path = f"2016_3_pose_res.xlsx"
# df_sorted.to_excel(output_file_path, index=False)

# # 输出保存路径
# print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")

# # move
# import pandas as pd
# import numpy as np

# def custom_sigmoid(x, theta_0=5, theta_1=-6):
#     return 0.5 / (1 + np.exp(-(theta_0 + theta_1 * x / 20))) + 0.5

# # 读取你的数据集（替换为实际文件路径）
# file_path = "/home/haoge/ymq1/2016-3-move-t.xlsx"  # 替换为你的实际文件路径
# df = pd.read_excel(file_path)
# df['是否回合被分割'] = df['是否回合被分割'].map({'是': 1, '否': 0})

# # 计算 "有效时长"
# df['有效时长'] = df['回合时长'] 

# # 定义网格搜索的参数范围
# theta_0_values = np.linspace(3, 7, 9)  # 你可以调整这个范围和步长
# theta_1_values = np.linspace(-7, -4, 16)  # 你可以调整这个范围和步长
# theta_0=5.5
# theta_1=-5.2

# # 计算有效得分
# df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 500) / (custom_sigmoid(df['有效时长'], theta_0, theta_1) * df['有效时长'])
# # df['有效得分'] = (df['回合总分'] + df['是否回合被分割'] * 500) 

# # 按 "有效得分" 排序（降序：从高到低）
# df_sorted = df.sort_values(by='有效得分', ascending=False)

# # 保存排序后的结果到新的 Excel 文件
# output_file_path = f"2016_3_move_res.xlsx"
# df_sorted.to_excel(output_file_path, index=False)

# # 输出保存路径
# print(f"参数组合 theta_0={theta_0}, theta_1={theta_1} 的排序结果已保存到: {output_file_path}")