import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表样式
sns.set(style="whitegrid")

# 读取文件夹中所有的Excel文件
input_folder = '/home/haoge/ymq1/round-grade'  # 输入文件夹路径
output_folder = '/home/haoge/ymq1/round-grade-out'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有 Excel 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_excel(file_path)

        # 提取标注者名字（假设文件名为标注者名）
        annotator_name = file_name.replace('.xlsx', '')
        
        # 为每个标注者创建一个文件夹
        annotator_folder = os.path.join(output_folder, annotator_name)
        os.makedirs(annotator_folder, exist_ok=True)

        # 分别绘制每个分类数下的频次图
        for class_num in range(2, 8):  # 从 2 类到 7 类
            class_col = f'精彩程度{class_num}分类'
            print(class_col)
            if class_col in df.columns:
                # 统计该分类列的频次
                class_freq = df[class_col].value_counts().sort_index()
                print(class_freq)

                # 绘制频次图
                plt.figure(figsize=(8, 6))
                class_freq.plot(kind='bar', color='skyblue', edgecolor='black')
                plt.title(f"{class_num}class-Frequency")
                plt.xlabel(f'class (0-{class_num-1})')
                plt.ylabel('Frequency')
                plt.xticks(rotation=0)
                plt.tight_layout()

                # 保存图像
                output_path = os.path.join(annotator_folder, f"{annotator_name}_class_{class_num}.png")
                plt.savefig(output_path)
                plt.close()

        print(f"已为标注者 {annotator_name} 创建图像并保存至 {annotator_folder}")
