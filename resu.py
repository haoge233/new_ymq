import pandas as pd

# 设定文件路径
file1 = '/home/haoge/ymq1/grid_time/grid_search_result_theta0_3.0_theta1_-4.4.xlsx'
file2 = '/home/haoge/ymq1/grid_time/move_grid_search_result_theta0_5.5_theta1_-5.2.xlsx'
output_file = '2016-1-res-4.xlsx'

# 读取两个 Excel 文件
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# 确保两个文件都有 '回合' 和 '有效得分' 列
if '回合' in df1.columns and '有效得分' in df1.columns and '回合' in df2.columns and '有效得分' in df2.columns:
    # 以 file1 为标准，合并 file2 的 '有效得分'（基于 '回合' 列）
    merged_df = pd.merge(df1, df2[['回合', '有效得分']], on='回合', how='left', suffixes=('_file1', '_file2'))

    # 将 file2 的 '有效得分' 列相加到 file1 的 '有效得分' 列
    merged_df['有效得分'] = merged_df['有效得分_file1'] + 0.7*merged_df['有效得分_file2'].fillna(0)

    # 删除临时列
    merged_df.drop(columns=['有效得分_file1', '有效得分_file2'], inplace=True)

    # 按照 '有效得分' 列进行排序
    merged_df = merged_df.sort_values(by='有效得分', ascending=False)

    # 保存到新的 Excel 文件
    merged_df.to_excel(output_file, index=False)
    print(f"结果已保存到: {output_file}")
else:
    print("两个文件都必须包含 '回合' 和 '有效得分' 列。")