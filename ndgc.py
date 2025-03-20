import pandas as pd
import numpy as np

# 读取 XLSX 文件
df = pd.read_excel("/home/haoge/ymq1/2016-1-xlsx/2016-1-res.xlsx")  # 替换为你的文件路径

# 查看数据
print(df.head())

# 1. 计算 DCG
def calculate_dcg(scores, k):
    dcg = 0
    for i in range(k):
        dcg += scores[i] / np.log2(i + 2)  # log2(i+2) 因为排名从1开始
    return dcg

# 2. 计算 IDCG（理想情况下的 DCG）
def calculate_idcg(ground_truth, k):
    # 按照真实精彩指数从高到低排序
    ideal_scores = sorted(ground_truth, reverse=True)
    return calculate_dcg(ideal_scores, k)

# 3. 计算 NDCG
def calculate_ndcg(df, k):
    # 先根据回合总分（或其他模型得分）排序，得到模型的排序
    df_sorted = df.sort_values(by='回合总分', ascending=False)
    
    df_top_k = df.head(k)
    
    # 获取这些回合的真实评分，即“精彩指数”
    true_scores = df_top_k['精彩指数'].values
    
    print(true_scores)
    # 计算 DCG 和 IDCG
    dcg = calculate_dcg(true_scores, k)
    print(dcg)
    idcg = calculate_idcg(true_scores, k)
    print(idcg)
    
    # 计算 NDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

# 计算 NDCG@5（例如前5个回合）
ndcg_5 = calculate_ndcg(df, 33)
print(f"NDCG@5: {ndcg_5:.4f}")