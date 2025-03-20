import array
import numpy as np
from sympy import true

# 真实标签（前5名）
# true_labels = np.array([4,4,4,4,3,3,3,3,3,2])

# 模型1预测排序（前5名）
# model1_predictions = np.array([1, 3, 1, 0, 1])
model1_predictions = np.array([1, 3, 1, 2, 1,3,0,3,0,2,0,4,2,2,4,3,1,4,4,0,1,0,0,0,0,3,0,2,0,0])
true_labels = np.sort(model1_predictions)[::-1]
print(true_labels)
# 模型2预测排序（前5名）
# model2_predictions = np.array([4, 1, 0, 0, 3])
model2_predictions = np.array([4, 1, 2, 1, 3,1,0,4,3,0,3,2,4,2,2,1,0,0,0,0,3,1,0,3,2,0,0,4,0,0])
model3_predictions=np.array([3,1,1,3,3,1,4,2,4,0,2,0,4,2,1,0,0,0,0,3,2,0,1,3,0,0,4,0,2,0])
model4_predictions=np.array([4,1,1,3,3,1,2,3,1,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# 计算DCG
def calculate_dcg(scores, k):
    return np.sum([score / np.log2(i + 2) for i, score in enumerate(scores[:k])])

# 计算IDCG（理想情况下的DCG）
def calculate_idcg(true_scores, k):
    sorted_true_scores = sorted(true_scores, reverse=True)
    return calculate_dcg(sorted_true_scores, k)

# 设置k值
k = 30

# 计算DCG和IDCG
dcg_model1 = calculate_dcg(model1_predictions, k)
dcg_model2 = calculate_dcg(model2_predictions, k)
dcg_model3 = calculate_dcg(model3_predictions, k)
dcg_model4 = calculate_dcg(model4_predictions, k)
idcg = calculate_idcg(true_labels, k)

# 计算NDCG
ndcg_model1 = dcg_model1 / idcg
ndcg_model2 = dcg_model2 / idcg
ndcg_model3 = dcg_model3 / idcg
ndcg_model4 = dcg_model4 / idcg
print(f'NDCG for Model 1: {ndcg_model1}')
print(f'NDCG for Model 2: {ndcg_model2}')
print(f'NDCG for Model 3: {ndcg_model3}')
print(f'NDCG for Model 4: {ndcg_model4}')
