import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 数据准备
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'mu': [147, 46, 60, 96, 25, 222, 135, 185, 50, 25, 15, 13, 18, 35, 20],
    'di': [1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 3, 3, 3, 2, 2]
}

# 构造特征矩阵和标签向量
X = np.array(list(zip(data['mu'], data['di'])))
y = np.array(data['di'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建SVM模型（使用线性核函数）
model = svm.SVC(kernel='linear')

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算误差率
N_test = len(y_test)
e_test = np.sum(y_test != y_pred) / N_test

# 输出结果
print(f'测试样本容量 N\' = {N_test}')
print(f'误差率 e_test = {e_test}')

# 假设检验
alpha = 0.05
if e_test <= alpha:
    print(f'接受原假设 H_0: 存在超平面 $w^T x + b = 0$ 完全正确分类数据，e_test <= {alpha}')
else:
    print(f'拒绝原假设 H_0: 不存在超平面 $w^T x + b = 0$ 完全正确分类数据，e_test >= {alpha}')
