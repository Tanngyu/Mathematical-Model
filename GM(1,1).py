import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def grey_predict(Past, n):
    if len(Past) < 2:
        raise ValueError("Past序列长度必须至少为2")

    # 累加生成序列
    X0 = np.array(Past)
    X1 = np.cumsum(X0)

    # 构造数据矩阵B和向量Y
    B = np.zeros((len(X0) - 1, 2))
    Y = np.zeros(len(X0) - 1)
    for i in range(len(X0) - 1):
        B[i, 0] = -0.5 * (X1[i] + X1[i + 1])
        B[i, 1] = 1
        Y[i] = X0[i + 1]

    # 求解参数a和b
    B_T_B = np.dot(B.T, B)
    B_T_Y = np.dot(B.T, Y)
    coeff = np.linalg.solve(B_T_B, B_T_Y)
    a, b = coeff

    # 预测公式
    def predict(k):
        return (X0[0] - b / a) * np.exp(-a * (k - 1)) + b / a
    # 累加预测值
    X1_future = [predict(k) for k in range(len(X0) + 1, len(X0) + n + 1)]

    # 累减还原
    X0_future = [X1_future[0] - X1[-1]]  # 第一个预测值相对累加序列的最后一个值
    for i in range(1, len(X1_future)):
        X0_future.append(X1_future[i] - X1_future[i - 1])

    return X0_future

def test_up_down(predictions , q):
    k = abs(predictions - q)/q
    if k > t and predictions - q > 0 :
        return 1
    elif k > t and predictions - q < 0 :
        return 2
    else: return 0

file_path = '11个太阳能发电站.xlsx'  # 替换为你Excel文件的路径
df = pd.read_excel(file_path)

df.set_index(df.columns[0], inplace=True)  # 设置第一列为索引
sum_column = df['sum']  # 提取名为 'sum' 的列
sum_column = sum_column[:11000]
t = 0


Past =  sum_column[0:1800]      # 读取前半个小时的数据
Past = np.array(Past)

init = 10
get_pre = 1
Time_initidute = 3500
Pre_result = []


for Initidute in range(Time_initidute):
    Past_Example = sum_column[0:(49 + Initidute*(init*get_pre))].copy()            # 清空所有预测的值并将实际的数据填充进来。以上过程共进行 Time_initidute次
    for i in range(init):
        predictions = grey_predict(Past_Example[-30:],get_pre)    # 基于现有序列预测未来1秒的数据
        Pre_result = np.concatenate((Pre_result, predictions))  # 记录所有的预测序列
        Past_Example = np.concatenate((Past_Example, predictions))      # 将预测数据作为实际数据继续预测直到有 init（120、300） 秒




# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 42

sum_column += 15


plt.plot(sum_column[0:10000], linestyle='-', color='black', linewidth=3 , label = "Actual")
plt.plot(Pre_result[0:10000], linestyle='-', color='green', linewidth=3 , label = "Prediction")
plt.ylim(0, 30)
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend(prop={'size': 32})
plt.show()


Pre_result = pd.DataFrame(Pre_result, columns=['Pre'])  # 指定列名为 'Column_Name'
output_file = 'output2.xlsx'  # 替换为你想保存的文件名
Pre_result.to_excel(output_file, index=False)  # index=False 表示不保存索引列

print(f"数组已保存为 Excel 文件：{output_file}")