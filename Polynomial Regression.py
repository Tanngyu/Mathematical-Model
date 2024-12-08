import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示成方块的问题

# 读取表格数据
file_path = '回归分析训练集.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path, header=0)  # 读取数据，第一行作为列名

# 提取时间序列和样本值
time_series = pd.to_datetime(data.iloc[:, 0])  # 日期列转换为时间时间对象
time_series = (time_series - time_series.min()).dt.days.values.reshape(-1, 1)  # 转换为相对天数
values = data.iloc[:, 1:].values  # 从第二列开始作为样本值

# 获取列名
column_names = data.columns[1:]

# 剔除全为零的变量
non_zero_columns = np.any(values != 0, axis=0)
values = values[:, non_zero_columns]
data = data.iloc[:, np.append([0], np.where(non_zero_columns)[0] + 1)]

# 设置多项式的度数
degree = 3  # 你可以调整这个值

# 创建一个多项式特征转换器
poly = PolynomialFeatures(degree=degree)

# 对时间序列进行多项式特征转换
time_series_poly = poly.fit_transform(time_series)

# 使用线性回归模型进行多项式回归
model = LinearRegression()

# 预测下一个时间点的数据
next_time_point = np.array([[time_series.max() + 1]])  # 下一个时间点
next_time_point_poly = poly.transform(next_time_point)

# 存储每列的拟合结果和预测数据
results = []

# 确定子图的行列数，使图表呈矩形排列
num_cols = values.shape[1]
n_rows = int(np.ceil(np.sqrt(num_cols)))
n_cols = int(np.ceil(num_cols / n_rows))

# 创建子图
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), sharex=True)

# 确保axes是一个二维数组
if num_cols == 1:
    axes = np.array([[axes]])

for i in range(num_cols):
    ax = axes[i // n_cols, i % n_cols]  # 选择当前子图

    # 拟合模型
    model.fit(time_series_poly, values[:, i])

    # 进行预测
    predictions = model.predict(time_series_poly)
    next_prediction = model.predict(next_time_point_poly)[0]

    # 保存下一个时间点的预测值
    results.append({'Column': column_names[i], 'Next_Time_Point': time_series.max() + 1, 'Prediction': next_prediction})

    # 绘制拟合曲线
    ax.plot(time_series, predictions, color='blue')

    # 绘制原始数据
    ax.scatter(time_series, values[:, i], alpha=0.5, color='red')

    # 添加标题
    ax.set_title(column_names[i], fontsize=12)  # 使用列名作为标题

    # 设置刻度标签大小
    ax.tick_params(axis='both', labelsize=10)  # 修改刻度标签大小

    # 移除图例
    ax.legend().set_visible(False)  # 隐藏图例

# 调整布局
plt.suptitle('多项式回归拟合结果', fontsize=16)  # 添加总标题
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.4)  # 调整子图间距
plt.show()

# 将结果保存到Excel文件
results_df = pd.DataFrame(results)
results_df.to_excel('回归分析预测结果.xlsx', index=False)
