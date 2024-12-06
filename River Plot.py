import pandas as pd
import matplotlib.pyplot as plt


# 读取Excel文件
file_path = '11个太阳能发电站.xlsx'  # 替换为你Excel文件的路径
df = pd.read_excel(file_path)

time_column = 'timestamp'
value_column = 'sum'

# 计算每100个数据点的最大值、最小值和均值
window_size = 10000

# 分割数据为每100个点的窗口
max_values = []
min_values = []
mean_values = []
time_labels = []

for i in range(0, len(df), window_size):
    window = df[i:i + window_size]  # 当前窗口的数据
    max_values.append(window[value_column].max())  # 计算最大值
    min_values.append(window[value_column].min())  # 计算最小值
    mean_values.append(window[value_column].mean())  # 计算均值
    time_labels.append(window[time_column].iloc[len(window) // 2])  # 取窗口中间的时间作为时间标签

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# # 绘制原始数据
# plt.figure(figsize=(10, 6))
# plt.plot(df[time_column], df[value_column], marker='o', linestyle='-', color='#7FBBB2', label='Original Data')

# 填充最大值和最小值之间的区域（浅蓝色）
plt.fill_between(time_labels, min_values, max_values, color='lightblue', alpha=0.5)

# 绘制最大值、最小值和均值曲线（黑色加粗，去掉点标记）
plt.plot(time_labels, max_values, linestyle='-', color='black', linewidth=3)
plt.plot(time_labels, min_values, linestyle='-', color='black', linewidth=3)
plt.plot(time_labels, mean_values, linestyle='-', color='green', linewidth=3, label='Average')

#7FBBB2 #72649E #A5796B
# 添加标题和标签
plt.xlabel('Time', fontsize=42)
plt.ylabel('Power', fontsize=42)

# 设置时间轴的刻度
plt.xticks(rotation=45)

# 添加图例
plt.legend(prop={'size': 42})
plt.grid()
# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()
