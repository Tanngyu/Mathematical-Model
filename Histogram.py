import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 读取数据
df = pd.read_excel('太阳偏差.xlsx')

# 设置图形的样式
sns.set(style="whitegrid")


# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# 创建一个画布和轴
plt.figure(figsize=(12, 6))
y = [r'$Sensor_{13}$',
r'$Sensor_8$',
r'$Sensor_{11}$',
r'$Sensor_7$',
r'$Sensor_5$',
r'$Sensor_{10}$',
r'$Sensor_6$',
r'$Sensor_4$',
r'$Sensor_1$',
r'$Sensor_2$',
r'$Sensor_9$',
]
# 绘制条形图
sns.barplot(x=y, y=df['MAD'], color="green", edgecolor="black")

# 添加标题和标签
plt.xlabel("Sensor", fontsize=42)  # 横轴为第一列 'Name'
plt.ylabel("Instability", fontsize=42)  # 纵轴为 'MAD'
plt.ylim(0.29,0.3025)
# 旋转横轴标签以避免重叠
plt.xticks(rotation=45, ha='right', fontsize=42)
plt.yticks(fontsize=42)
# plt.grid(False)
# 显示图形
plt.tight_layout()
plt.show()
