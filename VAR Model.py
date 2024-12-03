import pandas as pd
from statsmodels.tsa.api import VAR
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 读取数据
file_path = 'VAR模型训练集.xlsx'
df = pd.read_excel(file_path, header=0)

# 提取变量数据，排除第一列的销售日期和第一行的品类编码
data_vars = df.iloc[1:, 1:].reset_index(drop=True)  # 去除第一行，并重置索引

# 将数据类型转换为浮点数
data_vars = data_vars.astype(float)

# 设置滞后阶数
lag_order = 6

# 创建 VAR 模型并拟合
model = VAR(data_vars)
fitted_model = model.fit(lag_order)

# 获取模型参数
params = fitted_model.params

# 将模型参数保存为 Excel 文件
params_file_path = 'VAR模型参数.xlsx'
params.to_excel(params_file_path)

rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示成方块的问题

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(params, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# 添加标题和标签
plt.title('VAR模型参数热力图', fontsize=16)
plt.xlabel('估计量', fontsize=12)
plt.ylabel('参数', fontsize=12)

# 保存热力图为文件
heatmap_file_path = 'VAR模型参数热力图.png'
plt.savefig(heatmap_file_path, dpi=300)

# 显示热力图
plt.show()
