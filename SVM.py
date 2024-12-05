# 导入所需的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义绘图函数
def plot_svm_results(df, ax, title):
    # 提取特征和目标
    X = df[['id', 'mu']].values  # 特征 (id 和 mu 列)
    y = df['di'].values  # 目标标签 (di 列)

    # 训练SVM分类器
    svm_model = SVC(kernel='linear')  # 这里使用线性核
    svm_model.fit(X, y)

    # 创建一个绘图网格用于决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # 预测网格点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制分类结果的散点图
    palette = sns.color_palette('viridis', n_colors=len(df['di'].unique()))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+']

    # 绘制分类结果的散点图
    scatter = sns.scatterplot(data=df, x='id', y='mu', hue='di', palette=palette, s=100, alpha=0.8, style='di', markers=markers, legend='full', ax=ax)

    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.3, levels=len(np.unique(y)) - 1, cmap='viridis')

    # 计算每个 'di' 类别的中心点
    for i, di_value in enumerate(df['di'].unique()):
        subset = df[df['di'] == di_value]
        center_x = subset['id'].mean()
        center_y = subset['mu'].mean()
        # 绘制中心点
        ax.scatter(center_x, center_y, color=palette[i], s=250, edgecolor='red', alpha=0.7, marker='o', label=f'第{di_value}类耕地地中心点')

    # 修改图例的标题为“耕地类型”
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, [f'{label}' for label in labels], title='耕地类型', fontsize=14, title_fontsize='13')

    # 设置标题和标签的字体大小
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('作物编号', fontsize=28)
    ax.set_ylabel('种植面积/亩', fontsize=28)

    # 设置坐标轴刻度标签的字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加网格
    ax.grid(True)

# 创建保存结果的文件夹
output_dir = 'svm_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建绘图区域
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

try:
    # 读取并绘制SVM1数据
    df_svm1 = pd.read_excel('SVM1.xlsx')
    plot_svm_results(df_svm1, ax1, '前15种产品SVM分类结果')

    # 读取并绘制SVM2数据
    df_svm2 = pd.read_excel('SVM2.xlsx')
    plot_svm_results(df_svm2, ax2, '剩余产品SVM分类结果')

except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"发生错误: {e}")

# 调整布局
plt.tight_layout()

# 保存图形
output_path = os.path.join(output_dir, 'svm_classification_results.png')
plt.savefig(output_path)

# 显示图形
plt.show()
