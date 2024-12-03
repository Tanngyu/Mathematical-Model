import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

def compute_correlation_matrix(file_path, output_matrix_path):
    try:
        # 加载数据
        data = pd.read_csv(file_path, index_col=0, encoding='gbk')

        # 计算皮尔逊相关系数矩阵
        correlation_matrix = data.corr(method='pearson')

        # 保存相关系数矩阵到CSV文件
        correlation_matrix.to_csv(output_matrix_path)

        # 返回相关系数矩阵
        return correlation_matrix
    except Exception as e:
        return f"An error occurred: {e}"

def plot_heatmap(corr_matrix, output_path):
    try:
        # 设置中文字体
        rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
        rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示成方块的问题

        # 使用seaborn绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar=True,
                    linewidths=.5, xticklabels=False, yticklabels=False)  # 取消行列列表头
        plt.title('相关系数矩阵热力图')  # 中文标题

        # 保存图像到本地文件
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting: {e}")

# 调用函数，传入数据文件和输出路径
corr_matrix = compute_correlation_matrix('每种单品日销量（未差分）.csv', 'correlation_matrix.csv')
if isinstance(corr_matrix, pd.DataFrame):
    plot_heatmap(corr_matrix, 'correlation_heatmap.png')
else:
    print(corr_matrix)  # 输出错误信息
