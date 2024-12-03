import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 确保文件名正确，并且文件位于正确的位置
file_name = '每种种类日销量（差分后）.xlsx'

# 读取Excel文件，假设第一行是列名，第一列是蔬菜编码
try:
    df = pd.read_excel(file_name, header=0 )  # 使用header=0来指定第一行为列名
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    exit()

# 将第一列设置为索引，假设第一列是蔬菜编码
df.index = df.iloc[:, 0]

# 删除作为索引的第一列
df = df.drop(df.index[0])

# 创建一个字典来存储结果
results = {'Vegetable': [], 'p-value': []}

# 对每一列进行ADF检验
for column_name in df.columns[1:7]:
    print(f"\n正在对列 '{column_name}' 进行ADF平稳性检验...")
    data_series = df[column_name]

    # 对当前列进行ADF测试
    result = adfuller(data_series)

    # 存储结果
    results['Vegetable'].append(column_name)
    results['p-value'].append(result[1])

    # 打印ADF检验的结果
    print(f'ADF检验结果 - 列: {column_name}')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Value:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

print("所有列的ADF平稳性检验完成。")

# 将结果存储到数据框中
results_df = pd.DataFrame(results)

# 保存结果到Excel文件
results_df.to_excel('adfuller_results.xlsx', index=False)

print("ADF检验结果已保存到 'adfuller_results.xlsx'。")