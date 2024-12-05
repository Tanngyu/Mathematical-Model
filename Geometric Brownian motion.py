import numpy as np
import pandas as pd

# 全局参数
T = 7         # 总时间
N = 7 * 12    # 时间步数
dt = T / N    # 时间步长
files = ['c.xlsx', 'm.xlsx', 'p.xlsx', 'ts.xlsx']  # 文件列表

for file in files:
    # 读取数据
    df = pd.read_excel(file)

    # 生成时间数组
    t = np.linspace(0, T, N+1)

    # 初始化一个列表来保存所有资产的模拟结果
    results = []

    # 对每个资产进行几何布朗运动模拟
    for index, row in df.iterrows():
        S0 = row['x']
        mu = row['mu']
        sigma = row['sigma']

        # 生成布朗运动
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.concatenate(([0], np.cumsum(dW)))  # 在开始处插入0，以便与时间数组对齐

        # 初始化资产价格数组
        S = np.zeros(N+1)
        S[0] = S0

        # 模拟几何布朗运动
        for i in range(1, N+1):
            S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * (W[i] - W[i-1]))

        # 将模拟结果保存到列表中
        results.append(S)

    # 创建 DataFrame 仅保留 t = 1, 2, 3, 4, 5, 6, 7 时的值
    times_to_keep = [1, 2, 3, 4, 5, 6, 7]
    indices_to_keep = [np.argmin(np.abs(t - time)) for time in times_to_keep]

    filtered_results = pd.DataFrame(
        {f'Asset_{id}': [S[i] for i in indices_to_keep] for id, S in zip(df['id'], results)},
        index=[f't={time}' for time in times_to_keep]
    )

    # 将结果保存到表格文件
    output_file = file.replace('.xlsx', '_filtered_results.xlsx')
    filtered_results.to_excel(output_file)

print("处理完成！")
