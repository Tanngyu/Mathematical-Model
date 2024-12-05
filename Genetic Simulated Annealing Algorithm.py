import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 读取数据
data = pd.read_excel('data.xlsx')

# 获取不同的土地类型和作物ID
land_types = list(data['di'].unique())
crops = data['id'].unique()
land_crops = [
    list(range(0, 46 + 1)),  # [1, 2, ..., 45]
    [46],
    list(range(47, 100 + 1)),  # [47, 48, ..., 100]
    list(range(101, 103 + 1)),  # [101, 102, 103]
    list(range(47, 100 + 1)),  # [47, 48, ..., 100]
    list(range(104, 107 )),  # [104, 105, 106, 107]
    list(range(47, 100 + 1)),  # [47, 48, ..., 100]
    list(range(47, 100 ))   # [47, 48, ..., 100]
]


num_land_types = len(land_types)  # 土地类型数量
num_crops = len(crops)  # 作物ID数量

print(range(num_crops))
print(land_crops[0])

# 初始化参数
T0 = 100000 # 初始温度
alpha = 0.95  # 温度衰减因子
K = 1000  # 迭代次数
current_T = T0  # 当前温度

Temp_list = [T0]

# 土地面积
data2 = pd.read_excel('data2.xlsx')
columns = data2.columns
A = [data2[col].dropna().tolist() for col in columns]

# 生成初始0解
def generate_initial_solution():
    solution = {}
    for t in range(8):  # 包括t=0，假设有7个时间段，从0到6
        solution[t] = {}
        for i in range(num_land_types):
            solution[t][i] = {}
            for j in land_crops[i]:
                solution[t][i][j] = 0
    return solution

# 计算目标函数值
def calculate_objective(solution, scenario):
    profit = 0

    for t in range(7):
        for i in range(num_land_types):
            for j in land_crops[i]:
                pi = data.iloc[j]['pi']  # 作物j的单位利润
                p = data.iloc[j]['p']  # 作物j的市场价格
                c = data.iloc[j]['c']  # 作物j的成本
                total_area = solution[t][i][j]  # 时间t、土地i上作物j的总种植面积
                a = min(total_area, data.loc[j, 'ts'])  # 作物j的实际种植面积（受限于土地面积）

                if (total_area > a):
                    b = total_area - a  # 剩余面积
                else:
                    b = 0

                if scenario == 1:
                    profit += a * pi - b * c  # 场景1：利润计算方式



                elif scenario == 2:
                    profit += (a * pi) - (b * c) + (b * (0.5 * p)) # 场景2：利润计算方式



    return profit



# 约束条件检查
def check_constraints(solution):

    # 连续重茬种植检查
    for i in range(num_land_types):
        for j in land_crops[i]:
            for t in range(1, 8):  # 从第1个时间段开始检查
                if solution[t][i][j] * solution[t - 1][i][j] != 0:
                   solution[t][i][j] = 0

    # 豆类作物种植检查
    valid_ids = list(range(1, 16)) + list(range(47, 56))

    for i in [1,3]:
        # 遍历所有时间段
        for t in range(4, 8):  # 从时间段4到7检查
            # 检查是否有连续三个时间段没有种植豆类植物
            consecutive_missing = 0

            for k in range(t - 2, t + 1):
                if not any(solution[k][i].get(j, 0) > 0 for j in valid_ids):
                    consecutive_missing += 1
                else:
                    consecutive_missing = 0

                if consecutive_missing == 3:
                        j = random.choice(valid_ids)  # 选择一种豆类种植
                        solution[k][i][j] = random.choice(A[i])  # 随机生成新的种植面积作为迭代对象


    # 豆类作物种植检查
    valid_ids = list(range(1, 16)) + list(range(47, 56))

    for i in range(num_land_types):
        # 遍历所有时间段
        for t in range(4, 8):  # 从时间段4到7检查
            # 检查是否有连续三个时间段没有种植豆类植物
            consecutive_missing = 0

            for k in range(t - 2, t + 1):
                if not any(solution[k][i].get(j, 0) > 0 for j in valid_ids):
                    consecutive_missing += 1
                else:
                    consecutive_missing = 0

                if consecutive_missing == 3:

                   solution[k][i][j] = random.choice(valid_ids)     #若三年每种豆类，那么强制在最后一年种植豆类


    return True


    Ar = [1092, 109, 109, 109, 9.6, 9.6, 2.4, 2.4]

    for t in range(8):
        total_areas = []

        for i in range(num_land_types):
            # 计算每种土地类型的总面积
            total_area = sum(solution[t][i].values())
            total_areas.append(total_area)


        # 比较每个总面积值是否在允许范围内
        for i in range(num_land_types):
            if not (0.3 <= total_areas[i] <= Ar[i]):
                total_areas[i] = Ar[i]
    print(2)
    return True

# 产生新解
def generate_new_solution(current_solution):
    new_solution = current_solution.copy()

    # 批量应用单点交叉
    if random.random() < 0.95:  # 以95%的概率应用交叉
        parent1 = current_solution
        parent2 = batch_reverse_mutation(current_solution)
        new_solution = batch_crossover(parent1, parent2)  # 一次交叉多个数据

    # 批量应用翻转变异
    if random.random() < 0.05:  # 以5%的概率应用变异
        new_solution = batch_reverse_mutation(new_solution)  # 一次变异多个数据

    return new_solution


# 批量单点交叉
def batch_crossover(parent1, parent2, crossover_count=15):
    child = parent1.copy()

    for _ in range(crossover_count):
        t = random.randint(0, 6)  # 随机选择时间段
        i = random.randint(0, num_land_types - 1)  # 随机选择土地类型

        # 选择一个随机交叉点，强制进行交叉

        if min(land_crops[i]) < max(land_crops[i])-1:

            crossover_point = random.randint(min(land_crops[i]),max(land_crops[i])-1)

        else:
            continue

        # 从交叉点之后的基因全部从 parent2 继承
        for j in range(crossover_point, max(land_crops[i])-1):
            child[t][i][j] = parent2[t][i][j]

    return child


# 批量翻转变异
def batch_reverse_mutation(solution, mutation_count=15):
    for _ in range(mutation_count):
        t = random.randint(0, 6)  # 随机选择时间段
        i = random.randint(0, num_land_types - 1)  # 随机选择土地类型
        j = random.choice(land_crops[i])  # 随机选择作物
        temp = solution[t][i][j]
        solution[t][i][j] = max(A[i]) - temp  # 将面积翻转到对称值
    return solution

# 接受新解
def accept_new_solution(current_solution, new_solution, current_T,scenario):
    current_profit = calculate_objective(current_solution,scenario)  # 当前解的目标函数值
    new_profit = calculate_objective(new_solution,scenario)  # 新解的目标函数值

    if new_profit > current_profit:
        return new_solution  # 如果新解更优，接受新解
    else:
        delta = current_profit - new_profit  # 计算目标函数值的差异
        # 计算接受概率
        if np.isnan(delta) or np.isinf(delta):  # 检查delta的有效性
            delta = 0  # 设置为合理的默认值

        if current_T <= 1e-10:  # 防止温度为零或过低
            current_T = 1e-10  # 设为一个很小的正值


            print("温度很低了")


        probability = np.exp(-delta / current_T)  # 根据温度计算接受概率

        if current_T <= 1e-10:

                return current_solution  # 否则保持当前解,不再退火

        else:

            if random.random() < probability:
                return new_solution  # 以一定概率接受较差的新解
            else:
                return current_solution  # 否则保持当前解

# 多普勒降温曲线
def update_temperature(T, k):
    if k == K:
        return 0  # 达到最大迭代次数，温度降为0
    factor1 = np.cos(np.pi / (2 * (1 - k / K)))  # 计算温度衰减因子1
    factor2 = np.cos(np.pi / (2 * T0 * (1 - k / K)))  # 计算温度衰减因子2
    return T0 * alpha * (factor1 + factor2)  # 更新温度

# 模拟退火算法
def simulated_annealing(scenario):
    current_T = T0  # 初始化
    current_solution = generate_initial_solution()  # 生成初始解
    best_solution = current_solution
    best_profit = calculate_objective(current_solution, scenario)  # 计算初始解的目标函数值

    # 记录每次迭代的最佳目标函数值
    best_profits = [best_profit]

    for k in range(K):  # 迭代K次
        new_solution = generate_new_solution(current_solution)  # 生成新解
        if check_constraints(new_solution):  # 检查新解是否满足约束条件
            current_solution = accept_new_solution(current_solution, new_solution, current_T,scenario)  # 接受新解
            current_profit = calculate_objective(current_solution, scenario)  # 计算当前解的目标函数值
            if current_profit > best_profit:  # 更新最佳解
                best_solution = current_solution
                best_profit = current_profit

        print(best_profit)
        best_profits.append(best_profit)  # 记录当前最优解的目标函数值

        current_T = update_temperature(current_T, k)  # 更新温度

    return best_solution, best_profit, best_profits
# 执行模拟退火算法
best_solution1, best_profit1, profits1 = simulated_annealing(1)
best_solution2, best_profit2, profits2 = simulated_annealing(2)
print("Best profit for scenario 1:", best_profit1)  # 输出场景1的最佳利润
print("Best profit for scenario 2:", best_profit2)  # 输出场景2的最佳利润


# 将best_solution转换为DataFrame
def solution_to_dataframe(solution):
    data = []
    for t in solution:
        for i in solution[t]:
            for j in solution[t][i]:
                data.append({'Time': t, 'LandType': i, 'CropID': j, 'Area': solution[t][i][j]})
    df = pd.DataFrame(data)
    return df


# 将best_solution转换为DataFrame
best_solution_df1 = solution_to_dataframe(best_solution1)
best_solution_df2 = solution_to_dataframe(best_solution2)
# 将pivot_df保存到Excel文件，每个Time为一个工作表

with pd.ExcelWriter('best_solution1.xlsx') as writer:
    for time in best_solution_df1['Time'].unique():
        time_df = best_solution_df1[best_solution_df1['Time'] == time]
        pivot_df = time_df.pivot_table(index='LandType', columns='CropID', values='Area', aggfunc='sum', fill_value=0)
        pivot_df.to_excel(writer, sheet_name=f'Time_{time}')

with pd.ExcelWriter('best_solution2.xlsx') as writer:
    for time in best_solution_df2['Time'].unique():
        time_df = best_solution_df2[best_solution_df2['Time'] == time]
        pivot_df = time_df.pivot_table(index='LandType', columns='CropID', values='Area', aggfunc='sum', fill_value=0)
        pivot_df.to_excel(writer, sheet_name=f'Time_{time}')

print("Best solution has been saved to 'best_solution.xlsx' with each Time as a separate sheet.")

# 使用数据索引作为 x 轴
iterations = np.arange(len(profits1))

# 平滑曲线
spline1 = UnivariateSpline(iterations, profits1, s=0)
spline2 = UnivariateSpline(iterations, profits2, s=0)

# 生成更多的点用于平滑曲线
fine_iterations = np.linspace(iterations.min(), iterations.max(), 500)
smooth_profit1 = spline1(fine_iterations)
smooth_profit2 = spline2(fine_iterations)

# 找到并标记平滑曲线上的最高点
max_smooth_profit1_index = np.argmax(smooth_profit1)
max_smooth_profit1 = smooth_profit1[max_smooth_profit1_index]
max_iteration1 = fine_iterations[max_smooth_profit1_index]

max_smooth_profit2_index = np.argmax(smooth_profit2)
max_smooth_profit2 = smooth_profit2[max_smooth_profit2_index]
max_iteration2 = fine_iterations[max_smooth_profit2_index]

# 绘制平滑曲线
plt.figure(figsize=(10, 6))  # 调整图形大小
plt.plot(fine_iterations, smooth_profit1, color='b', linestyle='-', linewidth=2, label='情况一')
plt.plot(fine_iterations, smooth_profit2, color='r', linestyle='-', linewidth=2, label='情况二')

# 标记平滑曲线上的最高点
plt.scatter(max_iteration1, max_smooth_profit1, color='gold', s=150, edgecolor='k', marker='*', zorder=5)
plt.scatter(max_iteration2, max_smooth_profit2, color='gold', s=150, edgecolor='k', marker='*', zorder=5)

# 添加文本标签
plt.text(max_iteration1, max_smooth_profit1, f'({max_iteration1:.2f}, {max_smooth_profit1:.2f})', color='b', fontsize=12, ha='left')
plt.text(max_iteration2, max_smooth_profit2, f'({max_iteration2:.2f}, {max_smooth_profit2:.2f})', color='r', fontsize=12, ha='left')

# 添加标题和标签
plt.title('优化算法收敛趋势', fontsize=16)
plt.xlabel('迭代次数', fontsize=14)
plt.ylabel('目标函数值', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)



# 显示图形
plt.tight_layout()
plt.show()