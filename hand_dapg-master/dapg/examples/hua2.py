import matplotlib.pyplot as plt
import pandas as pd

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 200

# 设置中文字体（这里假设你已经解决了字体问题，如果没有，参考之前的建议）
plt.rcParams['font.sans-serif'] = ['SimHei']

# 定义文件路径列表
file_paths = ['log_bcrl.csv', 'log_dapg.csv']

# 初始化一个图形
plt.figure(figsize=(5, 3))

# 循环处理每个文件
for file_path in file_paths:
    # 加载数据
    df = pd.read_csv(file_path)

    # 计算 time_sampling，time_vpg，time_npg，time_VF 的和值，并将结果转换为小时
    df['time_sum'] = df[['time_sampling', 'time_vpg', 'time_npg', 'time_VF']].sum(axis=1) / 3600

    # 计算累计时间
    df['cumulative_time'] = df['time_sum'].cumsum()

    # 筛选出10小时内的数据
    df_10h = df[df['cumulative_time'] <= 10]

    # 绘制折线图，每条折线使用不同的颜色
    plt.plot(df_10h['cumulative_time'], df_10h['success_rate'])

# 设置x轴和y轴标签
plt.xlabel('累计时间 (小时)')
plt.xticks(rotation=45)
plt.ylabel('成功率')

# 设置标题
plt.title('10小时内成功率与累计时间的关系')

# 设置图例
plt.legend(['bcrl', 'npg', 'dapg'])

# 去除标题和标签的白边框（如果需要）
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 显示图形
plt.show()