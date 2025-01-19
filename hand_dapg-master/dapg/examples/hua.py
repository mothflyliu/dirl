import matplotlib.pyplot as plt
import pandas as pd
# 设置图片清晰度
plt.rcParams['figure.dpi'] = 100

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 加载数据
df = pd.read_csv('log_dapg.csv')

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
plt.title('成功率与累计时间的关系')

# 显示图形
plt.show()