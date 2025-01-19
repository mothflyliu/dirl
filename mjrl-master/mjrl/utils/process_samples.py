import numpy as np

def compute_returns(paths, gamma):
    # 遍历每个路径
    for path in paths:
        # 为当前路径计算并设置返回值（returns）
        path["returns"] = discount_sum(path["rewards"], gamma)

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    # 如果 gae_lambda 未设置、小于 0 或大于 1，执行标准模式
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        # 对于每个路径
        for path in paths:
            # 用基线模型预测路径的值
            path["baseline"] = baseline.predict(path)
            # 计算优势值为返回值减去基线预测值
            path["advantages"] = path["returns"] - path["baseline"]
        # 如果需要标准化优势值
        if normalize:
            # 将所有路径的优势值连接成一个数组
            alladv = np.concatenate([path["advantages"] for path in paths])
            # 计算平均值
            mean_adv = alladv.mean()
            # 计算标准差
            std_adv = alladv.std()
            # 对每个路径的优势值进行标准化
            for path in paths:
                path["advantages"] = (path["advantages"] - mean_adv) / (std_adv + 1e-8)
    # 如果 gae_lambda 在 0 到 1 之间，执行广义优势估计（GAE）模式
    else:
        for path in paths:
            # 预测路径的基线值并进行处理
            b = path["baseline"] = baseline.predict(path)
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            # 计算时间差分（TD）误差
            td_deltas = path["rewards"] + gamma * b1[1:] - b1[:-1]
            # 计算优势值
            path["advantages"] = discount_sum(td_deltas, gamma * gae_lambda)
        # 如果需要标准化优势值
        if normalize:
            # 将所有路径的优势值连接成一个数组
            alladv = np.concatenate([path["advantages"] for path in paths])
            # 计算平均值
            mean_adv = alladv.mean()
            # 计算标准差
            std_adv = alladv.std()
            # 对每个路径的优势值进行标准化
            for path in paths:
                path["advantages"] = (path["advantages"] - mean_adv) / (std_adv + 1e-8)

def discount_sum(x, gamma, terminal=0.0):
    # 初始化一个空列表 y 用于存储折扣后的累加结果
    y = []
    # 初始化累加和为终端值 terminal
    run_sum = terminal
    # 从数组 x 的最后一个元素开始，倒序遍历
    for t in range(len(x) - 1, -1, -1):
        # 计算当前元素加上折扣后的之前累加和
        run_sum = x[t] + gamma * run_sum
        # 将当前累加和添加到 y 列表中
        y.append(run_sum)

    # 将 y 列表反转并转换为 numpy 数组返回
    return np.array(y[::-1])