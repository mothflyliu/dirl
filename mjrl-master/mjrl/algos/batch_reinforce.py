"""
该代码实现了一个基本的基于策略的强化学习算法 REINFORCE ，并且具有在 KL 散度上进行线搜索的功能以提高稳定性。
"""
# 导入了所需的库和模块，包括日志相关、numpy 、时间处理、torch 及其自动求导模块，以及采样器、样本处理的实用函数和日志模块。
import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog


class BatchREINFORCE:
    # 定义了 BatchREINFORCE 类，其初始化方法接收环境 env 、策略 policy 、基线 baseline ，学习率 learn_rate
    # 随机种子 seed ，期望的 KL 散度 desired_kl 以及是否保存日志 save_logs 等参数。
    def __init__(self, env, policy, baseline,
                 learn_rate=0.01,
                 seed=123,
                 desired_kl=None,
                 save_logs=False,
                 **kwargs
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        self.desired_kl = desired_kl
        if save_logs: self.logger = DataLog()

    # 这是定义了一个名为 CPI_surrogate 的方法
    # 它属于当前类的实例方法，接收 observations （观测）、actions （动作）和 advantages （优势）作为输入参数。
    def CPI_surrogate(self, observations, actions, advantages):
        # 将输入的 advantages 数组转换为 torch 的 Variable 类型，并指定不需要计算梯度。
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        # 调用当前策略对象（self.policy）的 old_dist_info 和 new_dist_info 方法来获取旧策略和新策略的分布信息。
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        # 计算新策略和旧策略的似然比 LR 。
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        # 通过计算似然比 LR 与优势变量 adv_var 的乘积的均值来得到代理目标 surr 。
        surr = torch.mean(LR*adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        # 调用当前策略（self.policy）的 old_dist_info 和 new_dist_info 方法
        # 获取给定观测值和动作下的旧策略分布信息和新策略分布信息。
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        # 调用策略的 mean_kl 方法来计算新策略分布和旧策略分布之间的平均 KL 散度，并将结果存储在 mean_kl 变量中。
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        # 调用类中的 CPI_surrogate 方法来计算给定观测值、动作和优势下的代理目标值，并将结果存储在 cpi_sur 变量中。
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        # 使用 torch.autograd.grad 函数计算 cpi_surr 相对于策略的可训练参数
        # （self.policy.trainable_params）的梯度，并将梯度存储在 vpg_grad 中。
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        # 将梯度 vpg_grad 中的每个元素进行处理，将其转换为连续的视图并获取其 numpy 数组形式，然后将这些数组连接起来。
        vpg_grad = np.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    # ----------------------------------------------------------
    # 定义了 train_step 方法，用于执行训练的一个步骤。它接收多个参数，包括要收集的样本数量 N 、环境 env 、采样模式 sample_mode
    # 时间范围 horizon 、折扣因子 gamma 、广义优势估计的参数 gae_lambda 、使用的 CPU 数量 num_cpu 和环境的关键字参数 env_kwargs 。
    def train_step(self, N,
                   env=None,
                   p=None,
                   sample_mode='trajectories',
                   horizon=1e6,
                   gamma=0.995,
                   gae_lambda=0.97,
                   num_cpu='max',
                   env_kwargs=None,
                   ):

        # Clean up input arguments
        # 清理输入参数。如果没有提供环境，就使用当前类中的环境标识。同时检查采样模式是否正确，如果不正确就打印错误并退出。
        env = self.env.env_id if env is None else env
        if sample_mode != 'trajectories' and sample_mode != 'samples' and sample_mode != 'Discriminator' and sample_mode != 'GAIL':
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        # 记录开始时间。
        ts = timer.time()
        # 根据采样模式的不同，构建输入字典并使用相应的采样函数获取路径。
        # 轨迹
        if sample_mode == 'GAIL':
            paths = p

        if sample_mode == 'Discriminator':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_paths(**input_dict)
            return paths
        if sample_mode == 'trajectories':
            input_dict = dict(num_traj=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_paths(**input_dict)
        # 样品
        elif sample_mode == 'samples':
            input_dict = dict(num_samples=N, env=env, policy=self.policy, horizon=horizon,
                              base_seed=self.seed, num_cpu=num_cpu, env_kwargs=env_kwargs)
            paths = trajectory_sampler.sample_data_batch(**input_dict)
        # 如果需要保存日志，记录采样所花费的时间。
        if self.save_logs:
            self.logger.log_kv('time_sampling', timer.time() - ts)
        # 更新随机种子。
        self.seed = self.seed + N if self.seed is not None else self.seed
        # 计算路径的回报和优势。
        process_samples.compute_returns(paths, gamma)
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # 使用获取的路径进行训练，并获取评估统计信息，然后将样本数量添加到统计信息中。
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # 如果需要保存日志，记录样本数量。
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log_kv('num_samples', num_samples)
        # 如果需要保存日志，记录基线拟合的相关信息（包括拟合前和拟合后的误差、拟合所花费的时间）；否则直接进行基线拟合。
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log_kv('time_VF', timer.time()-ts)
            self.logger.log_kv('VF_error_before', error_before)
            self.logger.log_kv('VF_error_after', error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    # 定义了 train_from_paths 方法，用于从给定的路径数据中进行训练。
    def train_from_paths(self, paths):

        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # Policy update with linesearch
        # ------------------------------
        if self.desired_kl is not None:
            max_ctr = 100
            alpha = self.alpha
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params, set_new=True, set_old=False)
                kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
                if kl_dist <= self.desired_kl:
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', self.alpha)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass

        return base_stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else \
                        0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score


    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log_kv('stoc_pol_mean', mean_return)
        self.logger.log_kv('stoc_pol_std', std_return)
        self.logger.log_kv('stoc_pol_max', max_return)
        self.logger.log_kv('stoc_pol_min', min_return)
        try:
            success_rate = self.env.env.env.evaluate_success(paths)
            self.logger.log_kv('rollout_success', success_rate)
        except:
            pass
