import logging
logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

# samplers
import mjrl.samplers.core as trajectory_sampler

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

# Import Algs
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC

class DAPG(NPG):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # demo coef
                 lam_1=0.95, # decay coef
                 **kwargs,
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0
        if save_logs: self.logger = DataLog()

    def train_from_paths(self, paths):
        """
        定义一个名为 train_from_paths 的方法，接收一个路径列表 paths 作为参数
        """

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        """
        将所有路径中的观察值连接起来
        """
        actions = np.concatenate([path["actions"] for path in paths])
        """
        将所有路径中的动作连接起来
        """
        advantages = np.concatenate([path["advantages"] for path in paths])
        """
        将所有路径中的优势连接起来
        """
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)
        """
        对优势进行标准化处理，减去均值并除以标准差（加上一个极小值防止除零）
        """

        if self.demo_paths is not None and self.lam_0 > 0.0:
            """
            如果存在演示路径且相关参数满足条件
            """
            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            """
            将演示路径中的观察值连接起来
            """
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            """
            将演示路径中的动作连接起来
            """
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            """
            计算演示路径的优势
            """
            self.iter_count += 1
            # concatenate all
            all_obs = np.concatenate([observations, demo_obs])
            """
            将观察值和演示路径的观察值连接起来
            """
            all_act = np.concatenate([actions, demo_act])
            """
            将动作和演示路径的动作连接起来
            """
            all_adv = 1e-2 * np.concatenate([advantages / (np.std(advantages) + 1e-8), demo_adv])
            """
            将优势和演示路径的优势连接起来，并进行一定的缩放
            """
        else:
            all_obs = observations
            """
            否则，直接使用原来连接好的观察值
            """
            all_act = actions
            """
            直接使用原来连接好的动作
            """
            all_adv = advantages
            """
            直接使用原来连接好的优势
            """

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        """
        计算每个路径的总回报
        """
        mean_return = np.mean(path_returns)
        """
        计算平均回报
        """
        std_return = np.std(path_returns)
        """
        计算回报的标准差
        """
        min_return = np.amin(path_returns)
        """
        计算最小回报
        """
        max_return = np.amax(path_returns)
        """
        计算最大回报
        """
        base_stats = [mean_return, std_return, min_return, max_return]
        """
        将上述统计值组成一个列表
        """
        self.running_score = mean_return if self.running_score is None else \
            0.9 * self.running_score + 0.1 * mean_return  # approx avg of last 10 iters
        """
        更新运行分数，若之前没有则直接设置为当前平均回报，否则进行一定的加权平均
        """
        if self.save_logs: self.log_rollout_statistics(paths)
        """
        如果需要保存日志，记录展开的统计信息
        """

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0
        """
        初始化用于计算不同操作的时间变量
        """

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        """
        计算优化前的替代函数值
        """

        # DAPG
        ts = timer.time()
        """
        记录开始时间
        """
        sample_coef = all_adv.shape[0] / advantages.shape[0]
        """
        计算样本系数
        """
        dapg_grad = sample_coef * self.flat_vpg(all_obs, all_act, all_adv)
        """
        计算 DAPG 的梯度
        """
        t_gLL += timer.time() - ts
        """
        计算 DAPG 操作的时间并累加到 t_gLL 中
        """

        # NPG
        ts = timer.time()
        """
        记录开始时间
        """
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        """
        构建 HVP（Hessian-vector product）
        """
        npg_grad = cg_solve(hvp, dapg_grad, x_0=dapg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        """
        使用共轭梯度法求解
        """
        t_FIM += timer.time() - ts
        """
        计算 NPG 操作的时间并累加到 t_FIM 中
        """

        # Step size computation
        # --------------------------
        n_step_size = 2.0 * self.kl_dist
        """
        计算步长
        """
        alpha = np.sqrt(np.abs(n_step_size / (np.dot(dapg_grad.T, npg_grad) + 1e-20)))
        """
        计算最终的步长 alpha
        """

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        """
        获取当前策略的参数值
        """
        new_params = curr_params + alpha * npg_grad
        """
        更新策略的参数值
        """
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        """
        设置新的参数值
        """
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        """
        计算更新后的替代函数值
        """
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        """
        计算 KL 散度
        """
        self.policy.set_param_values(new_params, set_new=True, set_old=True)
        """
        再次设置参数值
        """

        # Log information
        if self.save_logs:
            """
            如果需要保存日志
            """
            self.logger.log_kv('alpha', alpha)
            """
            记录步长 alpha
            """
            self.logger.log_kv('delta', n_step_size)
            """
            记录步长相关的值 delta
            """
            self.logger.log_kv('time_vpg', t_gLL)
            """
            记录 DAPG 操作的时间
            """
            self.logger.log_kv('time_npg', t_FIM)
            """
            记录 NPG 操作的时间
            """
            self.logger.log_kv('kl_dist', kl_dist)
            """
            记录 KL 散度
            """
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            """
            记录替代函数值的改进
            """
            self.logger.log_kv('running_score', self.running_score)
            """
            记录运行分数
            """
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass
        """
        尝试评估成功情况并记录相关信息，处理可能的异常
        """
        return base_stats

