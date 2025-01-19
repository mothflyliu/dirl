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
from mjrl.algos.batch_reinforce import BatchREINFORCE


class DNPG(BatchREINFORCE):
    def __init__(self, env, policy, baseline, demo_paths,
                 const_learn_rate=None,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 normalized_step_size=0.01,
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 input_normalization=None,
                 lam_0=1.0,
                 lam_1=0.95,
                 **kwargs
                 ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """
        # 初始化环境、策略和基线
        self.env = env
        self.policy = policy
        self.baseline = baseline
        # 初始化学习率
        self.alpha = const_learn_rate
        # 根据 kl_dist 的有无确定步长
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        # 初始化随机种子
        self.seed = seed
        # 初始化是否保存日志
        self.save_logs = save_logs
        # 初始化 FIM 求解参数
        self.FIM_invert_args = FIM_invert_args
        # 初始化用于 Fisher 度量的样本分数
        self.hvp_subsample = hvp_sample_frac
        # 初始化运行得分
        self.running_score = None
        # 如果要保存日志，创建 DataLog 对象
        if save_logs:
            self.logger = DataLog()
        # 初始化输入归一化参数
        self.input_normalization = input_normalization
        # 对输入归一化参数进行有效性检查
        if self.input_normalization is not None:
            if self.input_normalization > 1 or self.input_normalization <= 0:
                self.input_normalization = None

        # 初始化处理专家数据的相关属性
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0

    def HVP(self, observations, actions, vector, regu_coef=None):
        """
        定义了一个名为 `HVP` 的方法

        参数：
        - `observations`：观察值
        - `actions`：动作
        - `vector`：用于计算 Hessian-vector 乘积的向量
        - `regu_coef`：正则化系数（可选，默认为 `self.FIM_invert_args['damping']`）
        """
        regu_coef = self.FIM_invert_args['damping'] if regu_coef is None else regu_coef
        """
        如果未传入正则化系数，使用默认值
        """
        vec = Variable(torch.from_numpy(vector).float(), requires_grad=False)
        """
        将输入的向量转换为 PyTorch 变量
        """
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            """
            如果设置了子采样参数且小于 0.99
            """
            num_samples = observations.shape[0]
            """
            获取观察值的样本数量
            """
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample * num_samples))
            """
            随机选择子样本的索引
            """
            obs = observations[rand_idx]
            """
            选取子样本的观察值
            """
            act = actions[rand_idx]
            """
            选取子样本的动作
            """
        else:
            obs = observations
            """
            否则，使用全部观察值
            """
            act = actions
            """
            否则，使用全部动作
            """
        old_dist_info = self.policy.old_dist_info(obs, act)
        """
        获取旧的分布信息
        """
        new_dist_info = self.policy.new_dist_info(obs, act)
        """
        获取新的分布信息
        """
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        """
        计算平均 KL 散度
        """
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        """
        对平均 KL 散度关于可训练参数求梯度，并创建计算图
        """
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        """
        将梯度展平并连接
        """
        h = torch.sum(flat_grad * vec)
        """
        计算展平梯度与输入向量的乘积之和
        """
        hvp = torch.autograd.grad(h, self.policy.trainable_params)
        """
        对上述乘积关于可训练参数求梯度，得到 Hessian-vector 乘积
        """
        hvp_flat = np.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])
        """
        将 Hessian-vector 乘积展平并转换为 NumPy 数组
        """
        return hvp_flat + regu_coef * vector

    def build_Hvp_eval(self, inputs, regu_coef=None):
        """
        定义了一个名为 `build_Hvp_eval` 的方法，它接受输入 `inputs` 和可选的 `regu_coef`

        参数：
        - `inputs`：可能是用于后续计算的一些输入数据
        - `regu_coef`：可能是一个正则化系数（可选）
        """

        def eval(v):
            """
            在方法内部定义了一个嵌套函数 `eval`，它接受一个参数 `v`

            参数：
            - `v`：可能是与输入 `inputs` 一起用于计算 `Hvp` 的另一个输入值
            """
            full_inp = inputs + [v] + [regu_coef]
            """
            将 `inputs`、`v` 和 `regu_coef` 组合成一个完整的输入列表 `full_inp`
            """
            Hvp = self.HVP(*full_inp)
            """
            使用 `self.HVP` 方法处理 `full_inp` 并得到 `Hvp`
            """
            return Hvp

        """
        返回嵌套函数 `eval`
        """
        return eval

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        """
        该方法用于处理输入的路径数据并进行训练

        参数:
        paths: 要处理的路径数据

        返回:
        base_stats
        """

        # 处理路径数据，根据是否有专家演示路径进行不同处理
        if self.demo_paths is not None and self.lam_0 > 0.0:
            # 从所有轨迹中连接观察值、动作和优势（包括专家数据）
            observations = np.concatenate([path["observations"] for path in paths])
            actions = np.concatenate([path["actions"] for path in paths])
            advantages = np.concatenate([path["advantages"] for path in paths])
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            self.iter_count += 1

            all_obs = np.concatenate([observations, demo_obs])
            all_act = np.concatenate([actions, demo_act])
            all_adv = 1e-2 * np.concatenate([advantages / (np.std(advantages) + 1e-8), demo_adv])

            path_returns = [sum(p["rewards"]) for p in paths]
            mean_return = np.mean(path_returns)
            std_return = np.std(path_returns)
            min_return = np.amin(path_returns)
            max_return = np.amax(path_returns)
            base_stats = [mean_return, std_return, min_return, max_return]
            self.running_score = mean_return if self.running_score is None else \
                0.9 * self.running_score + 0.1 * mean_return  # approx avg of last 10 iters
        else:
            observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)

        if self.save_logs:
            self.log_rollout_statistics(paths)  # 若需保存日志，记录路径统计信息

        # Keep track of times for various computations
        t_gLL = 0.0  # 用于记录 VPG 计算时间
        t_FIM = 0.0  # 用于记录 NPG 计算时间

        # normalize inputs if necessary
        if self.input_normalization:
            """
            若启用输入归一化，计算数据均值和标准差，并更新策略模型的变换参数
            """
            data_in_shift, data_in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            pi_in_shift, pi_in_scale = self.policy.model.in_shift.data.numpy(), self.policy.model.in_scale.data.numpy()
            pi_out_shift, pi_out_scale = self.policy.model.out_shift.data.numpy(), self.policy.model.out_scale.data.numpy()
            pi_in_shift = self.input_normalization * pi_in_shift + (1 - self.input_normalization) * data_in_shift
            pi_in_scale = self.input_normalization * pi_in_scale + (1 - self.input_normalization) * data_in_shift
            self.policy.model.set_transformations(pi_in_shift, pi_in_scale, pi_out_shift, pi_out_scale)

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]  # 优化前的代理损失

        # VPG
        ts = timer.time()  # 记录开始时间
        if self.demo_paths is not None and self.lam_0 > 0.0:
            vpg_grad = self.flat_vpg(all_obs, all_act, all_adv)
        else:
            vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts  # 计算 VPG 时间

        # NPG
        ts = timer.time()  # 记录开始时间
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        if self.demo_paths is not None and self.lam_0 > 0.0:
            npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                                cg_iters=self.FIM_invert_args['iters'])
        else:
            npg_grad = cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                                cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts  # 计算 NPG 时间

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            """
            若 alpha 已给定，直接使用给定值计算步长大小
            """
            alpha = self.alpha
            n_step_size = (alpha ** 2) * np.dot(vpg_grad.T, npg_grad)
        else:
            """
            否则，使用默认方式计算步长大小和 alpha
            """
            n_step_size = self.n_step_size
            alpha = np.sqrt(np.abs(self.n_step_size / (np.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()  # 获取当前策略参数
        new_params = curr_params + alpha * npg_grad  # 计算新的策略参数
        self.policy.set_param_values(new_params, set_new=True, set_old=False)  # 更新策略参数
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]  # 优化后的代理损失
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]  # 计算 KL 散度
        self.policy.set_param_values(new_params, set_new=True, set_old=True)  # 再次更新策略参数

        # Log information
        if self.save_logs:
            """
            若需保存日志，记录相关信息
            """
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
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