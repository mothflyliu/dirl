import numpy as np
from mjrl.utils.fc_network import FCNetwork
import torch
from torch.autograd import Variable


class MLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):
        """
        :param env_spec: 环境的规格说明（见 utils/gym_env.py）
        :param hidden_sizes: 网络隐藏层的大小（目前仅 2 层）
        :param min_log_std: log_std 被限制在这个值，不能低于它
        :param init_log_std: 初始的 log 标准偏差
        :param seed: 随机种子
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network 创建策略网络模型，并将其最后两层的参数值缩小 100 倍。
        # 初始化 log_std 为可训练变量并将模型参数和 log_std 组合为可训练参数列表
        self.model = FCNetwork(self.n, self.m, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # 创建旧的策略网络和对应的 log_std，并将旧网络的参数初始化为新网络对应参数的克隆
        self.old_model = FCNetwork(self.n, self.m, hidden_sizes)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # 计算和存储与可训练参数相关的一些信息，log_std 的值、参数的形状和大小以及参数的总数
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # 创建一个随机初始化的观测变量，且不可训练
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        """
        定义了一个名为 `set_param_values` 的方法，用于设置参数的值

        参数：
        - `new_params`：新的参数值
        - `set_new`：是否设置新的参数（默认为 `True`）
        - `set_old`：是否设置旧的参数（默认为 `True`）
        """
        if set_new:
            """
            如果要设置新的参数
            """
            current_idx = 0
            """
            初始化索引
            """
            for idx, param in enumerate(self.trainable_params):
                """
                遍历可训练的参数
                """
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                """
                提取对应参数的新值
                """
                vals = vals.reshape(self.param_shapes[idx])
                """
                重塑新值的形状
                """
                param.data = torch.from_numpy(vals).float()
                """
                设置参数的数据
                """
                current_idx += self.param_sizes[idx]
            """
            更新索引
            """
            # clip std at minimum value
            self.trainable_params[-1].data = torch.clamp(self.trainable_params[-1], self.min_log_std).data
            """
            对最后一个参数进行裁剪，确保其不小于最小的对数标准差
            """
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
            """
            更新用于采样的对数标准差值
            """
        if set_old:
            """
            如果要设置旧的参数
            """
            current_idx = 0
            """
            初始化索引
            """
            for idx, param in enumerate(self.old_params):
                """
                遍历旧的参数
                """
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                """
                提取对应参数的新值
                """
                vals = vals.reshape(self.param_shapes[idx])
                """
                重塑新值的形状
                """
                param.data = torch.from_numpy(vals).float()
                """
                设置参数的数据
                """
                current_idx += self.param_sizes[idx]
            """
            更新索引
            """
            # clip std at minimum value
            self.old_params[-1].data = torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        """
        定义了一个名为 `mean_LL` 的方法，用于计算均值和对数似然（Log Likelihood，LL）。

        参数：
        - `observations`：观察值。
        - `actions`：动作。
        - `model`：模型，如果未提供则使用自身的 `model`。
        - `log_std`：对数标准差，如果未提供则使用自身的 `log_std`。
        """

        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        """
        如果未传入 `model` 和 `log_std`，则使用当前对象的相应属性。
        """

        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        """
        如果 `observations` 不是 `torch.Tensor` 类型，将其转换为 `torch.Tensor` 并创建为变量。
        否则，直接使用传入的 `observations` 作为变量。
        """

        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        """
        与上面处理 `observations` 类似，对 `actions` 进行相同的类型转换和变量创建操作。
        """

        mean = model(obs_var)
        """
        使用传入或默认的模型对观察值变量进行计算，得到均值。
        """

        zs = (act_var - mean) / torch.exp(log_std)
        """
        计算标准化后的差值 `zs`。
        """

        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        """
        计算对数似然 `LL`。公式包含三项：
        - `- 0.5 * torch.sum(zs ** 2, dim=1)`：与标准化差值的平方和相关。
        - `- torch.sum(log_std)`：与对数标准差相关。
        - `- 0.5 * self.m * np.log(2 * np.pi)`：可能是一个常数项。
        """

        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)

    def sample(self, states):
        o = np.float32(states.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return action, {'mean': mean, 'log_std': self.log_std_val}