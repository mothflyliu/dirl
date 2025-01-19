"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm


class BC:
    def __init__(self, expert_paths,
                 policy,
                 epochs=5,
                 batch_size=64,
                 lr=1e-3,
                 optimizer=None,
                 loss_type='MSE',  # can be 'MLE' or 'MSE'
                 save_logs=True,
                 set_transforms=False,
                 **kwargs,
                 ):
        # 保存策略对象
        self.policy = policy
        # 保存专家路径
        self.expert_paths = expert_paths
        # 设定训练的轮数
        self.epochs = epochs
        # 设定小批量的大小
        self.mb_size = batch_size
        # 初始化日志记录器
        self.logger = DataLog()
        # 设定损失类型
        self.loss_type = loss_type
        # 设定是否保存日志
        self.save_logs = save_logs

        # 如果需要设置变换
        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # 构建优化器，如果没有提供则使用默认的 Adam 优化器
        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=lr) if optimizer is None else optimizer

        # 根据损失类型设置损失准则
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

        # 如果要保存日志，创建日志对象
        if self.save_logs:
            self.logger = DataLog()

    def compute_transformations(self):
        # 如果专家路径为空或为 None
        if self.expert_paths == [] or self.expert_paths is None:
            # 将变换设置为 None
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            # 拼接所有专家路径中的观察值
            observations = np.concatenate([path["observations"] for path in self.expert_paths])
            # 拼接所有专家路径中的动作
            actions = np.concatenate([path["actions"] for path in self.expert_paths])
            # 计算观察值的均值和标准差，作为输入的变换
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            # 计算动作的均值和标准差，作为输出的变换
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        # 返回输入和输出的变换（均值和标准差）
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # 在目标策略中设置变换（缩放和平移）
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        # 在目标策略的旧模型中设置相同的变换
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # 获取策略的参数值
        params = self.policy.get_param_values()
        # 对参数的一部分进行设置，将其设置为对数形式的输出标准差（加上一个小的常数以避免数值问题）
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        # 将修改后的参数值设置回策略
        self.policy.set_param_values(params)

    def loss(self, data, idx=None):
        # 根据损失类型选择计算损失的方法
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # 如果没有提供索引，使用所有数据的索引
        idx = range(data['observations'].shape[0]) if idx is None else idx
        # 如果索引是整数类型，转换为张量类型
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        # 获取观测值和专家动作
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        # 计算新的分布信息
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # 最小化负对数似然
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None):
        # 处理索引
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        # 获取观测值和专家动作
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx]
        # 如果观测值不是张量类型，转换为张量并设置相关属性
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
            act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
        # 通过策略模型获取动作
        act_pi = self.policy.model(obs)
        # 计算均方误差损失
        return self.loss_criterion(act_pi, act_expert.detach())

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # 验证数据是否包含必要的键
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        # 记录开始时间
        ts = timer.time()
        num_samples = data["observations"].shape[0]

        # 如果要保存日志，记录训练前的损失值
        if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)

        # 训练循环
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):
                # 随机选择小批量的索引
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                # 计算小批量的损失
                loss = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()
        # 获取优化后的参数值并设置回策略
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)

        # 如果要保存日志，记录训练后的相关信息
        if self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time() - ts))

    def train(self, **kwargs):
        # 拼接专家路径中的观测值和动作
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        data = dict(observations=observations, expert_actions=expert_actions)
        # 调用拟合方法进行训练
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)