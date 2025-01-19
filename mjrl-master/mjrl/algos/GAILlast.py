import logging

from mjrl.utils.gym_env import GymEnv

logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm
from mjrl.algos.batch_reinforce import BatchREINFORCE
import torch
import torch.nn.functional as F
from mjrl.utils.train_agent_ga import train_agent
class GAIL:
    def __init__(self,
                 expert_paths,
                 job_name,
                 policy,
                 agent,
                 save_freq,
                 evaluation_rollouts,
                 epochs=5,
                 seed=0,
                 batch_size=64,
                 lr_policy=1e-3,
                 lr_discriminator=1e-2,
                 optimizer_policy=None,
                 discriminator=None,
                 optimizer_discriminator=None,
                 save_logs=True,
                 set_transforms=False,
                 sample_mode='trajectories',
                 gamma=0.995,
                 gae_lambda=None,
                 num_cpu=1,
                 num_traj=50,
                 niter=100,
                 num_samples=50000,
                 **kwargs):
        # 保存策略对象
        self.policy = policy
        # 保存判别器对象
        self.discriminator = discriminator
        # 保存专家路径
        self.expert_paths = expert_paths
        # 设定训练的轮数
        self.epochs = epochs
        # 设定小批量的大小
        self.mb_size = batch_size
        # 初始化策略网络的日志记录器
        self.policy_logger = DataLog()
        # 初始化判别器的日志记录器
        self.discriminator_logger = DataLog()
        # 设定是否保存日志
        self.save_logs = save_logs
        # 设定模式
        self.sample_mode = sample_mode
        # 采样轨迹
        self.N = num_traj
        self.niter = niter
        self.job_name = job_name
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_cpu = num_cpu
        self.num_traj = num_traj
        self.num_samples = num_samples
        self.save_freq = save_freq
        self.evaluation_rollouts = evaluation_rollouts
        self.agent = agent
        self.seed = seed
        # 如果需要设置变换
        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # 构建策略网络的优化器，如果没有提供则使用默认的Adam优化器
        self.optimizer_policy = torch.optim.Adam(self.policy.trainable_params, lr=lr_policy) if optimizer_policy is None else optimizer_policy
        # 构建判别器的优化器，如果没有提供则使用默认的Adam优化器
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.trainable_params, lr=lr_discriminator) if optimizer_discriminator is None else optimizer_discriminator

    def compute_transformations(self):
        # 如果专家路径为空或为None
        if self.expert_paths == [] or self.expert_paths is None:
            # 将变换设置为None
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

    def discriminator_loss(self, expert_observation, expert_action, generated_observation, generated_action):
        # 真实数据通过判别器的输出
        expert_output = self.discriminator.forward(expert_observation, expert_action)
        # 生成数据通过判别器的输出
        generated_output = self.discriminator.forward(generated_observation, generated_action)

        # 处理expert_output
        mask_expert_greater_than_1 = expert_output > 1
        mask_expert_less_than_0 = expert_output < 0
        if mask_expert_greater_than_1.any():
            expert_output[mask_expert_greater_than_1] = 1
        if mask_expert_less_than_0.any():
            expert_output[mask_expert_less_than_0] = 0

        # 处理generated_output
        mask_generated_greater_than_1 = generated_output > 1
        mask_generated_less_than_0 = generated_output < 0
        if mask_generated_greater_than_1.any():
            generated_output[mask_generated_greater_than_1] = 1
        if mask_generated_less_than_0.any():
            generated_output[mask_generated_less_than_0] = 0

        # 计算判别器的损失，目标是让判别器能够区分真实数据和生成数据
        try:
            loss = torch.nn.BCELoss()(expert_output, torch.zeros_like(expert_output)) + \
                   torch.nn.BCELoss()(generated_output, torch.ones_like(generated_output))
        except Exception as e:
            print("An error occurred.", )
            print("expert_output:", expert_output)
            print("generated_output:", generated_output)
            raise e
        return loss
    def discriminator_output(self, data):
        # 策略生成的数据
        generated_actions = (data["actions"])
        # 生成数据通过判别器的输出
        discriminator_output = self.discriminator.forward(data["observations"], generated_actions)
        # 计算策略的损失，目标是让生成的数据能够骗过判别器

        return discriminator_output

    def fit_discriminator(self, expert_data, generated_data, suppress_fit_tqdm=False):
        # 验证数据是否包含必要的键
        assert all([k in expert_data.keys() for k in ["observations", "actions"]]) and \
               all([k in generated_data.keys() for k in ["observations", "actions"]])

        # 记录开始时间
        ts = timer.time()

        # 如果要保存日志，记录训练前的损失值
        if self.save_logs:
            loss_val = self.discriminator_loss(expert_data["observations"], expert_data["actions"],
                                               generated_data["observations"], generated_data["actions"]).data.numpy().ravel()[0]
            self.discriminator_logger.log_kv('loss_before', loss_val)

        # 训练判别器循环
        for ep in config_tqdm(range(1), suppress_fit_tqdm):
            #for mb in range(int(generated_data["observations"].shape[0] / self.mb_size)):
            for mb in range(int(generated_data["observations"].shape[0] / 9942)):
                self.optimizer_discriminator.zero_grad()

                # 随机选择小批量的索引
                # rand_idx_expert = np.random.choice(expert_data["observations"].shape[0], size=self.mb_size)
                rand_idx_expert = np.random.choice(expert_data["observations"].shape[0], size=32)
                rand_idx_generated = np.random.choice(generated_data["observations"].shape[0], size=32)

                expert_batch_obs = expert_data["observations"][rand_idx_expert]
                expert_batch_act = expert_data["actions"][rand_idx_expert]
                generated_batch_obs = generated_data["observations"][rand_idx_generated]
                generated_batch_act = generated_data["actions"][rand_idx_generated]
                loss = self.discriminator_loss(expert_batch_obs, expert_batch_act, generated_batch_obs, generated_batch_act)
                loss.backward()
                self.optimizer_discriminator.step()



        # 如果要保存日志，记录训练后的相关信息
        if self.save_logs:
            self.discriminator_logger.log_kv('epoch', self.epochs)
            loss_val = self.discriminator_loss(expert_data["observations"], expert_data["actions"],
                                               generated_data["observations"], generated_data["actions"]).data.numpy().ravel()[0]
            self.discriminator_logger.log_kv('loss_after', loss_val)
            self.discriminator_logger.log_kv('time', (timer.time() - ts))

    def new_stats(self, data, stats):
        # 验证数据是否包含必要的键
        validate_keys = all([k in data.keys() for k in ["observations", "actions"]])
        assert validate_keys is True
        batch_data_obs = data["observations"]
        batch_actions = data["actions"]

        rewards = -torch.log(self.discriminator_output({
            "observations": batch_data_obs,
            "actions": batch_actions
        }))

        rewards = rewards.detach().numpy()
        # 记录当前取用rewards的索引位置，初始化为0
        used_index = 0
        for item in stats:
            old_rewards = item['rewards']
            shape = old_rewards.shape
            num_elements_needed = np.prod(shape)
            # 从当前索引位置开始取相应数量的元素
            new_rewards = rewards[used_index:used_index + num_elements_needed].reshape(shape)
            item['rewards'] = new_rewards
            # 更新索引位置，指向下一次取用元素的起始处
            used_index += num_elements_needed
        return stats


    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        return observations, actions

    def train(self, **kwargs):
        np.random.seed(self.seed)
        GymEnv(self.agent.env.env_id)
        # 拼接专家路径中的观测值和动作，作为专家数据
        expert_observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        expert_data = dict(observations=expert_observations, actions=expert_actions)

        # 根据采样模式确定采样数量
        N = self.num_traj if self.sample_mode == 'Discriminator' else self.num_samples
        # 构建训练参数
        args = dict(N=N, sample_mode=self.sample_mode, gamma=self.gamma, gae_lambda=self.gae_lambda, num_cpu=self.num_cpu)
        # 执行训练步骤并获取统计信息
        stats = self.agent.train_step(**args)
        generated_observations, generated_actions, = self.process_paths(stats)
        generated_data = dict(observations=generated_observations, actions=generated_actions)
        # 设置随机数种子
        np.random.seed(self.seed)
        # 创建 Gym 环境
        e = GymEnv(self.agent.env.env_id)
        # 交替训练判别器和策略
        for i in range(self.epochs):
            self.fit_discriminator(expert_data, generated_data)
            new_stats = self.new_stats(generated_data, stats)
            train_agent(job_name=self.job_name,
                        a=i,
                        e=e,
                        p=new_stats,
                        agent=self.agent,
                        seed=self.seed,
                        niter=self.niter,
                        gamma=self.gamma,
                        gae_lambda=self.gae_lambda,
                        num_cpu=self.num_cpu,
                        sample_mode='GAIL',
                        num_traj=self.num_traj,
                        save_freq=self.save_freq,
                        evaluation_rollouts=self.evaluation_rollouts)
            # 执行训练步骤并获取统计信息
            stats = self.agent.train_step(**args)
            generated_observations, generated_actions, = self.process_paths(stats)
            generated_data = dict(observations=generated_observations, actions=generated_actions)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)