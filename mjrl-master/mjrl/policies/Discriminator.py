import numpy as np
import torch
from torch.autograd import Variable
from mjrl.utils.fc_network import FCNetwork


class Discriminator:
    def __init__(self, env_spec,
                 hidden_sizes=(64, 64),
                 seed=None):
        """
        判别器网络的初始化函数

        :param env_spec: 环境的规格说明，用于获取相关维度信息（类似策略网络中的用法）
        :param hidden_sizes: 网络隐藏层的大小（目前仅 2 层）
        :param seed: 随机种子
        """
        self.n = env_spec.observation_dim  # 观察值维度，可根据实际情况用于判别器输入的一部分
        self.m = env_spec.action_dim  # 动作维度，可根据实际情况用于判别器输入的一部分

        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 确定判别器的输入维度，这里假设判别器输入包含观察值和动作
        self.input_dim = self.n + self.m

        # 创建判别器网络模型
        self.model = FCNetwork(self.input_dim, 1, hidden_sizes)

        # 可训练参数列表
        self.trainable_params = list(self.model.parameters())

        # 创建一个随机初始化的输入变量，且不可训练
        self.input_var = Variable(torch.randn(self.input_dim), requires_grad=False)

    # Utility functions
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.trainable_params):
            vals = new_params[current_idx:current_idx + param.data.numpy().size]
            vals = vals.reshape(param.data.numpy().shape)
            param.data = torch.from_numpy(vals).float()
            current_idx += param.data.numpy().size

    # Main functions
    def forward(self, observation, action):
        """
        判别器的前向传播函数，接收观察值和动作作为输入

        :param observation: 观察值
        :param action: 动作
        """
        if type(observation) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observation).float(), requires_grad=False)
        else:
            obs_var = observation

        if type(action) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(action).float(), requires_grad=False)
        else:
            act_var = action

        # 将观察值和动作拼接作为判别器的输入
        input_data = torch.cat((obs_var, act_var), dim=1)

        output = self.model(input_data)
        a = torch.sigmoid(output)
        return a.squeeze()

    def predict(self, observation, action):
        """
        根据判别器的输出进行预测，判断输入数据更可能是来自专家的真实数据还是由策略网络生成的数据

        :param observation: 观察值
        :param action: 动作
        """
        output = self.forward(observation, action)
        return (output > 0.5).numpy().astype(int)

    def loss(self, expert_observation, expert_action, generated_observation, generated_action):
        """
        计算判别器的损失函数，目标是让判别器能够准确区分专家数据和生成数据

        :param expert_observation: 专家的观察值
        :param expert_action: 专家的动作
        :param generated_observation: 由策略网络生成的观察值
        :param generated_action: 由策略网络生成的动作
        """
        expert_output = self.forward(expert_observation, expert_action)
        generated_output = self.forward(generated_observation, generated_action)

        loss = torch.nn.BCELoss()(expert_output, torch.ones_like(expert_output)) + \
               torch.nn.BCELoss()(generated_output, torch.zeros_like(generated_output))

        return loss

    def accuracy(self, expert_observation, expert_action, generated_observation, generated_action):
        """
        计算判别器在区分专家数据和生成数据时的准确率

        :param expert_observation: 专家的观察值
        :param expert_action: 专家的动作
        :param generated_observation: 由策略网络生成的观察值
        :param generated_action: 由策略网络生成的动作
        """
        expert_prediction = self.predict(expert_observation, expert_action)
        generated_prediction = self.predict(generated_observation, generated_action)

        expert_correct = np.sum(expert_prediction == 1)
        generated_correct = np.sum(generated_prediction == 0)

        total = len(expert_observation) + len(generated_observation)
        accuracy = (expert_correct + generated_correct) / total

        return accuracy