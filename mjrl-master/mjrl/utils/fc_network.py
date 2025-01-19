import numpy as np
import torch
import torch.nn as nn


class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),   # 隐藏层大小
                 nonlinearity='tanh',    # either 'tanh' or 'relu' 非线性函数的类型
                 in_shift = None,        # 输入量偏移
                 in_scale = None,        # 输入量缩放
                 out_shift = None,       # 输出量偏移
                 out_scale = None):      # 输出量缩放
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim              # 观测维度
        self.act_dim = act_dim              # 动作维度
        assert type(hidden_sizes) == tuple  # 元组类型
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )         # 网络层大小、包含
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)  # 调用其方法设置输入输出的变换参数

        # hidden layers
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])         # 列表推导式创建全连接层的模块列表
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh  # 使用其值要选择两个非线性函数之一作为非线性激活函数

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        # 这个方法用于设置输入和输出的变换参数 作用是为网络的输入和输出提供可定制的变换参数，以便对数据进行预处理或后处理
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        # 如果 `in_shift` 不为 `None` ，将其转换为 `torch` 张量；否则，创建一个维度为 `self.obs_dim` 的全零张量
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        # TODO(Aravind): Remove clamping to CPU
        # This is a temp change that should be fixed shortly
        # 定义了网络的前向传播过程
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
            # 处理输入 `x` 的设备问题。如果 `x` 在 GPU 上（`x.is_cuda` 为 `True`），将其移到 CPU 上；否则，直接使用 `x` 作为 `out` 。
        out = (out - self.in_shift)/(self.in_scale + 1e-8)
        # 对 `out` 进行标准化处理，通过减去输入的偏移 `self.in_shift` 并除以输入的缩放 `self.in_scale + 1e-8` 来实现。这里加上一个很小的数 `1e-8` 是为了避免除以零的情况。
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
            # 通过隐藏层进行前向传播。对于除了最后一层的隐藏层，依次进行线性变换（通过 `self.fc_layers[i]` ）和非线性激活（通过 `self.nonlinearity` ）。
        out = self.fc_layers[-1](out)
        out = out * self.out_scale + self.out_shift
        # 通过输出层进行线性变换（通过 `self.fc_layers[-1]` ），并乘以输出的缩放 `self.out_scale` 再加上输出的偏移 `self.out_shift` ，得到最终的输出并返回。
        return out
