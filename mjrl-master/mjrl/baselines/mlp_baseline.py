import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from mjrl.utils.optimize_model import fit_data

import pickle

class MLPBaseline:
    def __init__(self, env_spec, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, use_gpu=False, hidden_sizes=(128, 128)):
        # 如果没有提供输入维度，则使用环境规格中的观察维度
        self.n = inp_dim if inp_dim is not None else env_spec.observation_dim
        # 设定批量大小
        self.batch_size = batch_size
        # 设定训练轮数
        self.epochs = epochs
        # 设定正则化系数
        self.reg_coef = reg_coef
        # 设定是否使用 GPU
        self.use_gpu = use_gpu
        # 设定输入类型（默认为'obs'，可能是观察值）
        self.inp = inp
        # 设定隐藏层的尺寸
        self.hidden_sizes = hidden_sizes

        # 创建神经网络模型序列
        self.model = nn.Sequential()
        # 定义各层的尺寸
        layer_sizes = (self.n + 4,) + hidden_sizes + (1,)
        # 为神经网络添加层
        for i in range(len(layer_sizes) - 1):
            layer_id = 'fc_' + str(i)
            relu_id = 'elu_' + str(i)
            self.model.add_module(layer_id, nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # 如果不是最后一层，添加 ReLU 激活函数
            if i != len(layer_sizes) - 2:
                self.model.add_module(relu_id, nn.ReLU())

        # 如果使用 GPU，将模型转移到 GPU 上
        if self.use_gpu:
            self.model.cuda()

        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        # 定义损失函数为均方误差损失
        self.loss_function = torch.nn.MSELoss()

    def _features(self, paths):
        # 根据输入类型决定拼接的内容
        if self.inp == 'env_features':
            o = np.concatenate([path["env_infos"]["env_features"][0] for path in paths])
        else:
            o = np.concatenate([path["observations"] for path in paths])
        # 对拼接后的结果进行裁剪并归一化
        o = np.clip(o, -10, 10) / 10.0
        # 如果维度大于 2，进行形状重塑
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        # 获取形状信息
        N, n = o.shape
        # 计算特征数量
        num_feat = int(n + 4)  # 包括线性特征和时间相关特征
        feat_mat = np.ones((N, num_feat))  # 分配内存

        # 填充线性特征部分
        feat_mat[:, :n] = o

        k = 0  # 起始行索引
        # 遍历路径
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l) / 1000.0
            # 填充时间相关特征
            for j in range(4):
                feat_mat[k:k + l, -4 + j] = al ** (j + 1)
            k += l
        # 返回构建好的特征矩阵
        return feat_mat


    def fit(self, paths, return_errors=False):

        featmat = self._features(paths)
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        featmat = featmat.astype('float32')
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        if self.use_gpu:
            featmat_var = Variable(torch.from_numpy(featmat).cuda(), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns).cuda(), requires_grad=False)
        else:
            featmat_var = Variable(torch.from_numpy(featmat), requires_grad=False)
            returns_var = Variable(torch.from_numpy(returns), requires_grad=False)

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

        epoch_losses = fit_data(self.model, featmat_var, returns_var, self.optimizer,
                                self.loss_function, self.batch_size, self.epochs)

        if return_errors:
            if self.use_gpu:
                predictions = self.model(featmat_var).cpu().data.numpy().ravel()
            else:
                predictions = self.model(featmat_var).data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def predict(self, path):
        # 将路径转换为特征矩阵
        featmat = self._features([path]).astype('float32')
        # 如果使用 GPU
        if self.use_gpu:
            # 将特征矩阵转换为 GPU 上的张量变量
            feat_var = Variable(torch.from_numpy(featmat).float().cuda(), requires_grad=False)
            # 使用模型进行预测，并将结果移回 CPU 并转换为 numpy 数组并展平
            prediction = self.model(feat_var).cpu().data.numpy().ravel()
        else:
            # 将特征矩阵转换为 CPU 上的张量变量
            feat_var = Variable(torch.from_numpy(featmat).float(), requires_grad=False)
            # 使用模型进行预测，并转换为 numpy 数组并展平
            prediction = self.model(feat_var).data.numpy().ravel()
        # 返回预测结果
        return prediction