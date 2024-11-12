import torch  # 导入 PyTorch 库
import torch.autograd as ag  # 导入 PyTorch 自动求导模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 神经网络函数模块
import numpy as np  # 导入 NumPy 库
import math  # 导入数学库
import functools  # 导入 functools 库
import random  # 导入随机库
import pandas as pd  # 导入 pandas 库
import numpy as np  # 重新导入 NumPy 库，可能是多余的
import matplotlib  # 导入 matplotlib 库
matplotlib.use("Agg")  # 设置 matplotlib 使用 Agg 后端
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import seaborn as sns  # 导入 seaborn 库
from torch.nn import functional as F  # 导入 PyTorch 的神经网络函数模块，并重命名为 F
from .layers import *  # 从当前目录下的 layers 模块中导入所有内容

def mean_distance(a, b, weight=None, training=True):
    """
    计算两个张量之间的平均距离
    参数:
        a: 张量 a
        b: 张量 b
        weight: 权重张量，默认为 None
        training: 是否处于训练模式，默认为 True
    返回:
        平均距离张量
    """
    dis = ((a - b) ** 2).sum(-1)

    if weight is not None:
        dis *= weight

    if not training:
        return dis
    else:
        return dis.mean().unsqueeze(0)


def distance(a, b):
    """
    计算两个张量之间的距离
    参数:
        a: 张量 a
        b: 张量 b
    返回:
        距离张量
    """
    return ((a - b) ** 2).sum(-1)


def heatmap(x, name='heatmap'):
    """
    生成热图
    参数:
        x: 输入张量
        name: 热图名称，默认为 'heatmap'
    返回:
        生成的热图结果
    """
    x = x.squeeze(-1)
    for j in range(x.shape[2]):
        plt.cla()
        y = x[0, :, j].reshape((32, 32))
        df = pd.DataFrame(y.data.cpu().numpy())
        sns.heatmap(df)
        plt.savefig('results/heatmap/{}_{}.png'.format(name, str(j)))
        plt.close()
    return True


class Meta_Prototype(nn.Module):
    def __init__(self, proto_size, feature_dim, key_dim, temp_update, temp_gather, shrink_thres=0):
        """
        Meta_Prototype 类的构造函数
        参数:
            proto_size: 原型大小
            feature_dim: 特征维度
            key_dim: 关键维度
            temp_update: 更新温度
            temp_gather: 聚集温度
            shrink_thres: 收缩门限，默认为0
        """
        super(Meta_Prototype, self).__init__()
        # 常量
        self.proto_size = proto_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        # 多头
        self.Mheads = nn.Linear(key_dim, proto_size, bias=False)
        self.shrink_thres = shrink_thres

    def get_score(self, pro, query):
        """
        获取得分
        参数:
            pro: 原型张量
            query: 查询张量
        返回:
            查询得分
        """
        bs, n, d = query.size()  # n=w*h
        bs, m, d = pro.size()
        score = torch.bmm(query, pro.permute(0, 2, 1))  # b X h X w X m
        score = score.view(bs, n, m)  # b X n X m

        score_query = F.softmax(score, dim=1)
        score_proto = F.softmax(score, dim=2)

        return score_query, score_proto

    def forward(self, key, query, weights, train=True):
        """
        前向传播
        参数:
            key: 键张量
            query: 查询张量
            weights: 权重张量
            train: 是否处于训练模式，默认为 True
        返回:
            更新的查询张量、原型张量、特征损失、一致性损失、距离损失
        """
        batch_size, dims, h, w = key.size()  # b X d X h X w
        key = key.permute(0, 2, 3, 1)  # b X h X w X d
        _, _, h_, w_ = query.size()
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        # 特征图bs,h,w,c-->bs,hw,c
        query = query.reshape((batch_size, -1, self.feature_dim))
        # 训练
        if train:
            if weights == None:
                multi_heads_weights = self.Mheads(key)
            else:
                # 取权重项, 取10个，h,w,10
                multi_heads_weights = linear(key, weights['prototype.Mheads.weight'])
            multi_heads_weights = multi_heads_weights.view((batch_size, h * w, self.proto_size, 1))
            # softmax on weights 将权重经过softmax转换为概率分布
            multi_heads_weights = F.softmax(multi_heads_weights, dim=1)
            key = key.reshape((batch_size, w * h, dims))
            protos = multi_heads_weights * key.unsqueeze(-2)
            protos = protos.sum(1)
            # 损失函数对注意力机制做出了限制，限制10个区域应该关注不同的区域
            updated_query, fea_loss, cst_loss, dis_loss = self.query_loss(query, protos, weights, train)

            # skip connection 更新后的权重项加入到原来的特征图
            updated_query = updated_query + query

            # reshape 调整到原来的维度
            updated_query = updated_query.permute(0, 2, 1)  # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, protos, fea_loss, cst_loss, dis_loss

        # test
        else:
            if weights == None:
                multi_heads_weights = self.Mheads(key)
            else:
                multi_heads_weights = linear(key, weights['prototype.Mheads.weight'])

            multi_heads_weights = multi_heads_weights.view((batch_size, h * w, self.proto_size, 1))

            # softmax on weights
            multi_heads_weights = F.softmax(multi_heads_weights, dim=1)

            key = key.reshape((batch_size, w * h, dims))
            protos = multi_heads_weights * key.unsqueeze(-2)
            protos = protos.sum(1)

            # loss
            updated_query, fea_loss, query = self.query_loss(query, protos, weights, train)

            # skip connection
            updated_query = updated_query + query
            # reshape
            updated_query = updated_query.permute(0, 2, 1)  # b X d X n
            updated_query = updated_query.view((batch_size, self.feature_dim, h_, w_))
            return updated_query, protos, query, fea_loss

    def query_loss(self, query, keys, weights, train):
        """
        查询损失函数
        参数:
            query: 查询张量
            keys: 键张量
            weights: 权重张量
            train: 是否处于训练模式
        返回:
            更新的查询张量、特征损失、一致性损失、距离损失
        """
        batch_size, n, dims = query.size()  # b X n X d, n=w*h
        if train:
            # Distinction constrain
            # proto两两之间算距离
            keys_ = F.normalize(keys, dim=-1)
            # 没有用到相似度
            similarity = torch.bmm(keys_, keys_.permute(0, 2, 1))
            dis = 1 - distance(keys_.unsqueeze(1), keys_.unsqueeze(2))
            mask = dis > 0
            dis *= mask.float()
            dis = torch.triu(dis, diagonal=1)  # 距离是对称矩阵，返回上三角部分
            dis_loss = dis.sum(1).sum(1) * 2 / (self.proto_size * (self.proto_size - 1))
            dis_loss = dis_loss.mean()

            # maintain the consistance of same attribute vector
            cst_loss = mean_distance(keys_[1:], keys_[:-1])

            # Normal constrain
            loss_mse = torch.nn.MSELoss()

            keys = F.normalize(keys, dim=-1)
            # 使用点集计算query和proto的相似度得分，softmax最终将得分转换为概率分布
            _, softmax_score_proto = self.get_score(keys, query)
            # 相似度作用到keys
            new_query = softmax_score_proto.unsqueeze(-1) * keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            # maintain the distinction among attribute vectors
            # 根据相似度得分，取最像的
            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)
            # 1st closest memories 取出最像的
            pos = torch.gather(keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))
            # 最像的和原来query特征图的相似度
            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, cst_loss, dis_loss


        else:
            loss_mse = torch.nn.MSELoss(reduction='none')

            keys = F.normalize(keys, dim=-1)
            softmax_score_query, softmax_score_proto = self.get_score(keys, query)

            new_query = softmax_score_proto.unsqueeze(-1) * keys.unsqueeze(1)
            new_query = new_query.sum(2)
            new_query = F.normalize(new_query, dim=-1)

            _, gathering_indices = torch.topk(softmax_score_proto, 2, dim=-1)

            # 1st closest memories
            pos = torch.gather(keys, 1, gathering_indices[:, :, :1].repeat((1, 1, dims)))

            fea_loss = loss_mse(query, pos)

            return new_query, fea_loss, query


