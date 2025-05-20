# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: 变异算子

import copy

import numpy as np


def PM1(p, mu, low=0, up=1):
    """
    多项式变异(PM)函数，用于遗传算法中的变异操作。

    :param p: numpy 数组，表示待变异个体
    :param mu: float，变异分布指数，控制变异的强度
    :param low: float，变异后个体值的下界，默认为0
    :param up: float，变异后个体值的上界，默认为1
    :return: numpy 数组，变异后的个体
    """
    D = p.shape[0]  # 决策向量的维度
    p_temp = np.copy(p)  # 复制决策向量
    for d in range(D):  # 遍历决策向量中的每个元素
        if np.random.rand() < 1 / D:  # 按照 1/D 的概率决定是否对当前元素进行变异
            u = np.random.rand()  # 生成一个均匀随机数 u，用于计算变异量
            if u <= 0.5:  # 根据 u 的值选择变异的公式
                # 计算变异量 delta，应用在 u <= 0.5 的情况
                delta = ((2 * u + (1 - 2 * u) * (1 - p[d]) ** (mu + 1)) ** (1 / (mu + 1))) - 1
                p_temp[d] += delta  # 更新决策向量中的当前元素
            else:
                # 计算变异量 delta，应用在 u > 0.5 的情况
                delta = 1 - ((2 * (1 - u) + 2 * (u - 0.5) * p[d] ** (mu + 1)) ** (1 / (mu + 1)))
                p_temp[d] += delta  # 更新决策向量中的当前元素
    p_temp = np.clip(p_temp, low, up)  # 边界处理

    return p_temp  # 返回变异后的决策向量


def PM2(p, mu, low=0, up=1):
    """
    多项式变异(PM)函数，用于遗传算法中的变异操作。

    :param p: numpy 数组，表示待变异个体
    :param mu: float，变异分布指数，控制变异的强度
    :param low: float，变异后个体值的下界，默认为0
    :param up: float，变异后个体值的上界，默认为1
    :return: numpy 数组，变异后的个体
    """
    D = p.shape[0]  # 获取个体的维度
    p_temp = np.copy(p)  # 复制个体
    for d in range(D):  # 遍历个体的每一个维度
        if np.random.rand() < 1 / D:  # 以1/D的概率决定是否对当前维度进行变异
            u = np.random.rand()  # 生成一个0到1之间的随机数
            if u <= 0.5:  # 根据随机数u的值选择变异公式
                # 计算变异量delta，当u小于等于0.5时使用此公式
                delta = (2 * u) ** (1 / (1 + mu)) - 1
                # 更新个体的当前维度值
                p_temp[d] = p[d] + delta * p[d]
            else:
                # 计算变异量delta，当u大于0.5时使用此公式
                delta = 1 - (2 * (1 - u)) ** (1 / (1 + mu))
                # 更新个体的当前维度值
                p_temp[d] = p[d] + delta * (1 - p[d])
    # 将个体的值限制在[low, up]范围内
    p_temp = np.clip(p_temp, low, up)
    # 返回变异后的个体
    return p_temp


def gaussian_mutation(p, sigma):
    """
    高斯变异(Gaussian Mutation)函数，用于遗传算法中的变异操作（只改变单个维度）。

    :param p: numpy 数组，表示待变异个体
    :param sigma: float，高斯分布标准差
    :return: numpy 数组，变异后的个体
    """
    p_temp = copy.deepcopy(p)
    # 选择变异基因的索引
    gene_idx = np.random.randint(len(p_temp))
    # 生成高斯噪声
    noise = np.random.normal(0, sigma)
    # 进行变异
    p_temp[gene_idx] += noise

    p_temp = np.clip(p_temp, 0, 1)

    return p_temp


def all_dim_gaussian_mutation(ind, p, sigma):
    """
    高斯变异(Gaussian Mutation)函数，用于遗传算法中的变异操作（以条件概率改变多个维度）。

    :param ind: numpy 数组，表示待变异个体
    :param p: float，变异维度选取概率
    :param sigma: float，高斯分布标准差
    :return: numpy 数组，变异后的个体
    """
    p_temp = copy.deepcopy(ind)
    for d in range(len(p_temp)):
        if np.random.rand() < p:
            # 生成高斯噪声
            noise = np.random.normal(0, sigma)
            # 进行变异
            p_temp[d] += noise

    p_temp = np.clip(p_temp, 0, 1)

    return p_temp
