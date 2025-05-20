# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: 交叉算子

import numpy as np


def BinomialCrossover(p1, p2, dim):
    """
    执行二项式交叉（Binomial Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param dim: 个体维度
    :return: 子代个体1, 子代个体2
    """
    c1, c2 = p1.copy(), p2.copy()
    # 随机生成交叉概率
    CR = np.random.uniform(0.1, 0.9)
    # 随机选择一个索引 r
    r1 = np.random.randint(dim)
    r2 = np.random.randint(dim)
    # 随机生成交叉点
    for d in range(dim):
        if np.random.rand() <= CR or d == r1:
            c1[d] = p2[d]
        if np.random.rand() <= CR or d == r2:
            c2[d] = p1[d]

    return c1, c2


def TwoPointCrossover(p1, p2, dim):
    """
    执行两点交叉（Two-Point Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param dim: 个体维度
    :return: 子代个体1, 子代个体2
    """
    # 初始化两个子代
    c1, c2 = np.zeros(dim), np.zeros(dim)
    # 随机生成两个不相同的索引，并排序
    idx1, idx2 = sorted(np.random.choice(range(0, dim), 2, replace=False))
    # 交叉生成子代个体
    c1[:idx1], c1[idx1:idx2 + 1], c1[idx2 + 1:] = p1[:idx1], p2[idx1:idx2 + 1], p1[idx2 + 1:]
    # 随机生成两个不相同的索引，并排序
    idx1, idx2 = sorted(np.random.choice(range(0, dim), 2, replace=False))
    # 交叉生成子代个体
    c2[:idx1], c2[idx1:idx2 + 1], c2[idx2 + 1:] = p2[:idx1], p1[idx1:idx2 + 1], p2[idx2 + 1:]

    return c1, c2


def UniformCrossover(p1, p2, dim):
    """
    执行均匀交叉（TUniform Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param dim: 个体维度
    :return: 子代个体1, 子代个体2
    """
    c1, c2 = np.zeros(dim), np.zeros(dim)
    for d in range(dim):
        # 如果随机数小于等于0.5，则c1(c2)继承父代p1(c2)的基因，否则继承父代p2(p1)的基因
        c1[d] = p1[d] if np.random.rand() <= 0.5 else p2[d]
        c2[d] = p1[d] if np.random.rand() <= 0.5 else p2[d]

    return c1, c2


def ArithmeticalCrossover(p1, p2, lamda=0.25, low=0, up=1):
    """
    执行算术交叉（Arithmetical Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param lamda: 预定义系数，通常位于[0,1]
    :param low: 变量下界，默认值为0
    :param up: 变量上界，默认值为1
    :return: 子代个体1, 子代个体2
    """
    # 计算子代个体
    c1 = lamda * p1 + (1 - lamda) * p2
    c2 = lamda * p2 + (1 - lamda) * p1
    # 通过边界值吸收法将子代个体的值限制在[low, up]范围内
    c1 = np.clip(c1, low, up)
    c2 = np.clip(c2, low, up)

    return c1, c2


def GeometricalCrossover(p1, p2, omega=0.25, low=0, up=1):
    """
    执行几何交叉（Geometrical Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param omega: 预定义系数，通常位于[0,1]
    :param low: 变量下界，默认值为0
    :param up: 变量上界，默认值为1
    :return: 子代个体1, 子代个体2
    """
    # 计算子代个体
    c1 = p1 ** omega * p2 ** (1 - omega)
    c2 = p2 ** omega * p1 ** (1 - omega)
    # 通过边界值吸收法将子代个体的值限制在[low, up]范围内
    c1 = np.clip(c1, low, up)
    c2 = np.clip(c2, low, up)

    return c1, c2


def BLXalpha(p1, p2, dim, alpha=0.3, low=0, up=1):
    """
    执行混合交叉（BLX-α Crossover）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param dim: 个体维度
    :param alpha: 预定义系数，通常位于[0,1]
    :param low: 变量下界，默认值为0
    :param up: 变量上界，默认值为1
    :return: 子代个体1, 子代个体2
    """
    c1, c2 = np.zeros(dim), np.zeros(dim)
    for d in range(dim):
        # 获取p1和p2在第d维度上的最大值和最小值
        pMax = max(p1[d], p2[d])
        pMin = min(p1[d], p2[d])
        # 计算区间的长度
        I = pMax - pMin
        # 在区间[pMin - alpha * I, pMax + alpha * I]内生成c1和c2的值
        c1[d] = np.random.uniform(pMin - alpha * I, pMax + alpha * I)
        c2[d] = np.random.uniform(pMin - alpha * I, pMax + alpha * I)

    # 通过边界值吸收法将子代个体的值限制在[low, up]范围内
    c1 = np.clip(c1, low, up)
    c2 = np.clip(c2, low, up)

    return c1, c2


def SBX(p1, p2, eta=2, low=0, up=1):
    """
    执行模拟二进制交叉（SBX）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param eta: 分布指数，默认值为2
    :param low: 变量下界，默认值为0
    :param up: 变量上界，默认值为1
    :return: 子代个体1, 子代个体2
    """
    dim = np.max([len(p1), len(p2)])
    # 生成一个范围在0到1之间的随机数数组，形状为dim
    u = np.random.uniform(0, 1, size=dim)
    # 根据随机数u计算交叉因子cf
    cf = np.where(u <= 0.5,
                  np.power(2 * u, 1 / (eta + 1)),
                  np.power(2 * (1 - u), -1 / (eta + 1)))

    # 计算子代个体
    c1 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
    c2 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)

    # 通过边界值吸收法将子代个体的值限制在[low, up]范围内
    c1 = np.clip(c1, low, up)
    c2 = np.clip(c2, low, up)

    return c1, c2


def SBX1(p1, p2, eta=2, low=0, up=1):
    """
    执行模拟二进制交叉（SBX）操作。

    :param p1: 第一个父代个体
    :param p2: 第二个父代个体
    :param eta: 分布指数，默认值为2
    :param low: 变量下界，默认值为0
    :param up: 变量上界，默认值为1
    :return: 子代个体1, 子代个体2
    """
    dim = np.max([len(p1), len(p2)])
    # 生成一个范围在0到1之间的随机数数组，形状为dim
    u = np.random.uniform(0, 1, size=dim)
    # 根据随机数u计算交叉因子cf
    cf = np.where(u <= 0.5,
                  np.power(2 * u, 1 / (eta + 1)),
                  np.power(2 * (1 - u), -1 / (eta + 1)))
    # 引入多样性，随机修改cf的值
    # 随机修改cf的符号
    cf = cf * np.where(np.random.rand(dim) < 0.5, 1, -1)
    # 将大约一半的cf置为1
    cf = np.where(np.random.rand(dim) < 0.5, 1, cf)

    # 计算子代个体
    c1 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
    c2 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)

    # 通过边界值吸收法将子代个体的值限制在[low, up]范围内
    c1 = np.clip(c1, low, up)
    c2 = np.clip(c2, low, up)

    return c1, c2
