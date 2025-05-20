# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: DE算子

import numpy as np


def DERand1(p, p1, p2, p3, D, F=0.5, CR=0.6):
    """
    差分进化算法（DE/rand/1）函数，变异与交叉

    :param p: numpy 数组，表示待进化个体基因
    :param p1: numpy 数组，表示用于变异操作的个体基因
    :param p2: numpy 数组，表示用于变异操作的个体基因
    :param p3: numpy 数组，表示用于变异操作的个体基因
    :param D: int，个体维度
    :param F: float，变异缩放因子，默认值为0.5
    :param CR: float，交叉概率，默认值为0.6
    :return: numpy 数组，变异后的个体基因
    """
    # 变异：生成变异向量 v
    v = p1 + F * (p2 - p3)

    # 交叉：生成交叉向量
    r = np.random.randint(D)  # 随机选择一个索引 r
    for i in range(D):
        # 如果随机数大于交叉概率 CR 且索引不等于 r，则使用原向量 p 的值
        if np.random.rand() > CR and i != r:
            v[i] = p[i]

    # 边界控制：将变异向量 v 的值限制在 [0, 1] 范围内
    np.clip(v, 0, 1, out=v)

    # 返回生成的向量 v
    return v


def DE(p, v, D, CR=0.6):
    """
    差分进化算法（DE/rand/1）函数，只交叉

    :param p: numpy 数组，表示待进化个体基因
    :param v: numpy 数组，表示变异个体的基因
    :param D: int，个体维度
    :param CR: float，交叉概率，默认值为0.6
    :return: numpy 数组，变异后的个体基因
    """
    # 交叉：生成交叉向量
    r = np.random.randint(D)  # 随机选择一个索引 r
    for i in range(D):
        # 如果随机数大于交叉概率 CR 且索引不等于 r，则使用原向量 p 的值
        if np.random.rand() > CR and i != r:
            v[i] = p[i]

    # 边界控制：将变异向量 v 的值限制在 [0, 1] 范围内
    np.clip(v, 0, 1, out=v)

    # 返回生成的向量 v
    return v
