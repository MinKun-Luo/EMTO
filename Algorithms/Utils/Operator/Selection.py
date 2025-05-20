# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: 选择算子

import numpy as np


def RouletteWheelSelection(values, num=1):
    """
    基于轮盘赌选择算法选择多个个体的索引。

    :param values: 所有个体的适应度值列表或数组
    :param num: 需要选择的个体数量
    :return: 选中的个体索引（列表）
    """
    selected_indices = []  # 用于保存选中的个体索引

    # 计算适应度总和
    total_fitness = np.sum(values)

    # 每次选择一个个体
    for _ in range(num):
        # 生成0到值总和之间的一个随机数
        rand_point = np.random.uniform(low=0, high=total_fitness)

        accumulator = 0  # 初始化累积器

        # 遍历值列表，累加取值
        for index, fitness in enumerate(values):
            accumulator += fitness

            # 如果累积值大于等于随机数，返回当前索引
            if accumulator >= rand_point:
                selected_indices.append(index)
                break

    return selected_indices


def StochasticUniversalSampling(values, num):
    """
    基于随机遍历选择算法选择个体的索引。

    :param values: 所有个体的取值（列表或数组）
    :param num: 选择个体数（整数）
    :return: 选中的个体索引（数组）
    """
    # 计算总和
    total_fitness = np.sum(values)

    # 步长，计算每个选择点之间的间隔
    step = total_fitness / num

    # 随机选择起始点
    start_point = np.random.uniform(0, step)

    # 累加值
    fitness_cumsum = np.cumsum(values)

    selected_indices = []
    current_point = start_point

    for i in range(num):
        # 查找累积适应度超过当前选择点的个体
        selected_index = np.searchsorted(fitness_cumsum, current_point)
        selected_indices.append(selected_index)

        # 更新当前选择点
        current_point += step

    return selected_indices


def TournamentSelection(values, num, k=5):
    """
    基于锦标赛选择策略选择个体的索引（值较大的）。

    :param values: 所有个体的取值（列表或数组）
    :param num: 选择个体数（整数）
    :param k: 锦标赛池大小（整数）
    :return: 选中的个体索引（数组）
    """
    selected_indices = []

    for _ in range(num):
        # 从种群中随机选择k个个体
        tournament_indices = np.random.choice(len(values), k, replace=False)
        tournament_fitness = np.array([values[i] for i in tournament_indices])

        # 选择取值最大的个体
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_indices.append(winner_index)

    return selected_indices
