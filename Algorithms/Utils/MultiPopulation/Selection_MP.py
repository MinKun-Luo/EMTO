# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 多种群选择算子
# @Remind: 同索引的父代与子代构成锦标赛池

import numpy as np

from Problems.Problem import Problem


def Selection_Elit(population, offspring, num=Problem.N):
    """
    根据因子成本，使用精英保留策略更新新种群

    :param population: list，父代个体列表
    :param offspring: list，子代个体列表
    :param num: int，保留个体数量，默认为 Problem.N
    :return: population: list，精英保留后的个体列表
    """
    # 合并父代与子代
    population.extend(offspring)
    # 获取所有个体的适应度（obj 值）
    costs = np.fromiter((ind.obj for ind in population), dtype=float)
    # 获取按适应度升序排序的索引
    best_idx = np.argsort(costs)[:num]
    # 精英保留：选择适应度最小的前 num 个个体
    return [population[i] for i in best_idx]


def Selection_Tournament(population, offspring):
    """
    使用锦标赛选择更新新种群（同索引的父代与子代进行比较）

    :param population: list，父代个体列表
    :param offspring: list，子代个体列表
    :return:
        population: list，锦标赛选择后的个体列表
        replace: numpy.array，标志是否由子代替换（True 表示子代获胜）
    """
    # 初始化替换标记，默认不替换（父代保留）
    replace = np.zeros(len(population), dtype=bool)

    # 遍历个体，逐个父子比拼
    for i in range(len(population)):
        if offspring[i].obj < population[i].obj:
            # 子代适应度更优，替换父代
            population[i] = offspring[i]
            replace[i] = True

    return population, replace
