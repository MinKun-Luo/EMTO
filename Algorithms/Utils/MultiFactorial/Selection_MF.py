# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 多因子算法中用于选择操作的精英保留机制
# @Remind: 根据标量适应度从父代和子代合并种群中选择前 num 个个体

import numpy as np

from Problems.Problem import Problem


def Selection_MF(population, offspring, num=Problem.N):
    """
    根据标量适应度，使用精英保留策略更新种群

    :param population: list，原始父代种群
    :param offspring: list，新产生的子代种群
    :param num: int，保留的个体数量（默认 Problem.N）
    :return: population: list，更新后的种群
    """
    # 合并种群
    population.extend(offspring)

    # 重新计算计算因子排名
    for t in range(Problem.T):
        costs = np.array([ind.MFCosts[t] for ind in population])
        sorted_idx = np.argsort(costs)
        for rank, idx in enumerate(sorted_idx):
            population[idx].MFRanks[t] = rank + 1  # 排名从1开始

    # 更新每个个体的技能因子（最低排名对应的任务，多个随机选一个）
    for ind in population:
        min_rank = np.min(ind.MFRanks)
        tied_tasks = np.where(ind.MFRanks == min_rank)[0]
        ind.MFFactor = np.random.choice(tied_tasks) + 1
        ind.obj = ind.MFCosts[ind.MFFactor - 1]  # 更新适应值为当前技能因子的代价

    # 计算标量适应度（适应度越高越好）
    fitness = np.array([1 / np.min(ind.MFRanks) for ind in population])
    top_indices = np.argsort(-fitness)[:num]  # 取前 num 个适应度最高的个体
    # 精英保留：选择前 num 个个体
    selected = [population[i] for i in top_indices]

    return selected
