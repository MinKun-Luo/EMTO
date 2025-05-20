# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 初始化多因子算法中的种群个体，设置基因型、适应度、排名和技能因子
# @Remind: 初始化个体维度基于所有任务的最大维度，技能因子索引从1开始

import numpy as np

from Problems.Problem import Problem


def Initialization_MF(Algo, Prob, Individual_Class):
    """
    生成初始种群

    :param Algo: 算法实例
    :param Prob: 问题实例列表（每个任务一个）
    :param Individual_Class: 个体类，用于生成个体对象
    :return: population: list，初始化后的种群个体列表
    """
    population = []

    # 随机初始化种群个体基因型
    for i in range(Problem.N):
        individual = Individual_Class()
        # 计算所有任务中最大的维度，用于统一个体长度
        D = max(p.dim for p in Prob)
        individual.rnvec = np.random.rand(D)
        population.append(individual)

    # 评估种群个体（得到因子成本）
    for t in range(Problem.T):
        # 评估种群（更新个体适应值）
        Algo.Evaluation(population, Prob[t], t)
        # 更新个体的因子代价
        for ind in population:
            ind.MFCosts[t] = ind.obj

    # 计算每个任务的个体排名，排名越小越优秀
    for t in range(Problem.T):
        costs = np.array([ind.MFCosts[t] for ind in population])
        rank = np.argsort(costs)
        for i in range(len(population)):
            population[rank[i]].MFRanks[t] = i + 1  # 排名从1开始

    # 确定每个个体的技能因子（最小排名对应任务，若多个随机选）
    for ind in population:
        min_rank = min(ind.MFRanks)
        min_tasks = np.where(ind.MFRanks == min_rank)[0]
        ind.MFFactor = np.random.choice(min_tasks) + 1  # 索引从1开始
        ind.obj = ind.MFCosts[ind.MFFactor - 1]  # 更新适应值为当前技能因子的代价

    return population
