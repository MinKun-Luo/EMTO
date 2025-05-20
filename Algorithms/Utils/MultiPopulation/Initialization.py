# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 初始化多任务进化算法的种群（支持维度填充）
# @Remind: 若任务维度不一致建议开启 isPadding 以统一染色体长度

import numpy as np

from Problems.Problem import Problem


def Initialization(Algo, Prob, Individual_Class, isPadding=True):
    """
    生成初始种群

    :param Algo: 算法实例
    :param Prob: 问题列表（每个任务一个问题实例）
    :param Individual_Class: 个体类，用于实例化个体
    :param isPadding: bool，是否统一染色体维度（默认 True）
    :return: population: list，每个任务的初始种群（二维列表）
    """
    # 初始化 T 个子种群（每个任务一个子种群）
    population = [[] for _ in range(Problem.T)]
    # 获取所有任务的最大维度
    max_dim = max(p.dim for p in Prob)

    # 逐任务生成个体
    for t in range(Problem.T):
        task_dim = Prob[t].dim  # 当前任务的维度
        for _ in range(Problem.N):
            individual = Individual_Class()
            # 染色体长度：统一为最大维度或保持原任务维度
            dim = max_dim if isPadding else task_dim
            individual.rnvec = np.random.rand(dim)
            population[t].append(individual)

        # 当前任务的种群适应度评估
        Algo.Evaluation(population[t], Prob[t], t)

    return population
