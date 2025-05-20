# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 多因子个体基类，继承自Individual，用于管理多因子优化问题中的个体

import numpy as np
from Algorithms.Utils.Individual.Individual import Individual
from Problems.Problem import Problem


class Individual_MF(Individual):
    def __init__(self):
        """
        初始化多因子个体。

        :param self: 初始化方法的实例自身（隐式参数）
        """
        super().__init__()
        self.MFCosts = np.full(Problem.T, np.inf)  # 因子代价，初始化为无穷大
        self.MFRanks = np.array([None] * Problem.T)  # 因子排名，初始化为None
        self.MFFactor = None  # 技能因子
