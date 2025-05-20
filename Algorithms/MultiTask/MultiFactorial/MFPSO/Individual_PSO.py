# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/22 16:45
# @Author: wzb
# @Introduction: MFPSO个体类

import numpy as np

from Algorithms.Utils.Individual.Individual_MF import Individual_MF


class Individual_PSO(Individual_MF):
    def __init__(self):
        # 调用父类的构造函数，初始化多任务维度和任务数量
        Individual_MF.__init__(self)
        # 个体速度
        self.v = np.empty(0, dtype=float)
        # 个体历史最优位置
        self.pBestRnvec = np.empty(0, dtype=float)
        # 个体历史最优适应度
        self.pBestObj = None


