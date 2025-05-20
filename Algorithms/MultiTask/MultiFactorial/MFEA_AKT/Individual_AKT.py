# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFEA-AKT个体类

import numpy as np

from Algorithms.Utils.Individual.Individual_MF import Individual_MF


class Individual_AKT(Individual_MF):
    def __init__(self):
        # 调用父类的构造函数，初始化多任务维度和任务数量
        Individual_MF.__init__(self)
        # 交叉因子，初始化为0
        self.CXFactor = np.random.randint(0, 6)
        # 是否为转移子代，初始化为False
        self.isTran = False
        # 直系父代索引，初始化为-1
        self.parNum = -1
