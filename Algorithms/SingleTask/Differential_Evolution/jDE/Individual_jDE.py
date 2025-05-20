import numpy as np

from Algorithms.Utils.Individual.Individual import Individual


class Individual_jDE(Individual):
    def __init__(self):
        # 调用父类的构造函数，初始化多任务维度和任务数量
        Individual.__init__(self)
        self.F = np.random.rand() * 0.9 + 0.1
        self.CR = np.random.rand()
