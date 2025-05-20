# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: EMTO-AI个体类

from Algorithms.Utils.Individual.Individual_MF import Individual_MF


class Individual_AI(Individual_MF):
    def __init__(self):
        # 调用父类的构造函数，初始化多任务维度和任务数量
        Individual_MF.__init__(self)
        # 是否为转移子代，初始化为False
        self.isTran = False
