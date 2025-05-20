# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 所有测试集的基类

class Problem:
    # 默认每个任务的种群大小
    defaultN = 50
    # 默认任务数量
    defaultT = 2
    # 默认最大评估次数
    defaultMaxFE = 100000

    # 类属性，所有类对象共用
    T = defaultT  # 任务数量
    N = defaultN  # 每个任务的种群大小（适用于多种群）
    maxFE = defaultMaxFE  # 最大评估次数（所有任务）

    # 实例属性，每个类对象独有
    def __init__(self, dim, lb, ub):
        self.dim = dim  # 维度
        self.lb = lb  # 下界
        self.ub = ub  # 上界

    def encode(self, vec):
        """
        编码，将真实值转换为0-1之间的值

        :param vec: 输入的真实值向量
        :return: 转换后的0-1之间的值
        """
        return (vec - self.lb) / (self.ub - self.lb)

    def decode(self, rnvec):
        """
        解码，将0-1之间的值转换为真实值

        :param rnvec: 输入的0-1之间的值向量
        :return: 转换后的真实值
        """
        return self.lb + rnvec * (self.ub - self.lb)
