from Algorithms.Utils.Individual.Individual import Individual


class Individual_rankDE(Individual):
    def __init__(self):
        # 调用父类的构造函数，初始化多任务维度和任务数量
        Individual.__init__(self)
        self.rank = None
        self.p = None
