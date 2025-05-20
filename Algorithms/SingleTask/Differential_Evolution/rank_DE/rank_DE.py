# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Differential Evolution With Ranking-Based Mutation Operators
# @Author: Gong, Wenyin and Cai, Zhihu
# @Journal: IEEE Transactions on Cybernetics
# @year: 2013
# @Doi: 10.1109/TCYB.2013.2239988

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: rank_DE论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.SingleTask.Differential_Evolution.rank_DE.Individual_rankDE import Individual_rankDE
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Tournament
from Algorithms.Utils.Operator.DE_operator.DE import *
from Problems.SingleTask.Classical_Function.Classical_Function import *


class rank_DE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.9

    def run(self, Prob, isPrint=False):
        # 初始化种群
        population = Initialization(self, Prob, Individual_rankDE, False)

        while self.notTerminated(Prob, isPrint):
            # 选择性评估
            for t in range(Problem.T):
                # 生成子代
                offspring = self.Generation(population[t])
                # 评估子代
                self.Evaluation(offspring, Prob[t], t)
                # 锦标赛选择子代
                population[t], replace = Selection_Tournament(population[t], offspring)

        return self

    def Generation(self, population):
        """
        生成子代种群。

        :param population: 父代种群，当前代的所有个体。
        :return: 生成的子代个体列表。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspring = [Individual_rankDE() for _ in range(N)]

        # 计算排名和选择概率
        ranks = np.argsort([ind.obj for ind in population])
        for idx, r in enumerate(ranks):
            population[r].rank = Problem.N - idx
            population[r].p = population[r].rank / Problem.N

        # 遍历种群进行差分进化操作
        for i in range(N):
            # 获取父代个体
            p = population[i]
            # 初始化子代个体
            c = offspring[i]

            # 根据排名选择DE个体索引
            # 基向量（优先选择）
            r1 = np.random.randint(0, N)
            while np.random.rand() > population[r1].p or r1 == i:
                r1 = np.random.randint(0, N)
            # 终端向量
            r2 = np.random.randint(0, N)
            while np.random.rand() > population[r2].p or r2 == i or r2 == r1:
                r2 = np.random.randint(0, N)
            # 随机向量
            r3 = np.random.randint(0, N)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = np.random.randint(0, N)

            # 进行差分进化操作
            c.rnvec = DERand1(p.rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                              len(p.rnvec),
                              self.F, self.CR)

        return offspring


def main():
    # 任务集，选择不同的任务集进行实验
    Prob = Func_Ackley()
    Prob = Func_Elliptic()
    # Prob = Func_Griewank()
    # Prob = Func_Rastrigin()
    # Prob = Func_Rosenbrock()
    # Prob = Func_Schwefel()
    # Prob = Func_Schwefel2()
    # Prob = Func_Sphere()
    # Prob = Func_Weierstrass()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = rank_DE().run(Prob, True)
        costs[i] = result.Best
        print(f'Repetition {i} :', result.Best, '\n')

        print(f'Values of the previous {i + 1} generations:')
        for j in range(i + 1):
            print(*costs[j])

        print(f'Average Values of the previous {i + 1} generations:')
        print(np.mean(costs[:i + 1], axis=0))
        print()


if __name__ == "__main__":
    main()
