# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: An empirical study of multifactorial PSO and multifactorial DE
# @Author: L. Feng; W. Zhou; L. Zhou; S. W. Jiang; J. H. Zhong; B. S. Da
# @Journal: 2017 IEEE Congress on Evolutionary Computation (CEC)
# @year: 2017
# @Doi: 10.1109/CEC.2017.7969407
# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/22 16:45
# @Author: wzb
# @Introduction: MFPSO论文复现

import copy

from Algorithms.Algorithm import Algorithm
from Algorithms.MultiTask.MultiFactorial.MFPSO.Individual_PSO import Individual_PSO
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MFPSO(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.rmp = 0.3  # 知识转移频率
        self.maxW = 0.9  # 最大惯性权重
        self.minW = 0.4  # 最小惯性权重
        self.c1 = 0.2  # 加速度系数
        self.c2 = 0.2  # 加速度系数
        self.c3 = 0.2  # 加速度系数

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization_MF(self, Prob, Individual_PSO)
        # 初始化个体的最优位置和最优值
        for ind in population:
            ind.pBestRnvec = copy.deepcopy(ind.rnvec)
            ind.pBestObj = ind.obj
            ind.v = np.zeros(int(max(p.dim for p in Prob)))

        while self.notTerminated(Prob, isPrint):
            # 计算当前权重
            W = self.maxW - (self.maxW - self.minW) * self.FE / Problem.maxFE
            # 生成子代
            population = self.Generation(population, W)
            # 临时种群
            population_temp = copy.deepcopy(population)
            # 选择性评估
            for t in range(Problem.T):
                self.Evaluation([ind for ind in population if ind.MFFactor == t + 1], Prob[t], t)

            # 更新个体的最优位置和最优值
            for i in range(len(population)):
                if population[i].obj < population_temp[i].pBestObj:
                    population[i].pBestRnvec = copy.deepcopy(population[i].rnvec)
                    population[i].pBestObj = population[i].obj

        return self

    def Generation(self, population, W):
        """
        生成子代种群。

        :param population: list 父代种群，当前代的所有个体。
        :param W: 权重系数
        :return: list population 更新的父代种群。
        """
        for ind in population:
            # 当前个体的位置
            p = ind.rnvec
            # 当前个体的历史最优位置
            pBest = ind.pBestRnvec
            # 当前任务的全局最优位置
            gBest = self.BestInd[ind.MFFactor - 1]
            # 随机选择一个不同任务的全局最优位置
            t = np.random.choice([i for i in range(Problem.T) if i != ind.MFFactor - 1])
            gOtherBest = self.BestInd[t]

            # 根据重组概率更新速度
            if np.random.rand() < self.rmp:
                ind.v = W * ind.v + self.c1 * np.random.rand() * (pBest - p) + self.c2 * np.random.rand() * (
                        gBest - p) + self.c3 * np.random.rand() * (gOtherBest - p)
            else:
                ind.v = W * ind.v + self.c1 * np.random.rand() * (pBest - p) + self.c2 * np.random.rand() * (gBest - p)

            # 更新个体位置
            ind.rnvec = ind.rnvec + ind.v
            # 限制位置在[0, 1]范围内
            ind.rnvec = np.clip(ind.rnvec, 0, 1)

        return population


def main():
    # 测试函数
    Prob = CI_HS()
    Prob = CI_MS()
    Prob = CI_LS()
    Prob = PI_HS()
    Prob = PI_MS()
    Prob = PI_LS()
    Prob = NI_HS()
    Prob = NI_MS()
    Prob = NI_LS()
    # Prob = Benchmark1()
    # Prob = Benchmark2()
    # Prob = Benchmark3()
    # Prob = Benchmark4()
    # Prob = Benchmark5()
    # Prob = Benchmark6()
    # Prob = Benchmark7()
    # Prob = Benchmark8()
    # Prob = Benchmark9()
    # Prob = Benchmark10()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = MFPSO().run(Prob, True)
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
