# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: EMTO-AI的多种群框架

import copy
from collections import deque

from Algorithms.Algorithm import Algorithm
from Algorithms.MultiTask.MultiFactorial.EMTO_AI.Individual_AI import Individual_AI
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class EMTO_AI_P(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.6  # CEC17交叉概率
        # self.CR = 0.9  # CEC22交叉概率
        self.g = 35  # 知识转移强度调整代数
        self.m = int(0.05 * 100 / 2)  # 知识存档的大小

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 50

        # 初始化种群
        population = Initialization(self, Prob, Individual_AI)
        # 知识存档
        archive = [deque(maxlen=self.m) for _ in range(2)]
        # 知识转移强度
        TR = np.zeros(Problem.T, dtype=float)

        while self.notTerminated(Prob, isPrint):
            # 更新知识存档
            self.updateArchives(population, archive)

            # 更新知识转移强度
            if self.Gen % self.g == 0 or self.Gen == 1:
                self.AITA(population, Prob, TR)

            # 生成子代并更新种群
            population = self.Generation(population, archive, Prob, TR)

        return self

    def Generation(self, population, archive, Prob, TR):
        """
        生成子代个体（包含知识转移机制的差分进化操作）。

        :param population: list 当前种群，包含多个任务的个体。
        :param archive: list 每个任务的知识存档，archive[t] 表示任务 t 的知识个体列表。
        :param Prob: 问题对象
        :param TR: list 每个任务的知识转移强度（概率）。
        :return: list offspring 子代个体列表。
        """
        # 遍历种群进行差分进化操作
        for t in range(Problem.T):
            # 子种群
            offspring = [Individual_AI() for _ in range(len(population[t]))]
            for idx in range(len(population[t])):
                # 从当前任务随机选取3个不同的个体
                index = np.random.choice([i for i in range(Problem.N) if i != idx], 3, replace=False)
                # 从知识存档中随机选取一个个体
                indexTran = np.random.randint(0, len(archive[1 - t]))
                # 获取父代个体
                p = population[t][idx]
                p2 = population[t][index[1]]
                p3 = population[t][index[2]]
                # 初始化子代个体
                c = offspring[idx]
                # 是否进行知识转移（即是否利用存档）
                if np.random.rand() < TR[t]:
                    # 从辅助任务的知识存档中选取一个个体
                    p1 = archive[1 - t][indexTran]
                    # 设置为转移子代
                    c.isTran = True
                else:
                    # 从当前任务的种群中选取一个个体
                    p1 = population[t][index[0]]
                # 差分进化操作
                c.rnvec = DERand1(p.rnvec, p1.rnvec, p2.rnvec, p3.rnvec, len(p.rnvec), self.F, self.CR)

            # 评估操作
            self.Evaluation(offspring, Prob[t], t)
            # 选择操作
            population[t] = Selection_Elit(population[t], offspring, Problem.N)

        return population

    def AITA(self, population, Prob, TR):
        """
        自适应调整任务间的知识转移强度（AIAT机制）。

        :param population: list 当前种群，包含两个任务的个体。
        :param Prob: list 多任务优化问题对象列表。
        :param TR: list 当前的知识转移强度列表。
        :return: list 更新后的知识转移强度 TR。
        """
        # 复制种群
        pop_temp = copy.deepcopy(population)
        for t in range(Problem.T):
            # 在当前任务评估辅助种群的个体
            self.Evaluation(pop_temp[1 - t], Prob[t], t)

            # 创建相关性矩阵
            RM = np.zeros((len(population[1 - t]), len(population[t])))
            # 更新相关性矩阵
            for i in range(len(population[1 - t])):
                # 计算辅助任务的因子代价
                value1 = pop_temp[1 - t][i].obj
                for j in range(len(population[t])):
                    # 计算当前任务的因子代价
                    value2 = population[t][j].obj
                    # 计算相关性矩阵
                    RM[i, j] = np.sign(value2 - value1)

            # 计算转移相关强度（矩阵中非负数的占比）
            TR[t] = np.sum(RM >= 0) / (RM.shape[0] * RM.shape[1])

        return TR

    def updateArchives(self, population, archive):
        """
        更新各任务的知识存档。

        :param population: list 当前种群，包含所有个体。
        :param archive: list 当前的知识存档，每个任务对应一个列表。
        :return: list 更新后的知识存档。
        """
        for t in range(Problem.T):
            # 当前任务的存档个体（存活下来的转移子代，并按其适应度排序）
            ka = sorted([ind for ind in population[t] if ind.isTran], key=lambda ind: ind.obj, reverse=False)
            n = len(ka)
            # 如果存档为空（表明没有待存档子代，即所有转移子代均被淘汰或为首次初始化），则直接添加当前任务m个最优个体
            if n == 0:
                ka = sorted([ind for ind in population[t]], key=lambda ind: ind.obj, reverse=False)
                archive[t].extend(ka[:self.m])
            # 如果存档个数小于m，则以“先进先出”的形式（即淘汰部分已有个体）存入
            elif n < self.m:
                archive[t].extend(ka)
            # 如果存档个数大于等于m，则清空存档，选取前m个最优待存个体存入
            else:
                archive[t].extend(ka[:self.m])

        return archive


def main():
    # 测试函数
    Prob = CI_HS()
    # Prob = CI_MS()
    # Prob = CI_LS()
    # Prob = PI_HS()
    # Prob = PI_MS()
    # Prob = PI_LS()
    # Prob = NI_HS()
    # Prob = NI_MS()
    # Prob = NI_LS()
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
        result = EMTO_AI_P().run(Prob, True)
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