# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Multi-Task Optimization With Adaptive Intensity of Knowledge Transfer
# @Author: Xinyu Zhou; Neng Mei; Maosheng Zhong; Mingwen Wang
# @Journal: IEEE Transactions on Emerging Topics in Computational Intelligence
# @year: 2024
# @Doi: 10.1109/TETCI.2024.3418810

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: EMTO-AI论文复现

from collections import deque

from Algorithms.Algorithm import Algorithm
from Algorithms.MultiTask.MultiFactorial.EMTO_AI.Individual_AI import Individual_AI
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Algorithms.Utils.MultiFactorial.Selection_MF import Selection_MF
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class EMTO_AI(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.6  # CEC17交叉概率
        # self.CR = 0.9  # CEC22交叉概率
        self.g = 35  # 知识转移强度调整代数
        self.m = int(0.05 * 200 / 2)  # 知识存档的大小

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization_MF(self, Prob, Individual_AI)
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

            # 生成子代
            offspring = self.Generation(population, archive, TR)
            # 选择性评估
            for t in range(Problem.T):
                offspring_t = [ind for ind in offspring if ind.MFFactor == t + 1]
                self.Evaluation(offspring_t, Prob[t], t)
                for ind in offspring_t:
                    ind.MFCosts[t] = ind.obj
            # 精英保留策略
            population = Selection_MF(population, offspring, num=Problem.N)

            # 删除子种群和临时种群
            del offspring, offspring_t

        return self

    def Generation(self, population, archive, TR):
        """
        生成子代个体（包含知识转移机制的差分进化操作）。

        :param population: list 当前种群，包含多个任务的个体。
        :param archive: list 每个任务的知识存档，archive[t] 表示任务 t 的知识个体列表。
        :param TR: list 每个任务的知识转移强度（概率）。
        :return: list offspring 子代个体列表。
        """
        # 子种群
        offspring = [Individual_AI() for _ in range(len(population))]
        # 遍历种群进行差分进化操作
        for t in range(Problem.T):
            # 获取当前任务的子种群
            subPop = [idx for idx, ind in enumerate(population) if ind.MFFactor == t + 1]
            for idx in subPop:
                # 从当前任务随机选取3个不同的个体
                index = np.random.choice([i for i in subPop if i != idx], 3, replace=False)
                # 从知识存档中随机选取一个个体
                indexTran = np.random.randint(0, len(archive[1 - t]))
                # 获取父代个体
                p = population[idx]
                p2 = population[index[1]]
                p3 = population[index[2]]
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
                    p1 = population[index[0]]
                # 差分进化操作
                c.rnvec = DERand1(p.rnvec, p1.rnvec, p2.rnvec, p3.rnvec, len(p.rnvec), self.F, self.CR)
                # 继承父代的技能因子
                c.MFFactor = p.MFFactor

        return offspring

    def AITA(self, population, Prob, TR):
        """
        自适应调整任务间的知识转移强度（AIAT机制）。

        :param population: list 当前种群，包含两个任务的个体。
        :param Prob: list 多任务优化问题对象列表。
        :param TR: list 当前的知识转移强度列表。
        :return: list 更新后的知识转移强度 TR。
        """
        # 将种群按照技能因子分为两个子种群
        subPop = [[ind for ind in population if ind.MFFactor == 1], [ind for ind in population if ind.MFFactor == 2]]

        for t in range(Problem.T):
            # 在当前任务评估辅助种群的个体
            subTemp = [ind for ind in subPop[1 - t] if ind.MFCosts[t] == np.inf]
            if len(subTemp) != 0:
                # 在当前任务评估辅助种群的所有个体
                self.Evaluation(subTemp, Prob[t], t)
                for ind in subTemp:
                    # 更新辅助种群的适应度
                    ind.MFCosts[t] = ind.obj

            # 创建相关性矩阵
            RM = np.zeros((len(subPop[1 - t]), len(subPop[t])))
            # 更新相关性矩阵
            for i in range(len(subPop[1 - t])):
                # 计算辅助任务的因子代价
                value1 = subPop[1 - t][i].MFCosts[t]
                for j in range(len(subPop[t])):
                    # 计算当前任务的因子代价
                    value2 = subPop[t][j].MFCosts[t]
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
            ka = sorted([ind for ind in population if ind.isTran and ind.MFFactor == t + 1],
                        key=lambda ind: ind.MFCosts[t], reverse=False)
            n = len(ka)
            # 如果存档为空（表明没有待存档子代，即所有转移子代均被淘汰或为首次初始化），则直接添加当前任务m个最优个体
            if n == 0:
                ka = sorted([ind for ind in population if ind.MFFactor == t + 1],
                            key=lambda ind: ind.MFCosts[t], reverse=False)
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
        result = EMTO_AI().run(Prob, True)
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