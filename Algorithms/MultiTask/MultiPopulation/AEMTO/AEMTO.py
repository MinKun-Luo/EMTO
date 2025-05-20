# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Multitask Optimization With Adaptive Knowledge Transfer
# @Author: Hao Xu; A. K. Qin; Siyu Xia
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2022
# @Doi: 10.1109/TEVC.2021.3107435

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: AEMTO论文复现


import collections
import copy

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.Crossover import BinomialCrossover
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Algorithms.Utils.Operator.Selection import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class AEMTO(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.9
        self.alpha = 0.3  # 滑动平均系数
        self.Ptsf = [0.05, 0.7]  # 知识转移频率上下界
        self.Pmin = 0.3 / (Problem.T - 1)  # 源知识被选择的最小概率

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization(self, Prob, Individual)
        # 初始化源任务（T-1个，不包含自身）对每个目标任务（T个）的知识转移强度（外层列表为每个目标任务，内层列表为源任务对当前目标任务）
        q_sel = [[0 for _ in range(Problem.T)] for _ in range(Problem.T)]
        # 初始化源任务（T-1个，不包含自身）对每个目标任务（T个）的被选择进行知识转移的概率（外层列表为每个目标任务，内层列表为源任务对当前目标任务）
        p_sel = [[1 / (Problem.T - 1) for _ in range(Problem.T)] for _ in range(Problem.T)]
        # 初始化每个任务的interKT（知识转移）的概率
        qo = [0 for _ in range(Problem.T)]
        # 初始化每个任务的intraSE（自我进化）的概率
        qs = [0 for _ in range(Problem.T)]
        # 初始化每个任务的知识转移概率（rmp）
        p_tsf = [np.mean(self.Ptsf) for _ in range(Problem.T)]

        while self.notTerminated(Prob, isPrint):
            # 进化
            for t in range(Problem.T):
                if np.random.rand() <= p_tsf[t]:
                    # 知识转移（interKT）
                    population, reward = self.interKT(population, Prob, q_sel[t], p_sel[t], t)
                    # 更新知识转移（interKT）的概率
                    qo[t] = self.alpha * qo[t] + (1 - self.alpha) * reward
                else:
                    # 自我进化（intraSE）
                    population[t], reward = self.intraSE(population[t], Prob, t)
                    # 更新自我进化（intraSE）的概率
                    qs[t] = self.alpha * qs[t] + (1 - self.alpha) * reward
                # 更新知识转移概率
                p_tsf[t] = self.Ptsf[0] + (self.Ptsf[1] - self.Ptsf[0]) * (qo[t] / (qo[t] + qs[t] + 1e-100))

        # 返回最优适应度
        return self

    def intraSE(self, population, Prob, t):
        """
        种群自我进化。

        :param population: 父种群。
        :param Prob: 问题对象。
        :param t: 任务索引。
        :return: 更新后的下一代种群和奖励。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspring = [Individual() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for n in range(N):
            # 初始化子代个体
            c = offspring[n]
            r1, r2, r3 = np.random.choice([i for i in range(N) if i != n], 3, replace=False)
            # 进行差分进化操作
            c.rnvec = DERand1(population[n].rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                              len(population[n].rnvec),
                              self.F, self.CR)
        # 评估子代
        self.Evaluation(offspring, Prob[t], t)
        # 精英保留策略
        population = Selection_Elit(population, offspring, Problem.N)

        # 子代存活数量
        surviveNum = 0
        # 计算奖励（子代的存活率）
        for ind in population:
            if ind in offspring:
                surviveNum += 1
        reward = surviveNum / Problem.N

        return population, reward

    def interKT(self, population, Prob, q_sel, p_sel, i):
        """
        种群知识转移。

        :param population: 父种群。
        :param Prob: 问题对象。
        :param q_sel: 知识转移强度。
        :param p_sel: 源种群被选择概率。
        :param i: 任务索引。
        :return: 更新后的下一代种群和奖励。
        """
        # 不选择目标任务
        p_sel[i] = 0
        # 基于p_sel采用SUS选取每个源任务的候选解数量
        n_temp = StochasticUniversalSampling(p_sel, Problem.N)
        # 统计各索引频次
        counter = collections.Counter(n_temp)
        # 构建结果列表（各源任务的候选解数量）
        n = [counter.get(f, 0) for f in range(max(n_temp) + 1)]

        # 知识池
        K = []
        # 选取每个源任务的候选个体（构建知识池）
        for j in range(Problem.T):
            if j != i and n[j] != 0:
                fitness = [1 / ind.obj for ind in population[j]]
                select_index = RouletteWheelSelection(fitness, n[j])
                K.extend([population[j][k] for k in select_index])

        # 各源种群的成功转移的候选解个数
        ns = [0 for _ in range(Problem.T)]
        # 计数器
        k = 0

        # 种群进化（所有源任务辅助目标任务）
        for j in range(Problem.T):
            if j != i and n[j] != 0:
                for s in range(n[j]):
                    # 从知识池取出相应个体
                    v = K[k]
                    # 从目标任务的种群中选择对应个体
                    x = population[i][k]
                    # 构造子代
                    offspring = Individual()
                    # 二项式交叉产生子代
                    c1, c2 = BinomialCrossover(v.rnvec, x.rnvec, len(x.rnvec))
                    offspring.rnvec = c1
                    # 在目标任务评价子代
                    self.Evaluation([offspring], Prob[i], i)
                    # 若适应度变好，则替换
                    if offspring.obj < x.obj:
                        # 替换（种群更新）
                        population[i][k] = copy.deepcopy(offspring)
                        # 转移成功的候选解加1
                        ns[j] += 1
                    k += 1

        # 更新知识转移强度
        for j in range(Problem.T):
            if j != i and n[j] != 0:
                # 源任务j对目标任务i的知识转移强度（替换成功率）
                q_sel[j] = self.alpha * q_sel[j] + (1 - self.alpha) * (ns[j] / n[j])

        # 更新源种群被选择概率
        for j in range(Problem.T):
            if j != i:
                # 源种群j被选择的概率（帮助率）
                p_sel[j] = self.Pmin + (1 - (Problem.T - 1) * self.Pmin) * (q_sel[j] / (np.sum(q_sel) + 1e-100))

        # 计算奖励(转移成功率)
        reward = np.sum(ns) / Problem.N

        # 返回更新后的种群和奖励
        return population, reward


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
        result = AEMTO().run(Prob, True)
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