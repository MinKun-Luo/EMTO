# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: A Meta-Knowledge Transfer-Based Differential Evolution for Multitask Optimization
# @Author: Jian-Yu Li; Zhi-Hui Zhan; Kay Chen Tan; Jun Zhang
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2022
# @Doi: 10.1109/TEVC.2021.3131236

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MKTDE论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class MKTDE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.6

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 50

        # 初始化种群
        population = Initialization(self, Prob, Individual)

        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                # 生成子代
                offspring = self.Generation(population, t)
                # 选择性评估
                self.Evaluation(offspring, Prob[t], t)
                # 精英保留策略
                population[t] = Selection_Elit(population[t], offspring, Problem.N)

            # EST
            for t in range(Problem.T):
                # 计算辅助种群精英个体在当前任务的适应度
                fitness = Prob[t].fnc(population[1 - t][0].rnvec)
                # 如果辅助种群精英个体的适应度优于当前任务的最差适应度，则将最差个体更新为辅助任务的精英个体
                if fitness < population[t][-1].obj:
                    population[t][-1].rnvec = population[1 - t][0].rnvec.copy()
                    population[t][-1].obj = fitness

        return self

    def Generation(self, population, t):
        """
        MKT进化产生子代

        :param population: 父种群。
        :param t: 任务索引。
        :return: 子种群。
        """
        # 初始化子代
        offspring = [Individual() for _ in range(Problem.N)]
        # 初始化辅助种群生成的转移目标种群
        assiTransPop = [Individual() for _ in range(len(population[1 - t]))]

        # MKT
        # 计算两个种群的质心
        ct = np.mean([ind.rnvec for ind in population[t]], axis=0)
        cs = np.mean([ind.rnvec for ind in population[1 - t]], axis=0)
        # 遍历辅助种群中的所有个体
        for idx, ind in enumerate(population[1 - t]):
            # 生成新个体
            assiTransPop[idx].rnvec = ind.rnvec - cs + ct
            # 归一化
            assiTransPop[idx].rnvec = np.clip(assiTransPop[idx].rnvec, 0, 1)
        # 将转移目标种群和原目标种群合并，得到融合种群
        mergedPop = population[t] + assiTransPop

        # 目标种群内进化
        for idx, ind in enumerate(population[t]):
            # 从当前种群中随机选取一个个体作为基向量
            r1 = np.random.choice([i for i in range(len(population[t])) if i != idx])
            # 从融合种群中随机选取两个个体作为差分向量
            fr1, fr2 = np.random.choice([i for i in range(len(mergedPop))], 2, replace=False)
            # 生成新个体
            offspring[idx].rnvec = DERand1(ind.rnvec, population[t][r1].rnvec, mergedPop[fr1].rnvec,
                                           mergedPop[fr2].rnvec, len(ind.rnvec), self.F, self.CR)

        return offspring


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
        result = MKTDE().run(Prob, True)
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