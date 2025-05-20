# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Block-Level Knowledge Transfer for Evolutionary Multitask Optimization
# @Author: Yi Jiang; Zhi-Hui Zhan; Kay Chen Tan; Jun Zhang
# @Journal: IEEE Transactions on Cybernetics
# @year: 2023
# @Doi: 10.1109/TCYB.2023.3273625

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: BLKT-DE论文复现

from sklearn.cluster import KMeans

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class BLKT_DE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5
        self.CR = 0.7

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization(self, Prob, Individual, False)
        minB = 1
        maxB = np.min([p.dim for p in Prob])
        # 随机初始化块大小
        divB = np.random.randint(minB, maxB)
        minK = 2
        maxK = int(Problem.N / 2)
        # 随机初始化聚类簇大小
        divK = np.random.randint(minK, maxK)

        while self.notTerminated(Prob, isPrint):
            # 块种群
            blockPop = []
            # 生成块种群
            for t in range(Problem.T):
                for i in range(Problem.N):
                    # 将个体的基因分块
                    for j in range(int(np.ceil(Prob[t].dim / divB))):
                        block = population[t][i].rnvec[j * divB: (j + 1) * divB]
                        # 如果块大小不足，则填充0
                        if len(block) < divB:
                            block = np.pad(block, (0, divB - len(block)), 'constant')
                        blockPop.append(block)

            # K-means聚类
            estimator = KMeans(n_clusters=divK)  # 构造聚类器
            estimator.fit(blockPop)  # 聚类
            label_pred = estimator.labels_  # 获取聚类标签

            # 子代块种群
            offspringBlock = [[] for _ in range(len(blockPop))]
            for i in range(divK):
                # 获取属于第i个簇的个体索引
                subPop = [idx for idx, label in enumerate(label_pred) if label == i]
                # 生成子代(簇内差分进化，生成的子代放在其父代的对应位置)
                self.BLKTDE(offspringBlock, subPop, blockPop, divB)

            # 重组BLKT-DE产生的子种群
            offspringBLKT = [[] for _ in range(Problem.T)]
            k = 0
            for t in range(Problem.T):
                for i in range(Problem.N):
                    # 拼接
                    offspring_temp = np.concatenate(np.array(offspringBlock[k:k + int(np.ceil(Prob[t].dim / divB))]))
                    # 保留个体的相应维度
                    offspring_one = Individual()
                    offspring_one.rnvec = offspring_temp[:Prob[t].dim]
                    offspringBLKT[t].append(offspring_one)
                    k = k + int(np.ceil(Prob[t].dim / divB))

            # 随机筛选DE和BLKT的子代（n个个体（最终变异子代））
            new_offspring = []
            # 任务改进标志
            succ_flag = [None] * Problem.T
            for t in range(Problem.T):
                # DE进化产生的子代
                offspringDE = self.DE(population[t])
                # 随机筛选DE和BLKT的子代
                combined_offspring = offspringBLKT[t] + offspringDE
                # 随机排列索引
                inorder = np.random.permutation(len(combined_offspring))
                new_offspring.append([combined_offspring[i] for i in inorder[:Problem.N]])
                # 适应值评估
                succ_flag[t] = self.Evaluation(new_offspring[t], Prob[t], t)
                # 精英保留策略
                population[t] = Selection_Elit(population[t], new_offspring[t], Problem.N)

            # FAS
            if all(not flag for flag in succ_flag):
                divB = np.random.randint(minB, maxB)
                divK = np.random.randint(minK, maxK)
            else:
                divB = np.random.randint(max(minB, divB - 1), min(divB + 1, maxB))
                divK = np.random.randint(max(minK, divK - 1), min(divK + 1, maxK))

        # 返回最优适应度
        return self

    def BLKTDE(self, offspringBlock, subPop, blockPop, divB):
        """
        块种群差分进化操作（BLKT）。

        :param offspringBlock: 子代块种群。
        :param subPop: 同一簇内个体索引。
        :param blockPop: 块种群。
        :param divB: 块大小。
        :return: numpy 数组，BLKT进化产生的子代块种群。
        """
        for idx in subPop:
            if len(subPop) < 4:
                offspringBlock[idx] = blockPop[idx]
                continue
            r1, r2, r3 = np.random.choice([i for i in subPop if i != idx], 3, replace=False)
            # 进行差分进化操作
            offspringBlock[idx] = DERand1(blockPop[idx], blockPop[r1], blockPop[r2], blockPop[r3], divB, self.F,
                                          self.CR)

    def DE(self, population):
        """
        子种群差分进化操作（DE/Rand/1）。

        :param population: 父种群。
        :return: numpy 数组，DE/Rand/1进化产生的子种群。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspringDE = [Individual() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for n in range(N):
            # 初始化子代个体
            c = offspringDE[n]
            r1, r2, r3 = np.random.choice([i for i in range(N) if i != n], 3, replace=False)
            # 进行差分进化操作
            c.rnvec = DERand1(population[n].rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                              len(population[n].rnvec),
                              self.F, self.CR)

        return offspringDE


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
        result = BLKT_DE().run(Prob, True)
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
