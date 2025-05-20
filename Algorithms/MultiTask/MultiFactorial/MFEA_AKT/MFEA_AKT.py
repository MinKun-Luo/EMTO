# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Toward Adaptive Knowledge Transfer in Multifactorial Evolutionary Computation
# @Author: Lei Zhou; Liang Feng; Kay Chen Tan; Jinghui Zhong; Zexuan Zhu; Kai Liu
# @Journal: IEEE Transactions on Cybernetics
# @year: 2021
# @Doi: 10.1109/TCYB.2020.2974100

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFEA-AKT论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.MultiTask.MultiFactorial.MFEA_AKT.Individual_AKT import Individual_AKT
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Algorithms.Utils.MultiFactorial.Selection_MF import Selection_MF
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MFEA_AKT(Algorithm):
    def __init__(self):
        super().__init__()
        self.rmp = 0.3
        # 自适应转移交叉指标的间隔代数
        self.Gap = 20
        # 记录每代使用的最佳交叉因子
        self.cfb_record = []
        self.eta = 2
        self.mu = 5

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization_MF(self, Prob, Individual_AKT)

        while self.notTerminated(Prob, isPrint):
            # 生成子代
            offspring = self.Generation(population)
            # 选择性评估
            for t in range(Problem.T):
                offspring_t = [ind for ind in offspring if ind.MFFactor == t + 1]
                self.Evaluation(offspring_t, Prob[t], t)
                for ind in offspring_t:
                    ind.MFCosts[t] = ind.obj
            # 自适应转移交叉指标
            offspring = self.adaptiveTci(population, offspring)
            # 精英保留策略
            population = Selection_MF(population, offspring, num=Problem.N)
            del offspring, offspring_t

        return self

    def Generation(self, population):
        """
        生成子代种群。

        :param population: 父代种群，当前代的所有个体。
        :return: 生成的子代个体列表。
        """
        N = len(population)
        # 随机排列种群索引
        inorder = np.random.permutation(len(population))
        offspring = [Individual_AKT() for _ in range(len(population))]
        # 遍历种群的一半进行交叉操作
        for i in range(int(N / 2)):
            # 选择两个父代个体索引
            p1_index, p2_index = int(inorder[i]), int(inorder[i + int(N / 2)])
            p1 = population[p1_index]
            p2 = population[p2_index]
            # 初始化两个子代个体
            c1 = offspring[i]
            c2 = offspring[i + int(N / 2)]
            # 遗传机制
            if p1.MFFactor == p2.MFFactor or np.random.rand() < self.rmp:
                if p1.MFFactor == p2.MFFactor:
                    # 如果技能因子相同或满足转移条件，则选择两个个体进行模拟二进制交叉和突变
                    c1.rnvec, c2.rnvec = SBX(p1.rnvec, p2.rnvec, self.eta)
                    # 子代继承父代的转移交叉指标
                    c1.CXFactor, c2.CXFactor = p1.CXFactor, p2.CXFactor
                    c1.MFFactor, c2.MFFactor = p1.MFFactor, p2.MFFactor
                else:
                    # 如果技能因子不同，满足随机交配概率，同样进行知识转移，产生后代为转移子代
                    CXF = p1.CXFactor if np.random.rand() < 0.5 else p2.CXFactor
                    # 选择与转移交叉指标相关的交叉算子进行知识转移
                    c1.rnvec, c2.rnvec = self.crossover(p1.rnvec, p2.rnvec, CXF)
                    # 将后代c1和c2的转移交叉指标设置为CXFactor
                    c1.CXFactor, c2.CXFactor = CXF, CXF
                    # 设置后代c1和c2为转移子代
                    c1.isTran, c2.isTran = True, True
                    # 随机分配转移子代c1和c2的技能因子和直系父代
                    if np.random.rand() < 0.5:
                        c1.MFFactor, c1.parNum = p1.MFFactor, p1_index
                    else:
                        c1.MFFactor, c1.parNum = p2.MFFactor, p2_index
                    if np.random.rand() < 0.5:
                        c2.MFFactor, c2.parNum = p1.MFFactor, p1_index
                    else:
                        c2.MFFactor, c2.parNum = p2.MFFactor, p2_index
            else:
                # 轻微变异，并继承父代的技能因子和转移交叉指标
                c1.rnvec = PM2(p1.rnvec, self.mu)
                c1.MFFactor, c1.CXFactor = p1.MFFactor, p1.CXFactor
                c2.rnvec = PM2(p2.rnvec, self.mu)
                c2.MFFactor, c2.CXFactor = p2.MFFactor, p2.CXFactor

        return offspring

    def adaptiveTci(self, population, offspring):
        """
        自适应更新转移交叉指标（Transfer Crossover Indicator）。

        :param population: 父代种群。
        :param offspring: 子代种群（部分个体可能为转移子代）。
        :return: 更新转移交叉指标后的子代种群。
        """
        # 记录每个交叉因子的最大改进率
        imp_num = np.zeros(6)
        # 筛选出转移子代
        tranPop = [ind for ind in offspring if ind.isTran]
        # 遍历转移子代
        for ind in tranPop:
            # 获取直系父代的因子成本
            pCost = population[ind.parNum].MFCosts[population[ind.parNum].MFFactor - 1]
            # 获取转移子代的因子成本
            cCost = ind.MFCosts[ind.MFFactor - 1]
            if (pCost - cCost) / pCost > imp_num[ind.CXFactor]:
                imp_num[ind.CXFactor] = (pCost - cCost) / pCost
        # 存在改进的交叉因子
        if np.any(imp_num):
            # 找到最大改进率的交叉因子
            max_idx = int(np.argmax(imp_num))
        # 不存在改进的交叉因子
        else:
            # 计算最近Gap代的转移交叉指标
            if self.Gen <= self.Gap:
                record_tmp = self.cfb_record[:self.Gen - 1]
            else:
                record_tmp = self.cfb_record[self.Gen - self.Gap - 1:self.Gen]
            if len(record_tmp) == 0:
                max_idx = np.random.randint(0, 6)
            else:
                max_idx = np.bincount(record_tmp).argmax()
        # 记录当代使用的最佳交叉因子
        self.cfb_record.append(max_idx)

        # 更新所有后代的转移交叉指标
        for ind in offspring:
            # 如果个体是转移子代
            if ind.isTran:
                # 如果转移子代的适应度优于其直系父代
                if ind.MFCosts[ind.MFFactor - 1] > population[ind.parNum].MFCosts[population[ind.parNum].MFFactor - 1]:
                    # 将转移子代的交叉因子设置为最佳交叉因子
                    ind.CXFactor = max_idx
            else:
                # 如果个体不是转移子代，随机决定是否使用最佳交叉因子
                ind.CXFactor = max_idx if np.random.rand() < 0.5 else np.random.randint(0, 6)

        return offspring

    @staticmethod
    def crossover(p1, p2, CXFactor):
        """
        根据交叉因子选择对应的交叉算子进行操作。

        :param p1: 父代个体1（向量形式）。
        :param p2: 父代个体2（向量形式）。
        :param CXFactor: 整数，表示使用的交叉算子编号（0~5）。
        :return: c1, c2: 两个子代向量（经过对应交叉操作生成）。
        """
        D = np.max([len(p1), len(p2)])
        if CXFactor == 0:
            return SBX(p1, p2)  # 模拟二进制交叉
        elif CXFactor == 1:
            return TwoPointCrossover(p1, p2, D)  # 两点交叉
        elif CXFactor == 2:
            return ArithmeticalCrossover(p1, p2)  # 算术交叉
        elif CXFactor == 3:
            return UniformCrossover(p1, p2, D)  # 均匀交叉
        elif CXFactor == 4:
            return GeometricalCrossover(p1, p2)  # 几何交叉
        elif CXFactor == 5:
            return BLXalpha(p1, p2, D)  # BLX-alpha交叉
        else:
            print('Invalid crossover factor!')  # 无效的交叉因子
            exit(1)


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
        result = MFEA_AKT().run(Prob, True)
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