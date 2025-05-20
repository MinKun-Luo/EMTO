# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFDE修改为多种群框架

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual

from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MPDE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.rmp = 0.3  # 重组概率
        self.F = 0.5  # 缩放因子
        self.CR = 0.9  # 交叉概率

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
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
                population[t] = Selection_Elit(population[t], offspring, num=Problem.N)

        return self

    def Generation(self, population, t):
        """
        生成子代种群。

        :param population: list 父代种群，当前代的所有个体。
        :param t: int 任务索引。
        :return: list offspring 生成的子代个体列表。
        """
        # 子代
        offspring = [Individual() for _ in range(Problem.N)]
        # 进化
        for i in range(Problem.N):
            r1, r2, r3 = np.random.choice([k for k in range(Problem.N) if k != i], 3, replace=False)
            # 原向量
            p = population[t][i].rnvec
            # 基向量
            p1 = population[t][r1].rnvec
            if np.random.random() < self.rmp:
                # 知识转移，从另一个种群中选择两个不同个体，形成差异向量
                r2, r3 = np.random.choice((range(Problem.N)), 2, replace=False)
                p2, p3 = population[1 - t][r2].rnvec, population[1 - t][r3].rnvec
            else:
                # 从当前种群中选择两个不同个体，形成差异向量
                p2, p3 = population[t][r2].rnvec, population[t][r3].rnvec

            # DERand1
            offspring[i].rnvec = DERand1(p, p1, p2, p3, len(p), self.F, self.CR)

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
        result = MPDE().run(Prob, True)
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
