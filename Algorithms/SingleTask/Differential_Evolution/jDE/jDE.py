# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems
# @Author: Brest, Janez and Greiner, Sao and Boskovic, Borko and Mernik, Marjan and Zumer, Viljem
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2006
# @Doi: 10.1109/TEVC.2006.872133

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: jDE论文复现

import copy

from Algorithms.Algorithm import Algorithm
from Algorithms.SingleTask.Differential_Evolution.jDE.Individual_jDE import Individual_jDE
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.Operator.DE_operator.DE import *
from Problems.SingleTask.Classical_Function.Classical_Function import *


class jDE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.tau1 = 0.1  # F修改的可能性
        self.tau2 = 0.1  # CR修改的可能性

    def run(self, Prob, isPrint=False):
        # 初始化种群
        population = Initialization(self, Prob, Individual_jDE, False)

        while self.notTerminated(Prob, isPrint):
            # 选择性评估
            for t in range(Problem.T):
                # 生成子代
                self.Generation(population[t], Prob[t], t)

        return self

    def Generation(self, population, Prob, t):
        """
        生成子代种群。

        :param population: 父代种群，当前代的所有个体。
        :param Prob: 问题对象。
        :param t: 任务索引。
        :return: 生成的子代个体列表。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspring = [Individual_jDE() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for i in range(N):
            # 自适应F和CR
            if np.random.rand() >= self.tau1:
                offspring[i].F = copy.deepcopy(population[i].F)
            if np.random.rand() >= self.tau2:
                offspring[i].CR = copy.deepcopy(population[i].CR)

            # 获取父代个体
            p = population[i]
            # 初始化子代个体
            c = offspring[i]
            # 选择三个不同的个体
            x = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            # 进行差分进化操作
            c.rnvec = DERand1(p.rnvec, population[x[0]].rnvec, population[x[1]].rnvec, population[x[2]].rnvec,
                              len(p.rnvec),
                              offspring[i].F, offspring[i].CR)
            # 锦标赛选择
            self.Evaluation([offspring[i]], Prob, t)
            if offspring[i].obj < p.obj:
                population[i] = copy.deepcopy(offspring[i])

        return offspring


def main():
    # 任务集，选择不同的任务集进行实验
    Prob = Func_Ackley()
    Prob = Func_Elliptic()
    Prob = Func_Griewank()
    Prob = Func_Rastrigin()
    Prob = Func_Rosenbrock()
    Prob = Func_Schwefel()
    Prob = Func_Schwefel2()
    Prob = Func_Sphere()
    # Prob = Func_Weierstrass()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = jDE().run(Prob, True)
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
