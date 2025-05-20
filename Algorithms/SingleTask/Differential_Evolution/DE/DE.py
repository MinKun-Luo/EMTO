# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces
# @Author: Rainer Storn & Kenneth Price
# @Journal: Journal of Global Optimization
# @year: 1997

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: DE论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import *
from Problems.SingleTask.Classical_Function.Classical_Function import *


class DE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5  # 缩放因子
        self.CR = 0.9  # 交叉概率

    def run(self, Prob, isPrint=False):
        # 初始化种群
        population = Initialization(self, Prob, Individual, False)

        while self.notTerminated(Prob, isPrint):
            # 选择性评估
            for t in range(Problem.T):
                # 生成子代
                offspring = self.Generation(population[t])
                self.Evaluation(offspring, Prob[t], t)
                # 精英保留策略
                population[t] = Selection_Elit(population[t], offspring, Problem.N)

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
        offspring = [Individual() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for i in range(N):
            # 获取父代个体
            p = population[i]
            # 初始化子代个体
            c = offspring[i]
            # 选择三个不同的个体
            x = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
            # 进行差分进化操作
            c.rnvec = DERand1(p.rnvec, population[x[0]].rnvec, population[x[1]].rnvec, population[x[2]].rnvec,
                              len(p.rnvec),
                              self.F, self.CR)

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
    Prob = Func_Weierstrass()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = DE().run(Prob, True)
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
