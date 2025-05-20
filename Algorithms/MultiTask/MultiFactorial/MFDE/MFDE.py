# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: An empirical study of multifactorial PSO and multifactorial DE
# @Author: L. Feng; W. Zhou; L. Zhou; S. W. Jiang; J. H. Zhong; B. S. Da
# @Journal: 2017 IEEE Congress on Evolutionary Computation (CEC)
# @year: 2017
# @Doi: 10.1109/CEC.2017.7969407

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFDE论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual_MF import Individual_MF
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Algorithms.Utils.MultiFactorial.Selection_MF import Selection_MF
from Algorithms.Utils.Operator.DE_operator.DE import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MFDE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.rmp = 0.3  # 重组概率
        self.F = 0.5  # 缩放因子
        self.CR = 0.9  # 交叉概率

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization_MF(self, Prob, Individual_MF)

        while self.notTerminated(Prob, isPrint):
            # 生成子代
            offspring = self.Generation(population)
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

    def Generation(self, population):
        """
        生成子代种群。

        :param population: list 父代种群，当前代的所有个体。
        :return: list offspring 生成的子代个体列表。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspring = [Individual_MF() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for i in range(N):
            # 获取父代个体
            p = population[i]
            # 初始化子代个体
            c = offspring[i]
            # 选择基向量索引（与p具有相同的技能因子）
            x1 = np.random.randint(N)
            while x1 == i or population[x1].MFFactor != p.MFFactor:
                x1 = np.random.randint(N)

            # 如果随机数小于随机交配概率（rmpGen），随机选择两个与p具有不同技能因子的个体进行差分进化操作
            if np.random.rand() < self.rmp:
                x2 = np.random.randint(N)
                while population[x2].MFFactor == p.MFFactor:
                    x2 = np.random.randint(N)
                x3 = np.random.randint(N)
                while x3 == x2 or population[x3].MFFactor == p.MFFactor:
                    x3 = np.random.randint(N)
                MFFactors = [population[x1].MFFactor, population[x2].MFFactor,
                             population[x3].MFFactor]
                c.MFFactor = MFFactors[np.random.randint(3)]
            # 否则，随机选择三个与p具有相同技能因子的个体进行差分进化操作
            else:
                x2 = np.random.randint(N)
                while x2 == i or x2 == x1 or population[x2].MFFactor != p.MFFactor:
                    x2 = np.random.randint(N)
                x3 = np.random.randint(N)
                while x3 == i or x3 == x1 or x3 == x2 or population[x3].MFFactor != p.MFFactor:
                    x3 = np.random.randint(N)
                c.MFFactor = p.MFFactor

            # 进行差分进化操作
            c.rnvec = DERand1(p.rnvec, population[x1].rnvec, population[x2].rnvec, population[x3].rnvec, len(p.rnvec),
                              self.F, self.CR)

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
        result = MFDE().run(Prob, True)
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