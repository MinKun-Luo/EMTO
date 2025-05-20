# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Multifactorial Evolution: Toward Evolutionary Multitasking
# @Author: Abhishek Gupta; Yew-Soon Ong; Liang Feng
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2016
# @Doi: 10.1109/TEVC.2015.2458037

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFEA论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual_MF import Individual_MF
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Algorithms.Utils.MultiFactorial.Selection_MF import Selection_MF
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MFEA(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.rmp = 0.3  # 重组概率
        self.eta = 2  # 模拟二进制交叉的分布指数
        self.mu = 5  # 多项式变异的分布指数

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
            # 精英保留策略m
            population = Selection_MF(population, offspring, num=Problem.N)
            # 删除子种群和临时种群
            del offspring, offspring_t

        return self

    def Generation(self, population):
        """
        生成子代种群。

        :param population: 父代种群，当前代的所有个体。
        :return: 生成的子代个体列表。
        """
        # 初始化子代个体
        offspring = [Individual_MF() for _ in range(len(population))]
        # 随机排列种群索引
        inorder = np.random.permutation(len(population))
        # 初始化计数器，用于存储后代个体
        count = 0

        # 遍历种群的一半进行交叉操作
        for i in range(int(len(population) / 2)):
            # 选择两个父代个体
            p1 = population[inorder[i]]
            p2 = population[inorder[i + int(len(population) / 2)]]

            # 初始化两个子代个体
            c1 = offspring[count]
            c2 = offspring[count + 1]
            # 更新计数器
            count += 2

            # 遗传机制
            if p1.MFFactor == p2.MFFactor or np.random.rand() < self.rmp:
                # 如果技能因子相同或满足转移条件，则选择两个个体进行模拟二进制交叉和突变
                c1.rnvec, c2.rnvec = SBX1(p1.rnvec, p2.rnvec, self.eta)
                # 更新技能因子，随机模仿父代中的一方
                c1.MFFactor = p1.MFFactor if np.random.rand() < 0.5 else p2.MFFactor
                c2.MFFactor = p1.MFFactor if np.random.rand() < 0.5 else p2.MFFactor
            else:
                # 轻微变异，并继承父代的技能因子
                c1.rnvec = PM2(p1.rnvec, self.mu)
                c1.MFFactor = p1.MFFactor
                c2.rnvec = PM2(p2.rnvec, self.mu)
                c2.MFFactor = p2.MFFactor

        # 返回生成的子种群
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
        result = MFEA().run(Prob, True)
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
