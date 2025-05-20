# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Multitasking Genetic Algorithm (MTGA) for Fuzzy System Optimization
# @Author: Dongrui Wu; Xianfeng Tan
# @Journal: IEEE Transactions on Fuzzy Systems
# @year: 2020
# @Doi: 10.1109/TFUZZ.2020.2968863

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MTGA论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class MTGA(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.eta = 2
        self.mu = 5
        self.nt = 40  # 偏差个体

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization(self, Prob, Individual, isPadding=False)
        # 按适应值升序排序（好的在前面）
        for t in range(Problem.T):
            population[t] = sorted(population[t], key=lambda x: x.obj, reverse=False)

        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                # 偏差临时种群
                pop_temp = copy.deepcopy(population[t])
                # 种群分布
                M1 = np.mean([ind.rnvec for ind in population[t][:self.nt]], axis=0)
                M2 = np.mean([ind.rnvec for ind in population[1 - t][:self.nt]], axis=0)
                # 偏差调整转移
                for i in range(self.nt):
                    # 替换索引
                    I = np.random.choice(range(Prob[1 - t].dim), Prob[t].dim, replace=Prob[t].dim > Prob[1 - t].dim)
                    pop_temp[-(i + 1)].rnvec = population[1 - t][i].rnvec[I] - M2[I] + M1

                # 生成子代
                offspring = self.Generation(pop_temp)
                # 选择性评估
                self.Evaluation(offspring, Prob[t], t)
                # 精英保留策略
                population[t] = Selection_Elit(population[t], offspring, Problem.N)

        # 返回最优适应度
        return self

    def Generation(self, population):
        """
        偏差种群进化产生子代

        :param population: 偏差种群。
        :return: offspring: 子种群。
        """
        # 随机排列种群索引
        inorder = np.random.permutation(len(population))
        # 初始化计数器，用于存储后代个体
        count = 0
        offspring = [Individual() for _ in range(len(population))]
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
            # 模拟二进制交叉
            c1.rnvec, c2.rnvec = SBX(p1.rnvec, p2.rnvec, self.eta)
            # 多项式变异
            c1.rnvec = PM2(c1.rnvec, self.mu)
            c2.rnvec = PM2(c2.rnvec, self.mu)

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
        result = MTGA().run(Prob, True)
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