# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Success-history based parameter adaptation for Differential Evolution
# @Author: Ryoji Tanabe; Alex Fukunaga
# @Journal: 2013 IEEE Congress on Evolutionary Computation
# @year: 2013
# @Doi: 10.1109/CEC.2013.6557555

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 基于SHA函数实现SHADE

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Tournament
from Algorithms.Utils.Operator.DE_operator.DE import DE
from Algorithms.Utils.Operator.DE_operator.SHA import SHA
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.SingleTask.Classical_Function.Classical_Function import *


class SHADET(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.SHA = SHA(H=100)

    def run(self, Prob, isPrint=False):
        Problem.N = 100

        # 初始化种群
        population = Initialization(self, Prob, Individual, False)

        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                F, CR = self.SHA.generate_f_cr(Problem.N, t)
                offspring = [Individual() for _ in range(Problem.N)]
                # 遍历目标种群，使用SHADE算子进化
                for i in range(Problem.N):
                    # 选取辅助种群适应值前p%中的一个个体
                    pbest_idx = np.random.choice(np.argsort([ind.obj for ind in population[t]])[
                                                 :int(Problem.N * np.random.uniform(2 / Problem.N, 0.2))])
                    while pbest_idx == i:
                        pbest_idx = np.random.choice(np.argsort([ind.obj for ind in population[t]])[
                                                     :int(Problem.N * np.random.uniform(2 / Problem.N, 0.2))])
                    pbest = population[t][pbest_idx]

                    # 基向量
                    p = population[t][i]
                    # 选择目标种群中的一个个体
                    r1 = np.random.choice([j for j in range(Problem.N) if j != i and j != pbest_idx])
                    p1 = population[t][r1]
                    # 选择目标种群和失败亲本存档结合处的一个个体
                    r2 = np.random.choice(
                        [j for j in range(Problem.N + len(self.SHA.failA[t])) if j != i and j != r1 and j != pbest_idx])
                    # 如果r2超出种群范围，则选择失败亲本存档
                    if r2 >= Problem.N:
                        p2 = self.SHA.failA[t][r2 - Problem.N]
                    else:
                        p2 = population[t][r2]

                    # 变异操作
                    offspring[i].rnvec = p.rnvec + F[i] * (pbest.rnvec - p.rnvec) + F[i] * (p1.rnvec - p2.rnvec)
                    # 交叉操作
                    offspring[i].rnvec = DE(p.rnvec, offspring[i].rnvec, len(offspring[i].rnvec), CR[i])

                # 评估操作
                self.Evaluation(offspring, Prob[t], t)
                self.SHA.update_memory(population[t], offspring, F, CR, t)
                population[t], replace = Selection_Tournament(population[t], offspring)

        return self


def main():
    # 任务集，选择不同的任务集进行实验
    # Prob = Func_Ackley()
    # Prob = Func_Elliptic()
    # Prob = Func_Griewank()
    # Prob = Func_Rastrigin()
    # Prob = Func_Rosenbrock()
    # Prob = Func_Schwefel()
    # Prob = Func_Schwefel2()
    # Prob = Func_Sphere()
    # Prob = Func_Weierstrass()
    Prob = CI_HS()  # 高复杂度的CI任务集
    # Prob = CI_MS()  # 中等复杂度的CI任务集
    # Prob = CI_LS()  # 低复杂度的CI任务集
    # Prob = PI_HS()  # 高复杂度的PI任务集
    # Prob = PI_MS()  # 中等复杂度的PI任务集
    Prob = PI_LS()  # 低复杂度的PI任务集
    # Prob = NI_HS()  # 高复杂度的NI任务集
    # Prob = NI_MS()  # 中等复杂度的NI任务集
    # Prob = NI_LS()  # 低复杂度的NI任务集
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
        result = SHADET().run(Prob, True)
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
