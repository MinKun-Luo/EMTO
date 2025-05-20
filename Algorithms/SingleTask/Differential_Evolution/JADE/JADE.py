# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Jade: Self-adaptive Differential Evolution with Fast and Reliable Convergence Performance
# @Author: Jingqiao Zhang and Sanderson, Arthur C.
# @Journal: 2007 IEEE Congress on Evolutionary Computation
# @year: 2007
# @Doi: 10.1109/CEC.2007.4424751

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: JADE论文复现

from collections import deque

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.Operator.DE_operator.DE import DE
from Problems.SingleTask.Classical_Function.Classical_Function import *


class JADE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.P = 0.1  # 贪婪选择概率
        self.C = 0.1  # 学习率

    def run(self, Prob, isPrint=False):
        # 初始化种群
        population = Initialization(self, Prob, Individual, False)

        # 初始化F和CR
        CRm = [0.5 for _ in range(Problem.T)]
        Fm = [0.5 for _ in range(Problem.T)]
        # 初始化选择被淘汰的亲本方案
        failA = [deque(maxlen=Problem.N) for _ in range(Problem.T)]

        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                # 初始化成功的F和CR
                SCR = np.array([])
                SF = np.array([])

                # 生成F和CR
                F = np.random.standard_cauchy(size=Problem.N) * 0.1 + Fm[t]
                F = np.where(F < 0, np.random.standard_cauchy(size=Problem.N) * 0.1 + Fm[t], F)
                F = np.clip(F, 0, 1)  # 缩放因子
                CR = np.clip(np.random.normal(CRm[t], 0.1, Problem.N), 0, 1)  # 交叉概率

                # 遍历目标种群，使用JADE算子进化
                for i in range(Problem.N):
                    # 子代
                    offspring = Individual()

                    # 基向量
                    p = population[t][i]
                    # 选取适应值前p%中的一个个体
                    pbest_idx = np.random.choice(np.argsort([ind.obj for ind in population[t]])[
                                                 :int(Problem.N * self.P)])
                    while pbest_idx == i:
                        pbest_idx = np.random.choice(np.argsort([ind.obj for ind in population[t]])[
                                                     :int(Problem.N * self.P)])
                    pbest = population[t][pbest_idx]
                    # 选择种群中的一个随机个体
                    r1 = np.random.choice([j for j in range(Problem.N) if j != i and j != pbest_idx])
                    p1 = population[t][r1]
                    # 选择种群和失败亲本存档结合处的一个个体
                    r2 = np.random.choice(
                        [j for j in range(Problem.N + len(failA[t])) if j != i and j != r1 and j != pbest_idx])
                    # 如果r2超出种群范围，则选择失败亲本存档
                    if r2 >= Problem.N:
                        p2 = failA[t][r2 - Problem.N]
                    else:
                        p2 = population[t][r2]

                    # 变异操作
                    offspring.rnvec = p.rnvec + F[i] * (pbest.rnvec - p.rnvec) + F[i] * (p1.rnvec - p2.rnvec)
                    # 交叉操作
                    offspring.rnvec = DE(p.rnvec, offspring.rnvec, len(offspring.rnvec), CR[i])
                    # 评估操作
                    self.Evaluation([offspring], Prob[t], t)
                    # 选择操作
                    if offspring.obj < p.obj:
                        # 更新存档（队列，先进先出）
                        failA[t].append(p)
                        # 更新种群（相当于锦标赛选择算子（对其索引的两个个体在同一竞赛池中））
                        population[t][i] = offspring
                        SCR = np.append(SCR, CR[i])
                        SF = np.append(SF, F[i])

                # 更新Fm和CRm
                Fm[t] = (1 - self.C) * Fm[t] + self.C * (np.sum(SF ** 2) / np.sum(SF))
                CRm[t] = (1 - self.C) * CRm[t] + self.C * np.mean(SCR)

        return self


def main():
    # 任务集，选择不同的任务集进行实验
    Prob = Func_Ackley()
    Prob = Func_Elliptic()
    Prob = Func_Griewank()
    # Prob = Func_Rastrigin()
    # Prob = Func_Rosenbrock()
    # Prob = Func_Schwefel()
    Prob = Func_Schwefel2()
    # Prob = Func_Sphere()
    # Prob = Func_Weierstrass()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = JADE().run(Prob, True)
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
