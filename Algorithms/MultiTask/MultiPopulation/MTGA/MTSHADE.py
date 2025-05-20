# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MTGA变体

import copy

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DE
from Algorithms.Utils.Operator.DE_operator.SHA import SHA
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class MTSHADE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.SHA = SHA(H=100)
        self.nt = 40

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization(self, Prob, Individual, False)

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

                # 子代
                offspring = [Individual() for _ in range(Problem.N)]

                F, CR = self.SHA.generate_f_cr(len(population[t]), t)
                # 遍历目标种群，使用SHADE算子进化
                for i in range(Problem.N):
                    # 选取辅助种群适应值前p%中的一个个体
                    pbest_idx = np.random.choice(np.argsort([ind.obj for ind in pop_temp])[
                                                 :int(Problem.N * np.random.uniform(2 / Problem.N, 0.2))])
                    while pbest_idx == i:
                        pbest_idx = np.random.choice(np.argsort([ind.obj for ind in pop_temp])[
                                                     :int(Problem.N * np.random.uniform(2 / Problem.N, 0.2))])
                    pbest = pop_temp[pbest_idx]

                    # 基向量
                    p = pop_temp[i]
                    # 选择目标种群中的一个个体
                    r1 = np.random.choice([j for j in range(Problem.N) if j != i and j != pbest_idx])
                    p1 = pop_temp[r1]
                    # 选择目标种群和失败亲本存档结合处的一个个体
                    r2 = np.random.choice(
                        [j for j in range(Problem.N + len(self.SHA.failA[t])) if j != i and j != r1 and j != pbest_idx])
                    # 如果r2超出种群范围，则选择失败亲本存档
                    if r2 >= Problem.N:
                        p2 = self.SHA.failA[t][r2 - Problem.N]
                    else:
                        p2 = pop_temp[r2]

                    # 变异操作
                    offspring[i].rnvec = p.rnvec + F[i] * (pbest.rnvec - p.rnvec) + F[i] * (p1.rnvec - p2.rnvec)
                    # 交叉操作
                    offspring[i].rnvec = DE(p.rnvec, offspring[i].rnvec, len(offspring[i].rnvec), CR[i])

                # 评估操作
                self.Evaluation(offspring, Prob[t], t)
                self.SHA.update_memory(pop_temp, offspring, F, CR, t)

                # 精英选择策略保留个体
                population[t] = Selection_Elit(pop_temp, offspring, Problem.N)

        # 返回最优适应度
        return self


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
        result = MTSHADE().run(Prob, True)
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