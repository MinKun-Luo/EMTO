# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MFDE的算子修改为SHADE

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual

from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import *
from Algorithms.Utils.Operator.DE_operator.DE import *
from Algorithms.Utils.Operator.DE_operator.SHA import SHA
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class MFSHADE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.rmp = 0.3  # 重组概率
        self.SHA = SHA(H=100)

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 10

        # 初始化种群
        population = Initialization(self, Prob, Individual)

        while self.notTerminated(Prob, isPrint=True):
            for t in range(Problem.T):
                F, CR = self.SHA.generate_f_cr(Problem.N, t)
                offspring = [Individual() for _ in range(Problem.N)]
                # 遍历目标种群，使用SHADE算子进化
                for i in range(Problem.N):
                    # 基向量
                    p = population[t][i]
                    # 选取目标种群适应值前p%中的一个个体
                    pbest_idx = np.random.choice([idx for idx in np.argsort([ind.obj for ind in population[t]])[:int(
                        Problem.N * np.random.uniform(2 / Problem.N, 0.2))] if idx != i])
                    pbest = population[t][pbest_idx]
                    # 子代类型选取
                    if np.random.rand() < 0.3:
                        # 选取辅助种群适应值前p%中的一个个体
                        pbest_idx = np.random.choice(np.argsort([ind.obj for ind in population[1 - t]])[
                                                     :int(Problem.N * np.random.uniform(2 / Problem.N, 0.2))])
                        pbest = population[1 - t][pbest_idx]
                        # # 随机选择转移种群TP_Next中的两个个体
                        r1, r2 = np.random.choice([j for j in range(Problem.N)], 2, replace=False)
                        p1, p2 = population[1 - t][r1], population[1 - t][r2]
                    else:
                        # 选择目标种群中的一个个体
                        r1 = np.random.choice([j for j in range(Problem.N) if j != i and j != pbest_idx])
                        p1 = population[t][r1]
                        # 选择目标种群和失败亲本存档结合处的一个个体
                        r2 = np.random.choice(
                            [j for j in range(Problem.N + len(self.SHA.failA[t])) if
                             j != i and j != r1 and j != pbest_idx])
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
                # 更新历史记忆存档
                self.SHA.update_memory(population[t], offspring, F, CR, t)
                # 锦标赛选择更新种群
                population[t] = Selection_Elit(population[t], offspring, num=Problem.N)

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
        result = MFSHADE().run(Prob, True)
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