if __name__ == "__main__":
    import sys
    # sys.path.append("/Users/beita/Desktop/EMTO")
    # sys.path.append("/home/minkun/EMTO")
    sys.path.append("C:/Users/14531/Desktop/EMTO")

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
import numpy as np


class MGDMTO(Algorithm):
    def __init__(self):
        Problem.N = 50
        super().__init__()
        self.top = Problem.N
        self.nt = 2
        self.F = 0.5
        self.CR = 0.9
    
    def run(self, Prob, isPrint):
        # 初始化种群
        population = Initialization(self, Prob, Individual, isPadding=True)
        # 种群按适应度值排序，好的在前，差的在后
        for t in range(Problem.T):
            population[t] = sorted(population[t], key=lambda x: x.obj, reverse=False)
        # 开始循环
        while self.notTerminated(Prob, isPrint):
            # 获得种群的均值和方差
            [M, Sigma] = self.get_mean_std(population)
            for t in range(Problem.T):
                offspring = self.Generation(population[t], M[1 - t], Sigma[1 - t])
                self.Evaluation(offspring, Prob[t], t)
                population[t] = Selection_Elit(population[t], offspring, Problem.N)
        return self

    def get_mean_std(self, population):
        # 计算每个种群前nt个个体的均值和方差
        M1 = np.mean([ind.rnvec for ind in population[0][:self.top]], axis=0)
        M2 = np.mean([ind.rnvec for ind in population[1][:self.top]], axis=0)
        sigma1 = np.cov([ind.rnvec for ind in population[0][:self.top]], rowvar=False)
        sigma2 = np.cov([ind.rnvec for ind in population[1][:self.top]], rowvar=False)
        M = []
        M.append(M1)
        M.append(M2)
        Sigma = []
        Sigma.append(sigma1)
        Sigma.append(sigma2)
        return M, Sigma

    def Generation(self,population, M, Sigma):
        offspring = [Individual() for _ in range(Problem.N)]
        for i in range(Problem.N - self.nt):
            p = population[i]
            r1, r2, r3 = np.random.choice(range(len(population)), 3, replace=False)
            offspring[i].rnvec = DERand1(p.rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                                            len(p.rnvec), self.F, self.CR)
        generated_data = np.random.multivariate_normal(M, Sigma, size=self.nt)
        count = 0
        for i in range(Problem.N - self.nt, Problem.N):
            offspring[i].rnvec = generated_data[count]
            offspring[i].rnvec = np.clip(offspring[i].rnvec, 0, 1)
            count +=1
        return offspring

def main():
    # 测试函数
    Prob = CI_HS()
    # Prob = Benchmark4()
    # 设置最大评估次数
    Problem.maxFE = 200000
    repeat = 3
    for i in range(repeat):
        print(f"第{i+1}次运行")
        MGDMTO().run(Prob, True)


if __name__ == "__main__":
    main()