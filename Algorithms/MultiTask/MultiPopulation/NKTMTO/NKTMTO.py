if __name__ == "__main__":
    import sys
    # sys.path.append("/Users/beita/Desktop/EMTO")
    # sys.path.append("/home/minkun/EMTO")
    # sys.path.append("C:/Users/14531/Desktop/EMTO")

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *

class NKTMTO(Algorithm):
    def __init__(self):
        Problem.N = 50
        super().__init__()
        self.F = 0.5
        self.CR = 0.9
    
    def run(self, Prob, isPrint=True):
        
        population = Initialization(self, Prob, Individual, isPadding=True)
        while self.notTerminated(Prob, isPrint):
            for t in range(Problem.T):
                offspring = self.Generation(population[t])
                self.Evaluation(offspring, Prob[t], t)
                population[t] = Selection_Elit(population[t], offspring, Problem.N)
        return self


    def Generation(self, population):
        """
        偏差种群进化产生子代

        :param population: 偏差种群。
        :return: offspring: 子种群。
        """
        offspring = [Individual() for _ in range(len(population))]
        for i in range(len(population)):
            p = population[i]
            r1, r2, r3 = np.random.choice(range(len(population)), 3, replace=False)
            offspring[i].rnvec = DERand1(p.rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                                         len(p.rnvec), self.F, self.CR)

        # 返回生成的子种群
        return offspring

def main():
    # 测试函数
    Prob = CI_HS()
    # Prob = Benchmark10()
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 200000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = NKTMTO().run(Prob, True)
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