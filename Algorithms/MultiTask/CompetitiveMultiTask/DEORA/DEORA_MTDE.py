# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: DEORA变体

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual

from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Tournament
from Algorithms.Utils.Operator.DE_operator.DE import *
from Algorithms.Utils.Operator.Selection import *
from Problems.MultiTask.Competitive_C2TOP.CEC17_MTSO_Competitive import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class DEORA_MTDE(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5  # 缩放因子
        self.CR = 0.9  # 交叉概率
        self.alpha = 0.5  # 学习率
        self.P0 = 0.3  # 初始概率

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 最大进化代数
        maxGen = int((Problem.maxFE - Problem.N * Problem.T) / Problem.N + 1)
        # 奖励增量
        delta = 1 / maxGen
        # 奖励存档(每个任务在每一代的奖励值)
        HR = np.zeros((Problem.T, 0))
        # 任务间通信概率(任务之间的通信概率)
        RMP = np.full(shape=(Problem.T, Problem.T), fill_value=self.P0 / (Problem.T - 1), dtype=float)
        np.fill_diagonal(RMP, 1 - self.P0)

        # 初始化种群
        population = Initialization(self, Prob, Individual, isPadding=True)

        while self.notTerminated(Prob, isPrint):
            for k in range(Problem.T):
                # 生成子代
                offspring, r1_task = self.Generation(population, RMP, k)
                # 全局最优值
                best_g = np.min(self.Best)
                # 主任务第g-1代的个体适应值
                fit_old = np.array([ind.obj for ind in population[k]])
                # 评估主任务的子种群
                self.Evaluation(offspring, Prob[k], k)
                # 锦标赛选择更新主任务种群
                population[k], replace = Selection_Tournament(population[k], offspring)
                # 主任务第g代的个体适应值
                fit_new = np.array([ind.obj for ind in population[k]])

                # 全局最优值改进率
                R_b = max((best_g - np.min(fit_new)) / best_g, 0)
                # 主种群的个体平均改进比率
                R_p = np.maximum((fit_old - fit_new) / fit_old, 0)
                # 当前代的奖励
                R = np.zeros((Problem.T, 1), dtype=float)
                # 计算奖励值
                for t in range(Problem.T):
                    if t == k:  # 主任务的奖励
                        R[t] = self.alpha * R_b + (1 - self.alpha) * (np.sum(R_p) / len(R_p))
                    else:  # 辅助任务的奖励
                        # 辅助任务参与的子代个体
                        index = np.where(r1_task == t)
                        if len(index[0]) != 0:
                            # 更新后的主任务最优值索引
                            minid = np.argmin(fit_new)
                            # 辅助任务参与的全局改进率和主任务个体的平均改进率
                            R[t] = self.alpha * (r1_task[minid] == t) * R_b + (1 - self.alpha) * (
                                    np.sum(R_p[index]) / len(index))
                        else:
                            R[t] = 0
                # 更新奖励存档
                HR = np.hstack((HR, R))

                # 更新任务间通信概率RMP
                for t in range(Problem.T):
                    if t == k:
                        continue
                    else:
                        # 辅助任务获得更多奖励，则适当增加通信概率，减少自身的通信概率
                        if R[t] > R[k]:
                            RMP[k][t] = min((RMP[k][t] + delta, 1))
                            RMP[k][k] = max((RMP[k][k] - delta, 0))
                        else:  # 辅助任务获得更少奖励，则适当减少通信概率，增加自身的通信概率
                            RMP[k][t] = max((RMP[k][t] - delta, 0))
                            RMP[k][k] = min((RMP[k][k] + delta, 1))

        return self

    def Generation(self, population, RMP, k):
        """
        生成子代种群。

        :param population: list，所有任务的种群，格式为二维列表 population[任务索引][个体]
        :param RMP: np.ndarray，任务间通信概率矩阵，用于选择基向量来源的任务
        :param k: int，当前主任务的索引
        :return:
            offspring (list): 生成的子代个体列表（仅主任务的子代）
            r1_task (np.ndarray): 每个子代个体中基向量的任务来源索引数组
        """
        # 基向量的任务来源索引
        r1_task = np.zeros(len(population[k]), dtype=int)
        # 子代
        offspring = [Individual() for _ in range(len(population[k]))]
        # 进化
        for i in range(len(population[k])):
            # 基于任务间通信概率RMP选择基向量来源
            r1_task[i] = RouletteWheelSelection(RMP[k], 1)[0]
            r1, r2, r3 = np.random.choice([k for k in range(Problem.N) if k != i], 3, replace=False)
            # 原向量
            p = population[k][i].rnvec
            # 差分向量
            p2, p3 = population[k][r2].rnvec, population[k][r3].rnvec
            # 基向量
            p1 = population[r1_task[i]][r1].rnvec

            # DERand1
            offspring[i].rnvec = DERand1(p, p1, p2, p3, len(p), self.F, self.CR)

        return offspring, r1_task


def main():
    # 测试函数
    case = 3
    Prob = C_CI_HS(case=case)
    Prob = C_CI_MS(case=case)
    Prob = C_CI_LS(case=case)
    Prob = C_PI_HS(case=case)
    Prob = C_PI_MS(case=case)
    Prob = C_PI_LS(case=case)
    Prob = C_NI_HS(case=case)
    Prob = C_NI_MS(case=case)
    Prob = C_NI_LS(case=case)
    # 重复次数
    repeat = 30
    # 设置最大评估次数
    Problem.maxFE = 100000
    #
    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = DEORA_MTDE().run(Prob, True)
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
