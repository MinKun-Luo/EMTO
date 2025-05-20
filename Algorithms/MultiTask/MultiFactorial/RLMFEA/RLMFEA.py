# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Multitasking via Reinforcement Learning
# @Author: Shuijia Li; Wenyin Gong; Ling Wang; Qiong Gu
# @Journal: IEEE Transactions on Emerging Topics in Computational Intelligence
# @year: 2024
# @Doi: 10.1109/TETCI.2023.3281876

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: RLMFEA论文复现

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual_MF import Individual_MF
from Algorithms.Utils.MultiFactorial.Initialization_MF import Initialization_MF
from Algorithms.Utils.MultiFactorial.Selection_MF import Selection_MF
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Algorithms.Utils.Operator.Mutation import *
from Algorithms.Utils.Operator.Selection import RouletteWheelSelection
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem

F = 0.5  # 缩放因子
CR = 0.6  # 交叉概率
eta = 10  # 模拟二进制交叉分布指数
mu = 5  # 多项式变异分布指数
alpha = 0.1  # 学习率
beta = 0.9  # 折扣率
w = 1  # 控制缩放参数
rmp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 随机交配概率
reward = [10, 5, 0]  # 奖励值


class RLMFEA(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数

    def run(self, Prob, isPrint=False):
        # 论文设定的种群大小
        Problem.N = 100

        # 初始化种群
        population = Initialization_MF(self, Prob, Individual_MF)
        # 初始化 Q 表，用于存储每个任务的状态-动作值
        Q = [np.zeros(shape=(3, 7)) for _ in range(Problem.T)]
        # 初始化选择概率表
        p = [np.zeros((3, 7)) for _ in range(Problem.T)]
        # 初始化随机交配概率索引
        rmp_index = np.full(2, 2)
        # 初始化当前状态
        state_now = np.ones(2, dtype=int)

        while self.notTerminated(Prob, isPrint):
            # 分配算子，0为DE，1为GA
            if np.random.rand() < 0.5:
                # 任务1-技能因子1-rank_DE 任务2-技能因子2-GA
                S1, S2 = 'rank_DE', 'GA'
            else:
                # 任务1-技能因子1-GA 任务2-技能因子2-rank_DE
                S1, S2 = 'GA', 'rank_DE'
            # 获取种群1和种群2的索引
            subPop1 = [idx for idx, ind in enumerate(population) if ind.MFFactor == 1]
            subPop2 = [idx for idx, ind in enumerate(population) if ind.MFFactor == 2]
            offspring = [Individual_MF() for _ in range(len(population))]
            if S1 == 'rank_DE':
                # 种群1采用DE算子
                self.DE(population, offspring, subPop1, subPop2, rmp[rmp_index[0]])
                # 种群2采用GA算子
                self.GA(population, offspring, subPop2, subPop1, rmp[rmp_index[1]])
            else:
                # 种群1采用GA算子
                self.GA(population, offspring, subPop1, subPop2, rmp[rmp_index[0]])
                # 种群2采用DE算子
                self.DE(population, offspring, subPop2, subPop1, rmp[rmp_index[1]])

            # 选择性评估
            for t in range(Problem.T):
                offspring_t = [ind for ind in offspring if ind.MFFactor == t + 1]
                self.Evaluation(offspring_t, Prob[t], t)
                for ind in offspring_t:
                    ind.MFCosts[t] = ind.obj

            # 更新Q表
            for t in range(Problem.T):
                # 记录子代的最优值
                c_best = np.min([ind.MFCosts[t] for ind in offspring if ind.MFFactor == t + 1])
                # 记录父代的最优值
                p_best = np.min([ind.MFCosts[t] for ind in population if ind.MFFactor == t + 1])
                if c_best < p_best:
                    rmp_index[t] = self.updateRmp(Q[t], p[t], state_now[t], 0, rmp_index[t], 0)
                    state_now[t] = 0
                elif np.abs(c_best - p_best <= 1e-10):
                    rmp_index[t] = self.updateRmp(Q[t], p[t], state_now[t], 1, rmp_index[t], 1)
                    state_now[t] = 1
                else:
                    rmp_index[t] = self.updateRmp(Q[t], p[t], state_now[t], 2, rmp_index[t], 2)
                    state_now[t] = 2

            # 精英保留策略
            population = Selection_MF(population, offspring, num=Problem.N)

            # 删除子种群和临时种群
            del offspring, offspring_t

        return self

    @staticmethod
    def DE(population, offspring, subPop1, subPop2, rmpGen):
        """
        种群自我进化（差分进化操作）。

        :param population: 父种群，包含当前所有个体的列表。
        :param offspring: 子种群，用于存储进化后产生的新个体。
        :param subPop1: 当前任务所属子种群的个体索引列表。
        :param subPop2: 另一任务所属子种群的个体索引列表（用于知识转移）。
        :param rmpGen: 当前的随机交配概率（用于控制是否进行知识转移）。
        :return: 更新后的子种群（可省略返回，因为是在原地更新）。
        """
        for idx in subPop1:
            # 从种群1中随机选择三个不同的个体
            p1_selected = np.random.choice([i for i in subPop1 if i != idx], 3, replace=False)
            # 从种群2中随机选择两个个体
            p2_selected = np.random.choice(subPop2, 2, replace=False)
            # 父代个体
            p = population[idx]
            # 基向量
            p1 = population[p1_selected[0]]
            # 后代个体
            c = offspring[idx]
            # 根据随机交配概率选择父代个体
            if np.random.rand() <= rmpGen:
                # 从另一个种群中选择两个不同的个体（实现知识转移）
                p2 = population[p2_selected[0]]
                p3 = population[p2_selected[1]]
            else:
                # 从当前种群中选择两个不同的个体（不实现知识转移）
                p2 = population[p1_selected[1]]
                p3 = population[p1_selected[2]]
            c.rnvec = DERand1(p.rnvec, p1.rnvec, p2.rnvec, p3.rnvec, len(p.rnvec), F, CR)
            # 更新技能因子
            c.MFFactor = p.MFFactor

    @staticmethod
    def GA(population, offspring, subPop1, subPop2, rmpGen):
        """
        种群自我进化（遗传算法操作）。

        :param population: 父种群，包含当前所有个体的列表。
        :param offspring: 子种群，用于存储进化后产生的新个体。
        :param subPop1: 当前任务所属子种群的个体索引列表。
        :param subPop2: 另一任务所属子种群的个体索引列表（用于知识转移）。
        :param rmpGen: 当前的随机交配概率（用于控制是否进行知识转移）。
        :return: 更新后的子种群（可省略返回，因为是在原地更新）。
        """
        for idx in subPop1:
            # 父代个体
            p1 = population[idx]
            # 从种群1中随机选择另一个不同的个体
            p1_selected = np.random.choice([i for i in subPop1 if i != idx], 1, replace=False)
            # 从种群2中随机一个个体
            p2_selected = np.random.choice(subPop2, 1, replace=False)
            # 初始化子代个体
            c = offspring[idx]
            # 根据随机交配概率选择父代个体
            if np.random.rand() <= rmpGen:
                # 从另一个种群中选择一个个体（实现知识转移）
                p2 = population[p2_selected[0]]
            else:
                # 从当前种群中选择一个个体（不实现知识转移）
                p2 = population[p1_selected[0]]
            # 如果技能因子相同或满足转移条件，则选择两个个体进行模拟二进制交叉和突变
            c1, c2 = SBX(p1.rnvec, p2.rnvec, eta)
            if np.random.rand() < 0.5:
                c.rnvec = c1
                c.MFFactor = p1.MFFactor
            else:
                c.rnvec = c2
                c.MFFactor = p2.MFFactor
            # 多项式变异
            c.rnvec = PM2(c.rnvec, mu)

        return offspring

    @staticmethod
    def updateRmp(Q, p, state_now, state_next, rmp_index, reward_index):
        """
        更新随机交配概率（用于强化学习中的Q表更新）。

        :param Q: 当前任务对应的 Q 表（状态-动作值表）。
        :param p: 当前任务对应的选择概率表。
        :param state_now: 当前状态（整数索引，范围0-2）。
        :param state_next: 下一状态（整数索引，范围0-2）。
        :param rmp_index: 当前使用的随机交配概率索引（0-6）。
        :param reward_index: 奖励索引，用于从 reward 列表中获取对应的奖励值。
        :return: updateRmp_index[0]: 更新后的随机交配概率索引（用于下一轮交配概率选择）。
        """
        # 更新Q表
        Q[state_now, rmp_index] = Q[state_now, rmp_index] + alpha * (
                reward[reward_index] + beta * np.max(Q[state_next, :]) - Q[state_now, rmp_index])

        # 计算选择概率
        for i in range(3):
            for j in range(7):
                p[i, j] = np.exp(Q[i, j] / w) / np.sum(np.exp(Q[i, :] / w))

        # 更新随机交配概率索引
        updateRmp_index = RouletteWheelSelection(p[state_next])

        return updateRmp_index[0]


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
        result = RLMFEA().run(Prob, True)
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