# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Evolutionary Multitasking via Explicit Autoencoding
# @Author: Liang Feng; Lei Zhou; Jinghui Zhong; Abhishek Gupta; Yew-Soon Ong; Kay-Chen Tan
# @Journal: IEEE Transactions on Cybernetics
# @year: 2018
# @Doi: 10.1109/TCYB.2018.2845361

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/5/11 下午7:02
# @Author: wzb
# @Introduction: EMEA论文复现
# @Remind: 现有的论文都复现不出原论文的效果，本次复现与其他论文的复现结果一致


from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual

from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.DE_operator.DE import *
from Algorithms.Utils.Operator.Mutation import *
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.Problem import Problem


class EMEA(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.Operator = ['GA', 'DE']  # 进化器
        self.SNum = 10  # 转移数量
        self.TGap = 10  # 转移间隔
        self.eta = 2  # 模拟二进制交叉的分布指数
        self.mu = 5  # 多项式变异的分布指数
        self.F = 0.5  # 缩放因子
        self.CR = 0.6  # 交叉概率

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 100

        # 初始化所有任务的种群
        population = Initialization(self, Prob, Individual, isPadding=False)

        # 进化主循环，直到满足终止条件（评估次数达到上限）
        while self.notTerminated(Prob, isPrint):
            # 遍历每一个任务
            for t in range(Problem.T):
                # 不同任务使用不同的进化器
                if self.Operator[t] == "GA":
                    # 使用遗传算法生成子代
                    offspring = self.GA(population[t])
                elif self.Operator[t] == "DE":
                    # 使用差分进化算法生成子代
                    offspring = self.DE(population[t])
                else:
                    raise ValueError(f"Unknown operator")

                # 知识迁移机制：每隔 TGap 代进行一次
                if self.SNum > 0 and self.Gen % self.TGap == 0:
                    # 遍历其他任务，将知识注入当前任务 t
                    for k in range(Problem.T):
                        if k == t:
                            continue  # 跳过自身

                        # 对目标任务（当前任务）按适应度排序，提取种群基因型矩阵
                        population[t] = sorted(population[t], key=lambda ind: ind.obj)
                        target = np.array([ind.rnvec for ind in population[t]])

                        # 对源任务（其他任务）按适应度排序，提取种群基因型矩阵
                        population[k] = sorted(population[k], key=lambda ind: ind.obj)
                        source = np.array([ind.rnvec for ind in population[k]])

                        # 执行迁移操作：从 source 向 target 进行知识映射
                        trans_rnvecs = self.mDA(target, source)

                        # 构造新的个体，并限制其变量在 [0, 1] 区间
                        trans_pop = [Individual(rnvec=np.clip(rnvec, 0, 1)) for rnvec in trans_rnvecs]

                        # 从子代中随机选择若干个体被替换为迁移个体
                        random_indices = np.random.choice(len(offspring), len(trans_pop), replace=False)
                        for idx, trans_ind in zip(random_indices, trans_pop):
                            offspring[idx] = trans_ind

                # 对子代进行评估（仅评估当前任务的目标函数）
                self.Evaluation(offspring, Prob[t], t)
                # 精英保留：从父代和子代中选出前 N 个最优个体
                population[t] = Selection_Elit(population[t], offspring, num=Problem.N)

        return self

    def mDA(self, target, source):
        """
        多源自编码知识转移方法（modified Domain Adaptation）

        :param target: 当前任务的种群表示矩阵 (N × d1)，其中 N 为个体数，d1 为当前任务维度
        :param source: 辅助任务的种群表示矩阵 (N × d2)，其中 N 为个体数，d2 为该任务维度
        :return: inj_solution 转移后的决策变量矩阵（N × d1），用于注入当前任务
        """
        # 获取源任务和目标任务的维度
        target_dim = target.shape[1]
        source_dim = source.shape[1]

        # 对齐维度：低维补零，使 source 和 target 的维度一致，便于矩阵运算
        if target_dim < source_dim:
            padding = np.zeros((target.shape[0], source_dim - target_dim))
            target = np.hstack((target, padding))
        elif target_dim > source_dim:
            padding = np.zeros((source.shape[0], target_dim - source_dim))
            source = np.hstack((source, padding))

        # 构造输入矩阵（增加一维偏置项1）
        xx = np.vstack((target.T, np.ones((1, target.shape[0]))))  # (d+1) × N
        noise = np.vstack((source.T, np.ones((1, source.shape[0]))))  # (d+1) × N

        # 计算协方差矩阵与映射矩阵
        Q = np.dot(noise, noise.T)  # 噪声自相关矩阵
        P = np.dot(xx, noise.T)  # 源域与目标域的交叉相关矩阵

        # 正则化项（避免矩阵奇异）
        lam = 1e-5
        reg = lam * np.eye(Q.shape[0])
        reg[-1, -1] = 0  # 偏置项不正则化

        # 计算映射矩阵 W（类似自编码器的投影矩阵）
        W = np.dot(P, np.linalg.inv(Q + reg))
        W = W[:-1, :-1]  # 去除偏置维度，得到最终权重矩阵

        # 提取源任务中前 SNum 个最优解作为迁移对(源种群的维度较低时，已在上面对齐)
        source_bestSolution = source[:self.SNum]

        # 执行知识转移
        tmp_solution = np.dot(W, source_bestSolution.T).T
        # 截断到目标任务的维度
        inj_solution = tmp_solution[:, :target_dim]

        # 转移后的决策变量矩阵
        return inj_solution

    def GA(self, population):
        """
        通过GA生成子代种群

        :param population: 父代种群，当前代的所有个体
        :return: 生成的子代个体列表
        """
        # 初始化子代个体
        offspring = [Individual() for _ in range(len(population))]
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
            c1.rnvec, c2.rnvec = SBX1(p1.rnvec, p2.rnvec, self.eta)
            c1.rnvec = PM2(p1.rnvec, self.mu)
            c2.rnvec = PM2(p2.rnvec, self.mu)

        # 返回生成的子种群
        return offspring

    def DE(self, population):
        """
        通过DE/Rand/1生成子代种群

        :param population: list 父代种群，当前代的所有个体
        :return: list offspring 生成的子代个体列表
        """
        # 子代
        offspring = [Individual() for _ in range(Problem.N)]
        # 进化
        for i in range(Problem.N):
            # 随机选择三个不同的个体
            r1, r2, r3 = np.random.choice([k for k in range(Problem.N) if k != i], 3, replace=False)
            # 原向量
            p = population[i].rnvec
            # 从当前种群中选择两个不同个体，形成差异向量
            p1, p2, p3 = population[r1].rnvec, population[r2].rnvec, population[r3].rnvec
            # DERand1
            offspring[i].rnvec = DERand1(p, p1, p2, p3, len(p), self.F, self.CR)

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
    Problem.maxFE = 200000

    costs = np.zeros((repeat, Problem.T))
    for i in range(repeat):
        print(f'Repetition: {i} :')
        result = EMEA().run(Prob, True)
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
