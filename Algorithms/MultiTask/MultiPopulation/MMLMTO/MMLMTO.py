# <-*--*--*--*- Reference -*--*--*--*--*->
# @title: Multi-Level and Multi-Segment Learning Multitask Optimization via a Niching Method
# @Author: Zhao-Feng Xue; Zi-Jia Wang; Yi Jiang; Zhi-Hui Zhan; Sam Kwong; Jun Zhang
# @Journal: IEEE Transactions on Evolutionary Computation
# @year: 2024
# @Doi: 10.1109/TEVC.2024.3511941

# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: MMLMTO论文复现
# @Remind: 进化池更新权重（原论文：全局最优值连续两代停滞，则随机生成weight3）。效果不及预期，修改为固定权重

from Algorithms.Algorithm import Algorithm
from Algorithms.Utils.Individual.Individual import Individual
from Algorithms.Utils.MultiPopulation.Initialization import Initialization
from Algorithms.Utils.MultiPopulation.Selection_MP import Selection_Elit
from Algorithms.Utils.Operator.Crossover import *
from Algorithms.Utils.Operator.DE_operator.DE import DERand1
from Algorithms.Utils.Operator.Mutation import *
from Algorithms.Utils.Operator.Selection import RouletteWheelSelection
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *


class MMLMTO(Algorithm):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.F = 0.5  # 缩放因子
        self.CR = 0.9  # 交叉概率
        self.pkt = 0.3  # 知识转移概率
        self.nt = 5  # 每个层次参与知识转移的个体数
        self.S = 15  # 源种群中被选为知识转移的最优个体数
        self.L = [5, 15]  # 片段长度取值范围

    def run(self, Prob, isPrint=False):
        # 论文设定的每个子种群大小
        Problem.N = 75

        # 初始化种群
        population = Initialization(self, Prob, Individual, False)
        # 初始化标志和权重
        flag = [0, 0]
        weight = [0.5, 0.5]

        while self.notTerminated(Prob, isPrint):
            flag_temp = [False, False]
            # DE进化
            for t in range(Problem.T):
                # rank_DE/rand/1 产生子代
                offspring = self.DE(population[t])
                # 选择性评估
                flag_temp[t] = self.Evaluation(offspring, Prob[t], t)
                # 精英保留策略（亲本和DE子代）
                population[t] = Selection_Elit(population[t], offspring, Problem.N)

            # 知识转移
            for t in range(Problem.T):
                # 知识转移
                if np.random.rand() < self.pkt:
                    # 多层次学习策略
                    cp_t, cp_s, pop_t_level = self.MML(population[t], population[1 - t])
                    # 多分段学习策略
                    childrens = self.MSL(cp_t, cp_s, weight[t])
                    # 选择性评估
                    for children in childrens:
                        if self.Evaluation(children, Prob[t], t):
                            flag_temp[t] = True
                    # 各层次精英保留策略（亲本和转移子代）
                    population[t] = []
                    for i in range(3):
                        population[t].extend(Selection_Elit(pop_t_level[i], childrens[i], int(Problem.N / 3)))
            # 进化池更新权重（全局最优值连续两代停滞，则随机生成weight3）
            # 效果不如预期，暂时注释
            # for t in range(Problem.T):
            #     if flag_temp[t]:
            #         flag[t] = flag[t] + 1
            #         if flag[t] == 2:
            #             flag[t] = 0
            #             weight[t] = np.random.uniform(0.1, 0.9)
            #     else:
            #         flag[t] = 0

        # 返回最优适应度
        return self

    def DE(self, population):
        """
        子种群差分进化操作（rank_DE）

        :param population: 父种群。
        :return: offspring: numpy 数组，DE进化产生的子种群。
        """
        # 获取种群大小
        N = len(population)
        # 初始化子代个体
        offspring = [Individual() for _ in range(N)]
        # 遍历种群进行差分进化操作
        for n in range(N):
            # 初始化子代个体
            c = offspring[n]
            r1, r2, r3 = np.random.choice([i for i in range(N) if i != n], 3, replace=False)
            # 进行差分进化操作
            c.rnvec = DERand1(population[n].rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                              len(population[n].rnvec),
                              self.F, self.CR)

        return offspring

    def MML(self, pop_t, pop_s):
        """
        多层次学习策略（MLL）

        :param pop_t: 目标种群。
        :param pop_s: 源种群。
        :return:
            cp_t: 目标种群的候选亲本（各层次转移个体）。
            cp_s: 源种群的候选亲本（知识库来源）。
            pop_t_level: 划分为三个层次的亲本种群。
        """
        # 将目标种群和源种群按照适应度排序
        pop_t = sorted(pop_t, key=lambda x: x.obj)
        pop_s = sorted(pop_s, key=lambda x: x.obj)
        # 各层次的个体数
        num = int(Problem.N / 3)
        # 选取最优的S个个体作为知识转移的源种群
        cp_s = pop_s[:self.S]
        # 选取每个层次前nt个个体作为知识转移的目标种群
        cp_t = []
        # 划分为三个层次的亲本种群
        pop_t_level = []
        # 遍历三个层次
        for i in range(3):
            # 划分层次
            pop_t_level.append(pop_t[i * num:(i + 1) * num])
            # 选取各层次的转移个体
            cp_t.append(pop_t[i * num:i * num + self.nt])

        return cp_t, cp_s, pop_t_level

    def MSL(self, cp_t, cp_s, weight_t):
        """
        多分段学习策略（MSL）

        :param cp_t: 目标种群的候选亲本（各层次转移个体）。
        :param cp_s: 源种群的候选亲本（知识库来源）。
        :return: childrens: 三个层次的转移子代。
        """
        # 三个层次的转移子代
        childrens = [[] for _ in range(3)]
        # 遍历目标种群的各个层次（层次内进化）
        for index, cp_t_list in enumerate(cp_t):
            # 遍历每个层次的转移个体
            for ind in cp_t_list:
                # 随机选取进化池算子
                operator_idxs = RouletteWheelSelection([0.2, 0.3, weight_t])
                operator_index = operator_idxs[0]
                p = np.random.rand()
                # 随机选择片段长度
                rL = np.random.randint(self.L[0], self.L[1])
                # 个体维度
                Dt = len(ind.rnvec)

                # 转移子代
                offspring = Individual()
                offspring.rnvec = np.zeros(Dt)
                # 分割个体（存储所有分割片段，不可重叠）
                segs = []
                for i in range(int(np.ceil(Dt / rL))):
                    segs.append(ind.rnvec[i * rL:min((i + 1) * rL, Dt)])
                # 构建知识库
                kpool = self.knowPool(cp_s, rL)
                # 遍历每个片段
                for idx, seg in enumerate(segs):
                    # 片段长度不足时，重新构建知识库
                    if len(seg) < rL:
                        kpool = self.knowPool(cp_s, len(seg))

                    # 计算每个片段与知识库中个体的欧式距离
                    distances = [np.linalg.norm(seg - k) for k in kpool]
                    # 选择距离最近的个体，构成"小生境"
                    min_index = distances.index(min(distances))

                    # 交叉操作（交叉算子随机选取）
                    # if operator_index == 1:
                    if p <= 0.3:
                        off1, off2 = BinomialCrossover(seg, kpool[min_index], len(seg))  # 二项式交叉
                    # elif operator_index == 2:
                    elif p <= 0.5:
                        off1, off2 = SBX(seg, kpool[min_index])  # 模拟二进制交叉
                    else:
                        off1 = gaussian_mutation(seg, 0.01 * np.linalg.norm(seg - kpool[min_index]) / len(seg))  # 高斯变异
                        off2 = gaussian_mutation(kpool[min_index],
                                                 0.01 * np.linalg.norm(seg - kpool[min_index]) / len(seg))  # 高斯变异

                    # 组合转移子代(偏向于seg)
                    offspring.rnvec[idx * rL:min((idx + 1) * rL, Dt)] = off1 if np.random.rand() < 0.65 else off2

                # 添加各层次的转移子代
                childrens[index].append(offspring)

        return childrens

    @staticmethod
    def knowPool(cp_s, rL):
        # 从源种群中构建知识库
        kpool = []
        # 遍历源种群的每个个体
        for ind in cp_s:
            # 分割个体（可重叠）
            for i in range(len(ind.rnvec) - rL + 1):
                kpool.append(ind.rnvec[i:i + rL])

        return kpool


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
        result = MMLMTO().run(Prob, True)
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
