if __name__ == "__main__":
    import sys
    sys.path.append("/Users/beita/Desktop/EMTO")
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
import numpy as np
from sklearn.cluster import KMeans


class MGDMTO(Algorithm):
    def __init__(self):
        Problem.N = 50
        super().__init__()
        self.top = int(Problem.N/2)
        # 动态nt的参数设置
        self.min_nt = 1  # 最小生产个体数量
        self.max_nt = 5  # 最大生产个体数量
        self.nt = np.random.randint(self.min_nt, self.max_nt + 1)  # 随机初始值
        self.F = 0.5
        self.CR = 0.9

    def update_nt(self, similarity):
        """
        根据相似度更新生产个体数量nt
        
        :param similarity: 两个任务的相似度(0-1之间)
        :return: 更新后的nt值
        """
        # 使用非线性映射，使得低相似度时也有一定概率，高相似度时不会过高
        # 使用sqrt(s)使得曲线在低相似度时增长较快，在高相似度时增长较缓
        if similarity <= 0:
            self.nt = self.min_nt
        else:
            alpha = 0.5  # 控制非线性程度，<1时低值增长较快，>1时高值增长较快
            nt_float = self.min_nt + (self.max_nt - self.min_nt) * (similarity ** alpha)
            self.nt = int(round(nt_float))  # 四舍五入取整
        
        return self.nt

    def run(self, Prob, isPrint):
        # 初始化种群
        population = Initialization(self, Prob, Individual, isPadding=True)
        
        # 获得相似度并初始化nt
        s = self.get_similarity(population=population)
        self.update_nt(s)
        print(f"当前相似度: {s:.4f}, 生产个体数量nt: {self.nt}")
        
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
            
            # 每次迭代更新相似度和nt
            s = self.get_similarity(population)
            self.update_nt(s)
            if self.Gen %100 == 0 :
                print(f"当前相似度: {s:.4f}, 生产个体数量nt: {self.nt}")
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

    def Generation(self, population, M, Sigma):
        offspring = [Individual() for _ in range(Problem.N)]
        for i in range(Problem.N - self.nt):
            p = population[i]
            r1, r2, r3 = np.random.choice(range(len(population)), 3, replace=False)
            offspring[i].rnvec = DERand1(p.rnvec, population[r1].rnvec, population[r2].rnvec, population[r3].rnvec,
                                        len(p.rnvec), self.F, self.CR)
            # 添加任务标签，从父个体继承
            offspring[i].task_id = p.task_id
    
        generated_data = np.random.multivariate_normal(M, Sigma, size=self.nt)
        count = 0
        for i in range(Problem.N - self.nt, Problem.N):
            offspring[i].rnvec = generated_data[count]
            offspring[i].rnvec = np.clip(offspring[i].rnvec, 0, 1)
            # 添加任务标签，与当前种群相同
            offspring[i].task_id = population[0].task_id if len(population) > 0 else 0
            count +=1
        return offspring
    
    def get_similarity(self, population):
        # 合并两个种群
        all_individuals = population[0] + population[1]
        total_count = len(all_individuals)
        target_count = total_count // 2  # 每个类的目标数量
        
        # 获得两个种群的染色体
        X = np.array([ind.rnvec for ind in all_individuals])
        estimator = KMeans(n_clusters=2)
        estimator.fit(X)  # 聚类
        label_pred = estimator.labels_  # 获取聚类标签
        
        # 计算每个类的个体数量
        cluster_0_indices = np.where(label_pred == 0)[0]
        cluster_1_indices = np.where(label_pred == 1)[0]
        
        cluster_0_count = len(cluster_0_indices)
        cluster_1_count = len(cluster_1_indices)
        
        # 平衡两个类的大小
        if cluster_0_count > cluster_1_count:
            # 需要将一些标签为0的个体移到类1
            larger_cluster = 0
            diff = cluster_0_count - target_count
        else:
            # 需要将一些标签为1的个体移到类0
            larger_cluster = 1
            diff = cluster_1_count - target_count
    
        if diff > 0:
            # 计算每个点到另一个类中心的距离
            centers = estimator.cluster_centers_
            if larger_cluster == 0:
                # 计算类0中个体到类1中心的距离
                larger_indices = cluster_0_indices
                target_center = centers[1]
            else:
                # 计算类1中个体到类0中心的距离
                larger_indices = cluster_1_indices
                target_center = centers[0]
            
            # 计算较大类中所有点到另一类中心的距离
            distances = []
            for idx in larger_indices:
                dist = np.linalg.norm(X[idx] - target_center)
                distances.append((idx, dist))
            
            # 按距离排序，选择最近的diff个点
            distances.sort(key=lambda x: x[1])
            points_to_move = [idx for idx, _ in distances[:diff]]
            
            # 修改标签
            for idx in points_to_move:
                label_pred[idx] = 1 - label_pred[idx]  # 切换标签（0->1 或 1->0）
    
        # 获取标签为0的所有个体
        cluster_0_individuals = [all_individuals[i] for i in range(len(all_individuals)) if label_pred[i] == 0]
        
        # 统计标签为0的个体中不同任务标签的数量
        task_count = {}
        for ind in cluster_0_individuals:
            task_id = ind.task_id
            if task_id in task_count:
                task_count[task_id] += 1
            else:
                task_count[task_id] = 1
        
        # 找出数量最少的任务标签及其数量
        min_task_count = min(task_count.values()) if task_count else 0

        # 有问题，当两个种群分开的时候，task_count = {0: 50, 1: 50}，导致min_task_count为50,计算s为2
        # min_task_count = min(task_count.values()) if task_count else 0
        # s = (min_task_count * 2)/Problem.N
        if min_task_count == 50:
            s = 0
            return s
        s = min_task_count*2 / Problem.N
        return s


def main():
    # 测试函数
    # Prob = PI_HS()
    Prob = Benchmark9()
    # 设置最大评估次数
    Problem.maxFE = 200000
    repeat = 3
    for i in range(repeat):
        print(f"第{i+1}次运行")
        MGDMTO().run(Prob, True)

if __name__ == "__main__":
    main()
    # 17用cr=0.7
    # 22用cr = 