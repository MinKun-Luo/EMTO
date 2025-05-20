# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午9:12
# @Author: wzb
# @Introduction: 实现SHADE算法中F/CR的历史自适应调整机制

from collections import deque

import numpy as np

from Problems.Problem import Problem


class SHA:
    def __init__(self, H=100):
        """
        初始化 SHADE 自适应历史存储器

        :param H: int，记忆窗口大小，用于存储历史 F 和 CR
        """
        self.H = H
        self.MF = [deque([0.5] * H, maxlen=H) for _ in range(Problem.T)]  # 记录历史 F 值
        self.MCR = [deque([0.5] * H, maxlen=H) for _ in range(Problem.T)]  # 记录历史 CR 值
        self.failA = [deque(maxlen=Problem.N) for _ in range(Problem.T)]  # 存储失败个体

    def generate_f_cr(self, num, t):
        """
        为第 t 个任务的 num 个个体生成新的 F 和 CR

        :param num: int，需要生成的参数数量
        :param t: int，任务索引
        :return:
            F: numpy.ndarray，生成的变异因子
            CR: numpy.ndarray，生成的交叉概率
        """
        F = np.zeros(num)
        CR = np.zeros(num)
        for i in range(num):
            # 从记忆中随机选择，并加上扰动
            mem_idx = np.random.randint(0, self.H)
            CR[i] = np.clip(np.random.normal(self.MCR[t][mem_idx], 0.1), 0, 1)

            # 从柯西分布采样并修正
            f_trial = np.random.standard_cauchy() * 0.1 + self.MF[t][mem_idx]
            while f_trial <= 0:
                f_trial = np.random.standard_cauchy() * 0.1 + self.MF[t][mem_idx]
            F[i] = min(f_trial, 1.0)

        return F, CR

    def update_memory(self, population, offspring, F, CR, t):
        """
        基于成功子代更新第 t 个任务的历史 F 和 CR 记忆

        :param population: list，父代个体
        :param offspring: list，子代个体
        :param F: numpy.ndarray，使用的变异因子
        :param CR: numpy.ndarray，使用的交叉概率
        :param t: int，任务索引
        """
        # 初始化成功的F和CR(以及对应的权重w)
        SCR, SF, w = np.array([]), np.array([]), np.array([])

        for i in range(len(population)):
            # 如果子代适应度更好，则更新存档
            if offspring[i].obj < population[i].obj:
                # 将失败个体存入存档（先进先出队列）
                self.failA[t].append(population[i])
                # 更新成功的F和CR(以及对应的权重w)
                w = np.append(w, population[i].obj - offspring[i].obj)
                SCR = np.append(SCR, CR[i])
                SF = np.append(SF, F[i])

        # 如果有成功个体，则更新历史记忆
        if len(SCR) > 0:
            # 归一化权重
            w = w / np.sum(w)
            # 更新 CR 存档
            self.MCR[t].append(np.sum(SCR * w))
            # 更新 F 存档，使用加权平方和公式
            self.MF[t].append(np.sum(SF ** 2 * w) / np.sum(SF * w))
