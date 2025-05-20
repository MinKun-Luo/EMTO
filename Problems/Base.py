# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 基准函数
import numpy as np

from Problems.Problem import Problem


class BaseFunction(Problem):
    """基准函数的基类，处理通用逻辑"""

    def __init__(self, M, opt, dim, lb, ub, g=0):
        super().__init__(dim, lb, ub)
        self.M = M
        self.opt = opt
        self.g = g

    def preprocess(self, var):
        """
        预处理变量：解码、平移、旋转
        
        :param var: 输入的变量向量
        :return: 预处理后的变量
        """        # 处理标量 M 和 opt
        M = self.M * np.eye(self.dim) if np.isscalar(self.M) else self.M
        opt = self.opt * np.ones(self.dim) if np.isscalar(self.opt) else self.opt.flatten()

        # 解码并变换变量
        var = self.decode(var[:self.dim])
        var = np.dot(M, var - opt)  # 平移后旋转
        return var


class Ackley(BaseFunction):
    """Ackley 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 计算平方和与余弦和的均值
        avg_sum1 = np.mean(var ** 2)
        avg_sum2 = np.mean(np.cos(2 * np.pi * var))
        # 计算目标值
        obj = -20 * np.exp(-0.2 * np.sqrt(avg_sum1)) - np.exp(avg_sum2) + 20 + np.exp(1) + self.g
        return obj


class Griewank(BaseFunction):
    """Griewank 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 计算平方和与余弦积
        sum1 = np.sum(var ** 2)
        sum2 = np.prod(np.cos(var / np.sqrt(np.arange(1, self.dim + 1))))
        # 计算目标值
        obj = 1 + sum1 / 4000 - sum2 + self.g
        return obj


class Rastrigin(BaseFunction):
    """Rastrigin 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 向量化计算目标值
        obj = np.sum(var ** 2 - 10 * np.cos(2 * np.pi * var) + 10) + self.g
        return obj


class Rosenbrock(BaseFunction):
    """Rosenbrock 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 计算相邻变量的差平方和
        obj = np.sum(100 * (var[1:] - var[:-1] ** 2) ** 2 + (var[:-1] - 1) ** 2) + self.g
        return obj


class Schwefel(BaseFunction):
    """Schwefel 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 向量化计算目标值
        obj = 418.9829 * self.dim - np.sum(var * np.sin(np.sqrt(np.abs(var)))) + self.g
        return obj


class Schwefel2(BaseFunction):
    """Schwefel2 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 计算累积和的平方
        obj = np.sum(np.cumsum(var) ** 2) + self.g
        return obj


class Sphere(BaseFunction):
    """Sphere 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        # 计算平方和
        obj = np.sum(var ** 2) + self.g
        return obj


class Weierstrass(BaseFunction):
    """Weierstrass 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        a, b, kmax = 0.5, 3, 20
        k = np.arange(kmax + 1)
        # 对每个维度 i 求和，确保与原代码顺序一致
        obj = np.sum([
            np.sum(a ** k * np.cos(2 * np.pi * b ** k * (var[i] + 0.5)))
            for i in range(self.dim)
        ])
        # 常数项：对 k 求和，乘以维度
        obj -= self.dim * np.sum(a ** k * np.cos(2 * np.pi * b ** k * 0.5))
        obj += self.g
        return obj


class Elliptic(BaseFunction):
    """Elliptic 函数"""

    def fnc(self, var):
        var = self.preprocess(var)
        a = 1e6
        # 计算加权平方和
        if self.dim == 1:
            obj = var[0] ** 2 + self.g
        else:
            weights = a ** (np.arange(self.dim) / (self.dim - 1))
            obj = np.sum(weights * var ** 2) + self.g
        return obj
