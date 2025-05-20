# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wcl、wzb
# @Introduction: CEC22(WCCI20/22) 多任务优化问题集
# @Remind: 默认评估次数为200000，种群大小为50

import os

import numpy as np

from Problems.Problem import Problem


class Task(Problem):
    def __init__(self, lb=0, ub=1, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(D, lb, ub)
        # 如果提供了偏置，则使用提供的偏置，否则初始化为零向量
        if bias is not None:
            self.center = bias
        else:
            self.center = np.zeros(shape=D)
        # 如果提供了系数矩阵，则使用提供的系数矩阵，否则初始化为单位矩阵
        if coeffi is not None:
            self.M = coeffi
        else:
            self.M = np.zeros(shape=(D, D))
            for i in range(D):
                self.M[i, i] = 1
        # 设置shuffle参数
        self.shuffle = shuffle
        # 设置缩放率
        self.sh_rate = sh_rate

    def decode(self, X):
        # 将输入X从[0, 1]范围映射到[lb, ub]范围
        X1 = self.lb + (self.ub - self.lb) * X
        # 截取前dim个元素
        X1 = X1[0:self.dim]
        # 减去中心偏置（偏置）
        X1 = X1 - self.center
        # 乘以缩放率（放缩）
        X1 = X1 * self.sh_rate
        # 返回经过系数矩阵M变换后的结果（旋转）
        return np.dot(self.M, X1.T)


# 函数子类

class Ellips(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X
        indexs = np.arange(self.dim)
        return np.sum((10 ** (6 * indexs / (self.dim - 1))) * temp * temp)

    def Info(self):
        return 'Ellips ' + str(self.dim)


class Discus(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X
        return temp[0] * temp[0] * (10 ** 6) + np.sum(temp[1:] * temp[1:])

    def Info(self):
        return 'Discus ' + str(self.dim)


class Rosenbrock(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=2.048 / 100.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X * self.sh_rate
        return np.sum(100 * ((temp[:self.dim - 1] ** 2 - temp[1:]) ** 2) + (temp[:self.dim - 1] - 1) ** 2)

    def Info(self):
        return 'Rosenbrock ' + str(self.dim)


class Ackley(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(temp ** 2) / self.dim)) - np.exp(
            np.sum(np.cos(2 * np.pi * temp)) / self.dim) + 500

    def Info(self):
        return 'Ackley ' + str(self.dim)


class Weierstrass(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=0.5 / 100):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X * self.sh_rate
        a = 0.5
        b = 3
        kmax = 20
        sums = 0
        for k in range(0, kmax + 1):
            sums += np.sum(a ** k * np.cos(2 * np.pi * b ** k * (temp + 0.5))) - self.dim * a ** k * np.cos(
                np.pi * b ** k)
        return sums + 600

    def Info(self):
        return 'Weierstrass ' + str(self.dim)


class Griewank(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=600.0 / 100):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X * self.sh_rate
        return 1 + np.sum(temp ** 2) / 4000 - np.prod(np.cos(temp / np.sqrt(np.arange(1, self.dim + 1)))) + 700

    def Info(self):
        return 'Griewank ' + str(self.dim)


class Rastrigin(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=5.12 / 100):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X)
        else:
            temp = X * self.sh_rate
        return np.sum(temp ** 2 - 10 * np.cos(2 * np.pi * temp) + 10)

    def Info(self):
        return 'Rastrigin ' + str(self.dim)


class Schwefel(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1000.0 / 100):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            z = self.decode(X) + 4.209687462275036e+002
        else:
            temp = X * self.sh_rate
            z = temp + 4.209687462275036e+002
        tmp = z.copy()
        z[z < -500] = -500 + np.fmod(np.abs(z[z < -500]), 500)
        z[z > 500] = 500 - np.fmod(np.abs(z[z > 500]), 500)
        return 4.189828872724338e+002 * self.dim - np.sum(z * np.sin(np.sqrt(np.abs(z)))) + np.sum(
            (tmp[tmp < -500] + 500) ** 2 / 10000 / self.dim) + np.sum(
            (tmp[tmp > 500] - 500) ** 2 / 10000 / self.dim) + 1100

    def Info(self):
        return 'Schwefel ' + str(self.dim)


class Katsuura(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            z = self.decode(X)
        else:
            z = X * self.sh_rate
        nx = self.dim
        f = 1.0
        tmp3 = np.power(1.0 * nx, 1.2)
        for i in range(nx):
            temp = 0.0
            for j in range(1, 33):
                tmp1 = np.power(2.0, j)
                tmp2 = tmp1 * z[i]
                temp += np.abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
            f *= np.power(1.0 + (i + 1) * temp, 10.0 / tmp3)
        tmp1 = 10.0 / nx / nx
        f = f * tmp1 - tmp1
        return f

    def Info(self):
        return 'Katsuura ' + str(self.dim)


class GrieRosen(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X) + 1
        else:
            temp = X * self.sh_rate + 1
        temp1 = np.append(temp[1:], temp[0])
        temp1 = 100 * (temp * temp - temp1) * (temp * temp - temp1) + (temp - 1) * (temp - 1)
        return np.sum(temp1 * temp1 / 4000 - np.cos(temp1) + 1) + 1500

    def Info(self):
        return 'Grie_Rosen ' + str(self.dim)


class Escaffer6(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            z = self.decode(X)
        else:
            z = X
        nx = self.dim
        f = 0.0
        for i in range(nx - 1):
            temp1 = np.sin(np.sqrt(z[i] ** 2 + z[i + 1] ** 2))
            temp1 = temp1 ** 2
            temp2 = 1.0 + 0.001 * (z[i] ** 2 + z[i + 1] ** 2)
            f += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        temp1 = np.sin(np.sqrt(z[nx - 1] ** 2 + z[0] ** 2))
        temp1 = temp1 ** 2
        temp2 = 1.0 + 0.001 * (z[nx - 1] ** 2 + z[0] ** 2)
        f += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        return f + 1600

    def Info(self):
        return 'Escaffer6 ' + str(self.dim)


class HappyCat(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X) - 1
        else:
            temp = X * self.sh_rate - 1
        r2 = np.sum(temp * temp)
        sum_z = np.sum(temp)
        return np.abs(r2 - self.dim) ** (1 / 4) + (0.5 * r2 + sum_z) / self.dim + 0.5 + 1300

    def Info(self):
        return 'Happycat ' + str(self.dim)


class Hgbat(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=5.0 / 100.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)

    def fnc(self, X, tag=False):
        if not tag:
            temp = self.decode(X) - 1
        else:
            temp = X * self.sh_rate - 1
        return np.sqrt(np.abs((np.sum(temp * temp) ** 2 - np.sum(temp) ** 2))) + (
                np.sum(temp * temp) / 2 + np.sum(temp)) / self.dim + 0.5

    def Info(self):
        return 'Hgbat ' + str(self.dim)


class Hf01(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)
        self.Sch = Schwefel(D=15)
        self.Ras = Rastrigin(D=15)
        self.Elp = Ellips(D=20)

    def fnc(self, X):
        x = self.decode(X)
        nx = self.dim
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Sch.fnc(temp[:15], tag=True)
        func += self.Ras.fnc(temp[15:30], tag=True)
        func += self.Elp.fnc(temp[30:], tag=True)
        return func + 600

    def Info(self):
        return 'Hf01 ' + str(self.dim)


class Hf04(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)
        self.Hg = Hgbat(D=10)
        self.Dis = Discus(D=10)
        self.GR = GrieRosen(D=15)
        self.Ras = Rastrigin(D=15)

    def fnc(self, X):
        x = self.decode(X)
        var = self.dim
        temp = np.zeros(var)
        for i in range(var):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Hg.fnc(temp[:10], tag=True)
        func += self.Dis.fnc(temp[10:20], tag=True)
        func += self.GR.fnc(temp[20:35], tag=True)
        func += self.Ras.fnc(temp[35:], tag=True)
        return func + 500

    def Info(self):
        return 'Hf04 ' + str(self.dim)


class Hf05(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)
        self.Es = Escaffer6(D=5)
        self.Hg = Hgbat(D=10)
        self.Ros = Rosenbrock(D=10)
        self.Sch = Schwefel(D=10)
        self.Elp = Ellips(D=15)

    def fnc(self, X):
        x = self.decode(X)
        nx = self.dim
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Es.fnc(temp[:5], tag=True)
        func += self.Hg.fnc(temp[5:15], tag=True)
        func += self.Ros.fnc(temp[15:25], tag=True)
        func += self.Sch.fnc(temp[25:35], tag=True)
        func += self.Elp.fnc(temp[35:], tag=True)
        return func - 600

    def Info(self):
        return 'Hf05 ' + str(self.dim)


class Hf06(Task):
    def __init__(self, lb=-100, ub=100, D=50, coeffi=None, bias=None, shuffle=None, sh_rate=1.0):
        super().__init__(lb, ub, D, coeffi, bias, shuffle, sh_rate)
        self.Kat = Katsuura(D=5)
        self.HC = HappyCat(D=10)
        self.GR = GrieRosen(D=10)
        self.Sch = Schwefel(D=10)
        self.Ack = Ackley(D=15)

    def fnc(self, X):
        x = self.decode(X)
        nx = self.dim
        temp = np.zeros(nx)
        for i in range(nx):
            temp[i] = x[int(self.shuffle[i]) - 1]
        func = 0
        func += self.Kat.fnc(temp[:5], tag=True)
        func += self.HC.fnc(temp[5:15], tag=True)
        func += self.GR.fnc(temp[15:25], tag=True)
        func += self.Sch.fnc(temp[25:35], tag=True)
        func += self.Ack.fnc(temp[35:], tag=True)
        return func - 2200

    def Info(self):
        return 'Hf06 ' + str(self.dim)


path = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径


# 读取文件
def GetMatrixs(filepath):
    filepath = path + filepath  # 构建文件的完整路径
    bias1 = np.loadtxt(filepath + 'bias_1')
    bias2 = np.loadtxt(filepath + 'bias_2')
    matrix1 = np.loadtxt(filepath + 'matrix_1')
    matrix2 = np.loadtxt(filepath + 'matrix_2')
    return {'bias1': bias1, 'bias2': bias2, 'ma1': matrix1, 'ma2': matrix2}


def GetShuffle(filepath):
    filepath = path + filepath  # 构建文件的完整路径
    shuffle = np.loadtxt(filepath)
    return {'shuffle': shuffle}


# 10个基本问题
def Benchmark1(filename='/Tasks/benchmark_1/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Weierstrass(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Weierstrass(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark2(filename='/Tasks/benchmark_2/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Griewank(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Griewank(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark3(filename='/Tasks/benchmark_3/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_17_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf01(coeffi=params['ma1'], bias=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf01(coeffi=params['ma2'], bias=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark4(filename='/Tasks/benchmark_4/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = HappyCat(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = HappyCat(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark5(filename='/Tasks/benchmark_5/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = GrieRosen(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = GrieRosen(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark6(filename='/Tasks/benchmark_6/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_21_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf05(coeffi=params['ma1'], bias=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark7(filename='/Tasks/benchmark_7/'):
    params = GetMatrixs(filename)
    s = GetShuffle('/Tasks/shuffle/shuffle_data_22_D50.txt')
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Hf06(coeffi=params['ma1'], bias=params['bias1'], shuffle=s['shuffle'])
    Task2 = Hf06(coeffi=params['ma2'], bias=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark8(filename='/Tasks/benchmark_8/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Ackley(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Ackley(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark9(filename='/Tasks/benchmark_9/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    Task1 = Schwefel(coeffi=params['ma1'], bias=params['bias1'])
    Task2 = Escaffer6(coeffi=params['ma2'], bias=params['bias2'])
    Probs = [Task1, Task2]
    return Probs


def Benchmark10(filename='/Tasks/benchmark_10/'):
    params = GetMatrixs(filename)
    Problem.maxFE = 100 * 2000
    Problem.T = 2
    s = GetShuffle('/Tasks/shuffle/shuffle_data_20_D50.txt')
    Task1 = Hf04(coeffi=params['ma1'], bias=params['bias1'], shuffle=s['shuffle'])
    s = GetShuffle('/Tasks/shuffle/shuffle_data_21_D50.txt')
    Task2 = Hf05(coeffi=params['ma2'], bias=params['bias2'], shuffle=s['shuffle'])
    Probs = [Task1, Task2]
    return Probs
