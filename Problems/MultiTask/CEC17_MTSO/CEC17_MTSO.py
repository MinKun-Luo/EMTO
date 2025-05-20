# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: CEC17 多任务优化问题集
# @Remind: 默认评估次数为100000，种群大小为50

import os

import scipy.io as sio

from Problems.Base import *
from Problems.Problem import Problem


def mat2python(filename, flags):
    path = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径
    file = path + filename  # 构建文件的完整路径
    data = sio.loadmat(file)  # 使用 scipy.io.loadmat 函数加载 .mat 文件
    names = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']  # 定义参数名称列表
    parameters = []  # 初始化参数列表
    for i, flag in enumerate(flags):  # 遍历 flags 列表
        if flag is not None:  # 如果 flag 不为 None
            name = names[i]  # 获取对应的参数名称
            parameters.append(data[name])  # 从 Data 中提取参数并添加到 parameters 列表
        else:
            parameters.append(None)  # 如果 flag 为 None，添加 None 到 parameters 列表
    return parameters  # 返回提取的参数列表


def CI_HS(filename=r'/Tasks/CI_H.mat'):
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Griewank(M=params[2], opt=params[0], dim=50, lb=-100, ub=100, g=0)
    Task2 = Rastrigin(M=params[3], opt=params[1], dim=50, lb=-50, ub=50, g=0)
    Probs = [Task1, Task2]
    return Probs


def CI_MS(filename=r'/Tasks/CI_M.mat'):
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Ackley(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Rastrigin(M=params[3], opt=params[1], dim=50, lb=-50, ub=50, g=0)
    Probs = [Task1, Task2]
    return Probs


def CI_LS(filename=r'/Tasks/CI_L.mat'):
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Ackley(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Schwefel(M=1, opt=0, dim=50, lb=-500, ub=500, g=0)
    Probs = [Task1, Task2]
    return Probs


def PI_HS(filename=r'/Tasks/PI_H.mat'):
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Rastrigin(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Sphere(M=1, opt=params[1], dim=50, lb=-100, ub=100, g=0)
    Probs = [Task1, Task2]
    return Probs


def PI_MS(filename=r'/Tasks/PI_M.mat'):
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Ackley(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Rosenbrock(M=1, opt=0, dim=50, lb=-50, ub=50, g=0)
    Probs = [Task1, Task2]
    return Probs


def PI_LS(filename=r'/Tasks/PI_L.mat'):
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Ackley(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Weierstrass(M=params[3], opt=params[1], dim=25, lb=-0.5, ub=0.5, g=0)
    Probs = [Task1, Task2]
    return Probs


def NI_HS(filename=r'/Tasks/NI_H.mat'):
    flags = [None, 'GO_Task2', None, 'Rotation_Task2']
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Rosenbrock(M=1, opt=0, dim=50, lb=-50, ub=50, g=0)
    Task2 = Rastrigin(M=params[3], opt=params[1], dim=50, lb=-50, ub=50, g=0)
    Probs = [Task1, Task2]
    return Probs


def NI_MS(filename=r'/Tasks/NI_M.mat'):
    flags = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Griewank(M=params[2], opt=params[0], dim=50, lb=-100, ub=100, g=0)
    Task2 = Weierstrass(M=params[3], opt=params[1], dim=50, lb=-0.5, ub=0.5, g=0)
    Probs = [Task1, Task2]
    return Probs


def NI_LS(filename=r'/Tasks/NI_L.mat'):
    flags = ['GO_Task1', None, 'Rotation_Task1', None]
    params = mat2python(filename, flags)
    Problem.maxFE = 100 * 1000
    Problem.T = 2
    Task1 = Rastrigin(M=params[2], opt=params[0], dim=50, lb=-50, ub=50, g=0)
    Task2 = Schwefel(M=1, opt=0, dim=50, lb=-500, ub=500, g=0)
    Probs = [Task1, Task2]
    return Probs
