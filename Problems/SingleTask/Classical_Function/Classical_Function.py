# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 单任务优化的古典函数问题集

from Problems.Base import *
from Problems.Problem import Problem


def Func_Ackley():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Ackley(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Elliptic():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Elliptic(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Griewank():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Griewank(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Rastrigin():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Rastrigin(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Rosenbrock():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Rosenbrock(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Schwefel():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Schwefel(M=1, opt=0, dim=50, lb=-500, ub=500, g=0)
    return [Task]


def Func_Schwefel2():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Schwefel2(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Sphere():
    Problem.N = 100
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Sphere(M=1, opt=0, dim=50, lb=-100, ub=100, g=0)
    return [Task]


def Func_Weierstrass():
    Problem.maxFE = Problem.N * 500
    Problem.T = 1
    Task = Weierstrass(M=1, opt=0, dim=50, lb=-0.5, ub=0.5, g=0)
    return [Task]
