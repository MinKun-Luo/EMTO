# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 个体基类，用于表示优化问题中的个体

import numpy as np


class Individual:
    def __init__(self):
        """
        初始化个体。

        :param self: 初始化方法的实例自身
        """
        self.rnvec = np.empty(0, dtype=float)  # 基因型，初始化为空NumPy数组
        self.obj = None  # 适应度，初始化为None
