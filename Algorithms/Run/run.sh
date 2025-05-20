#!/bin/bash
# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: pwc
# @Introduction: 并行运行 output.py 脚本
# @Remind: 根据运行需要修改 seq 的范围

# seq 0-8: CEC17-MTSO
# seq 9-18: WCCI20-MTSO(CEC22-MTSO、WCCI22-MTSO)
for func_num in $(seq 0 18); do
  # 后台运行 output.py，并传递任务编号参数(形参)
  python output.py --func "$func_num" &
done