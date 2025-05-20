# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 并行运行多任务优化算法，保留运行结果至相应的Excel文件
# @Remind: 运行前请确保参数均调整完毕
import argparse
import sys
import os
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, Border, Side, Alignment


def set_project_root(target_folder_name: str = "EMTO") -> Path:
    """
    向上查找目标目录并切换为工作目录，同时返回该路径。
    可用于统一将当前目录切换至项目根目录。

    :param target_folder_name: 目标目录名称，默认为 "EMTO"
    :return: 查找到的目标目录路径（Path对象）
    :raises FileNotFoundError: 如果未找到目标目录
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == target_folder_name:
            sys.path.append(str(parent))
            os.chdir(parent)
            return parent
    raise FileNotFoundError(f"项目根目录 '{target_folder_name}' 未找到")


# 自动切换到项目根目录EMTO
project_root = set_project_root("EMTO")

# 算法导入
from Algorithms.MultiTask.MultiFactorial.MFEA.MFEA import MFEA  # MFEA 算法
from Algorithms.MultiTask.MultiFactorial.MFDE.MFDE import MFDE  # MFDE 算法
from Algorithms.MultiTask.MultiFactorial.MFEA_AKT.MFEA_AKT import MFEA_AKT  # MFEA_AKT 算法
from Algorithms.MultiTask.MultiFactorial.RLMFEA.RLMFEA import RLMFEA  # RLMFEA 算法
from Algorithms.MultiTask.MultiFactorial.EMTO_AI.EMTO_AI import EMTO_AI  # EMTO-AI 算法
from Algorithms.MultiTask.MultiPopulation.MTGA.MTGA import MTGA  # MTGA 算法
from Algorithms.MultiTask.MultiPopulation.MKTDE.MKTDE import MKTDE  # MKTDE 算法
from Algorithms.MultiTask.MultiPopulation.AEMTO.AEMTO import AEMTO  # AEMTO 算法
from Algorithms.MultiTask.MultiPopulation.BLKT_DE.BLKT_DE import BLKT_DE  # BLKT-DE 算法
from Algorithms.MultiTask.MultiPopulation.MMLMTO.MMLMTO import MMLMTO  # MMLMTO 算法
from Algorithms.MultiTask.MultiPopulation.NKTMTO.NKTMTO import NKTMTO # NKTMTO 算法
from Algorithms.MultiTask.MultiPopulation.MGDMTO.MGDMTO import MGDMTO # MGDMTO 算法

# 算法列表
Algos = [MFEA, MFDE, MFEA_AKT, RLMFEA, EMTO_AI, MTGA, MKTDE, AEMTO, BLKT_DE, MMLMTO, NKTMTO, MGDMTO]
strAlgos = ['MFEA', 'MFDE', 'MFEA-AKT', 'RLMFEA', 'EMTO-AI', 'MTGA', 'MKTDE', 'AEMTO', 'BLKT-DE', 'MMLMTO', 'NKTMTO', 'MGDMTO']
# 测试集导入
from Problems.MultiTask.CEC17_MTSO.CEC17_MTSO import *
from Problems.MultiTask.WCCI20_MTSO.WCCI20_MTSO import *
from Problems.MultiTask.Competitive_C2TOP.CEC17_MTSO_Competitive import *

# 设置参数
labs = ['CEC', 'STO', 'Comp', 'Param', 'ScaleTest', 'Real']  # 实验名称列表([0: CEC测试集, 1: STO, 2: 组件分析, 3: 参数调研, 4: 扩展性研究, 5: 真实世界问题])
labs_idx = 0  # 选择的实验索引
sub_exp_name = ''  # 具体实验问题名称(可为空，将作为保存文件的文件夹名称，故须符合文件夹命名规范)，表示具体实验下的具体问题名称(将'/'作为起始)，如：/没有策略1(组件分析)、/参数1/参数1取值(参数调研)、/真实世界问题(现实问题)、/扩展维度(扩展性问题)等
algo_idx = 11  # 算法索引([0: MFEA, 1: MFDE, 2: MFEA_AKT, 3: RLMFEA, 4: EMTO_AI, 5: MTGA, 6: MKTDE, 7: AEMTO, 8: BLKT_DE, 9: MMLMTO, 10: NKTMTO, 11：MGDMTO])
maxFE = 200000  # 最大函数评估次数
repeat = 30  # 重复运行次数
continuous_algebra = 30  # 连续代数，用于提取最优连续段
task_idx = 0  # 优化任务的索引
isUpdateExcel = True  # 是否更新保存到Excel文件
isPrint = True  # 是否打印运行过程
sample_points = 11  # 采样点数量
output_dir = f"Files/MultiTask/{strAlgos[algo_idx]}"  # 输出目录主路径

# 初始化问题集
Probs17 = [CI_HS(), CI_MS(), CI_LS(), PI_HS(), PI_MS(), PI_LS(), NI_HS(), NI_MS(), NI_LS()]
str17 = ['CI_HS', 'CI_MS', 'CI_LS', 'PI_HS', 'PI_MS', 'PI_LS', 'NI_HS', 'NI_MS', 'NI_LS']

Probs22 = [Benchmark1(), Benchmark2(), Benchmark3(), Benchmark4(), Benchmark5(), Benchmark6(), Benchmark7(),
           Benchmark8(), Benchmark9(), Benchmark10()]
str22 = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5', 'Benchmark6', 'Benchmark7', 'Benchmark8',
         'Benchmark9', 'Benchmark10']

case = 3  # 案例编号，用于选择C2TOP问题集
ProbsC2TOP = [C_CI_HS(case), C_CI_MS(case), C_CI_LS(case), C_PI_HS(case), C_PI_MS(case), C_PI_LS(case), C_NI_HS(case),
              C_NI_MS(case), C_NI_LS(case)]
strC2TOP = ['C_CI_HS', 'C_CI_MS', 'C_CI_LS', 'C_PI_HS', 'C_PI_MS', 'C_PI_LS', 'C_NI_HS', 'C_NI_MS', 'C_NI_LS']


def run_algorithm(prob, name: str) -> list[np.ndarray]:
    """
    运行多任务优化算法并收集结果。

    :param prob: 多任务优化问题实例
    :param name: 问题名称，用于输出和日志
    :return: 每个任务的采样结果数组列表（每个元素 shape: (repeat, sample_points)）
    """
    Problem.maxFE = maxFE
    T = [np.zeros((repeat, sample_points)) for _ in range(Problem.T)]
    print(f"{name} initialization completed!")
    for i in range(repeat):
        result = Algos[algo_idx]().run(prob, isPrint=isPrint)  # 重新初始化算法实例，并运行
        gen_best = result.Result

        for t in range(Problem.T):
            gen_best_array = np.array(gen_best[t])
            indices = np.linspace(0, len(gen_best_array) - 1, sample_points, dtype=int)
            T[t][i] = gen_best_array[indices]

        print(f"{name} run {i + 1} completed!")
        log_results(T, name, i + 1, Problem.T)

    print(f"{name} run completed!")
    return T


def log_results(T: list[np.ndarray], name: str, run_idx: int, num_tasks: int):
    """
    记录当前运行的每个任务最优值和平均值。

    :param T: 每个任务的采样结果数组列表
    :param name: 问题名称
    :param run_idx: 当前运行索引
    :param num_tasks: 任务数量
    """
    print(f"\n{name} Values of the previous {run_idx} generation:")
    for r in range(run_idx):
        best_values = [T[t][r, -1] for t in range(num_tasks)]
        print(" ".join(f"{value:.2E}" for value in best_values))

    print(f"{name} Average Values of the previous {run_idx} generation:")
    avg_values = [
        np.mean(T[t][:run_idx, -1]) for t in range(num_tasks)
    ]
    print(" ".join(f"{value:.2E}" for value in avg_values))
    print()


def extract_continuous_best(T: list[np.ndarray], task_idx: int) -> list[np.ndarray]:
    """
    提取指定任务的最优连续运行段。

    :param T: 每个任务的采样结果数组列表
    :param task_idx: 优化的任务索引
    :return: 裁剪后的结果（每个任务保留连续 best runs）
    """
    if continuous_algebra >= repeat:
        return T

    final_results = T[task_idx][:, -1]
    window_sums = np.convolve(final_results, np.ones(continuous_algebra), mode='valid')
    best_start_idx = np.argmin(window_sums)

    return [t[best_start_idx:best_start_idx + continuous_algebra] for t in T]


def compute_average_convergence(T: list[np.ndarray]) -> list[np.ndarray]:
    """
    计算每个任务的平均收敛曲线。

    :param T: 每个任务的采样结果数组列表
    :return: 添加平均收敛后的新列表
    """
    for t in range(len(T)):
        mean_curve = np.mean(T[t], axis=0)
        T[t] = np.vstack([T[t], mean_curve])
    return T


def save_to_excel(T: list[np.ndarray], name: str, num_tasks: int, testName: str) -> None:
    """
    将结果保存为格式化的Excel文件。

    :param T: 每个任务的采样结果数组列表
    :param name: 问题名称
    :param num_tasks: 任务数量
    :param testName: 测试机名称
    """
    output_path = f"{output_dir}/{labs[labs_idx]}{sub_exp_name}/{testName}_{int(Problem.maxFE / 10000)}"
    os.makedirs(output_path, exist_ok=True)
    output_path = f"{output_path}/{name}.xlsx"

    if not isUpdateExcel:
        print(f"{name} update failed!")
        return

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for t in range(num_tasks):
            df = pd.DataFrame(T[t].T)
            df.to_excel(writer, sheet_name=f'T{t + 1}', index=False)

        workbook = writer.book
        format_excel_sheets(workbook)

    print(f"{name} update completed! File has been saved to: {output_path}")


def format_excel_sheets(workbook):
    """
    为Excel工作表应用格式化样式。

    :param workbook: openpyxl工作簿对象
    """
    border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    title_font = Font(name='Times New Roman', bold=True)
    cell_font = Font(name='Times New Roman')

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        for row in worksheet.rows:
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal='center', vertical='center')
                cell.font = title_font if cell.row == 1 else cell_font
                cell.number_format = '0' if cell.row == 1 else '0.00E+00'


def select_problem(func_id: int) -> tuple:
    """
    根据功能编号选择问题及其名称。

    :param func_id: 功能编号（0-8: CEC17, 9-18: CEC22, 19-27: C2TOP）
    :return: 问题实例、问题名称
    """
    if func_id < 9:
        return Probs17[func_id], str17[func_id], 'CEC17'
    elif func_id < 19:
        return Probs22[func_id - 9], str22[func_id - 9], 'CEC22'
    elif func_id < 28:
        return ProbsC2TOP[func_id - 19], strC2TOP[func_id - 19], f"C2TOP_{case}"
    else:
        raise ValueError(f"无效的功能编号: {func_id}")


def main():
    """
    主函数，协调优化流程。
    """
    parser = argparse.ArgumentParser(description="运行指定多任务优化问题")
    parser.add_argument('--func', type=int, required=True, help='功能编号（0-27）')
    args = parser.parse_args()

    Prob, Name, testName = select_problem(args.func)

    T = run_algorithm(Prob, Name)
    T = extract_continuous_best(T, task_idx)
    T = compute_average_convergence(T)
    save_to_excel(T, Name, Problem.T, testName)


if __name__ == "__main__":
    main()
