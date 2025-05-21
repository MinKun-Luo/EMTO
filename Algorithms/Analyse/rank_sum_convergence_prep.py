# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 显著性分析和收敛图绘制的数据预处理
# @Remind: 运行前请确保路径准确且对应算法存在相应数据
import sys
import os
from pathlib import Path

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

import pandas as pd
from openpyxl.styles import Font, Border, Side, Alignment

from Algorithms.Analyse.convergence_plot import convergence_plot
from Algorithms.Analyse.wilcoxon_rank_sum import wilcoxon_rank_sum
from Algorithms.Run.output import set_project_root

# 配置变量
project_root = "EMTO"  # 项目根目录
data_options = ["17_10", "17_20", "22_10", "22_20"]  # 可选数据来源
# 文件处理顺序
file_order_17 = ['CI_HS', 'CI_MS', 'CI_LS', 'PI_HS', 'PI_MS', 'PI_LS', 'NI_HS', 'NI_MS', 'NI_LS']
file_order_22 = ['Benchmark1', 'Benchmark2', 'Benchmark3', 'Benchmark4', 'Benchmark5',
                 'Benchmark6', 'Benchmark7', 'Benchmark8', 'Benchmark9', 'Benchmark10']

# 比较算法数据位置(该顺序决定了显著性分析和收敛图绘制的顺序，第一个为自己的算法)
folder_paths_template = [
    "Files/MultiTask/MGDMTO_s_CR0.7/CEC/CEC{data}/",
    "Files/MultiTask/MFEA/CEC/CEC{data}/",
    "Files/MultiTask/MFEA_AKT/CEC/CEC{data}/",
    "Files/MultiTask/MFDE/CEC/CEC{data}/",
    "Files/MultiTask/MTGA/CEC/CEC{data}/",
    "Files/MultiTask/MKTDE/CEC/CEC{data}/",
    "Files/MultiTask/AEMTO/CEC/CEC{data}/",
    "Files/MultiTask/RLMFEA/CEC/CEC{data}/",
    "Files/MultiTask/EMTO_AI/CEC/CEC{data}/",
    "Files/MultiTask/BLKT_DE/CEC/CEC{data}/",
    "Files/MultiTask/MMLMTO/CEC/CEC{data}/",
]
# 对比算法名称(该顺序决定了显著性分析和收敛图绘制的顺序，须于上面一致)
algos = ['MGDMTO_s_CR0.7', 'MFEA', 'MFEA_AKT','MFDE', 'MTGA', 'MKTDE', 'AEMTO', 'RLMFEA', 'EMTO_AI', 'BLKT_DE', 'MMLMTO']

file_prefix = "Files/MultiTask/MGDMTO显著性分析/CEC"  # 文件前缀
data_file_template = "{file_prefix}/Data/CEC{data}_temp.xlsx"  # 显著性分析输出文件
plot_file_template = "{file_prefix}/Plot/CEC{data}_temp.xlsx"  # 收敛图输出文件
# Excel格式化设置
border = Border(left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'), bottom=Side(style='thin'))
title_font = Font(name='Times New Roman', bold=True)
cell_font = Font(name='Times New Roman')
alignment = Alignment(horizontal='center', vertical='center')
number_format = '0.00E+00'


def select_data_source():
    """
    提示用户选择数据来源，返回选中的值

    :return: str，选中的数据来源值
    """
    print("请选择数据来源：")
    for idx, option in enumerate(data_options, 1):
        print(f"{idx}. {option}")
    print(f"请输入编号（1-{len(data_options)}），或按回车使用默认值 {data_options[0]}：")

    try:
        choice = input().strip()
        if choice == "":
            return data_options[0]  # 默认值
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(data_options):
            return data_options[choice_idx]
        else:
            print(f"无效输入，使用默认值 {data_options[0]}")
            return data_options[0]
    except ValueError:
        print(f"输入格式错误，使用默认值 {data_options[0]}")
        return data_options[0]


def setup_environment(data_file, plot_file):
    """
    设置项目根目录并创建输出目录

    :param data_file: str，显著性分析输出文件路径
    :param plot_file: str，收敛图输出文件路径
    """
    set_project_root(project_root)
    os.makedirs(data_file, exist_ok=True)
    os.makedirs(plot_file, exist_ok=True)


def get_file_order(data_prefix):
    """
    根据数据前缀返回文件顺序

    :param data_prefix: str，数据前缀（例如 "17" 或 "22"）
    :return: list，文件顺序列表
    :raises ValueError: 如果数据前缀无效
    """
    if data_prefix == "17":
        return file_order_17
    elif data_prefix == "22":
        return file_order_22
    else:
        raise ValueError("无效的数据前缀，请检查输入！")


def process_excel_file(file_path, file_idx):
    """
    处理单个Excel文件，提取最后一列和最后一行数据

    :param file_path: str，Excel文件路径
    :param file_idx: int，文件索引
    :return:
        last_columns_data: list，包含最后一列数据的DataFrame列表
        last_rows_data: list，包含最后一行数据的DataFrame列表
    """
    last_columns_data = []
    last_rows_data = []

    if os.path.exists(file_path):
        excel_file = pd.ExcelFile(file_path)
        for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            col_name = f"P{file_idx + 1}-T{sheet_idx + 1}"

            # 提取最后一列，去掉NaN值
            last_column = df.iloc[:, -1].dropna()
            last_columns_data.append(pd.DataFrame({col_name: last_column}).reset_index(drop=True))

            # 提取最后一行（不含平均值列），去掉NaN值
            last_row = df.iloc[-1, :-1].dropna()
            last_rows_data.append(pd.DataFrame({col_name: last_row}).reset_index(drop=True))

    return last_columns_data, last_rows_data


def process_and_save_excel(data, folder_paths, data_file, plot_file):
    """
    处理Excel数据并保存到输出文件，同时应用格式化

    :param data: str，数据来源（例如 "17_10"）
    :param folder_paths: list，文件夹路径列表
    :param data_file: str，显著性分析输出文件路径
    :param plot_file: str，收敛图输出文件路径
    """
    file_order = get_file_order(data[:2])

    # 创建两个ExcelWriter对象，用于存储列数据和行数据
    with pd.ExcelWriter(plot_file, engine='openpyxl') as plot_writer, \
            pd.ExcelWriter(data_file, engine='openpyxl') as data_writer:

        # 遍历每个文件夹路径
        for folder_idx, folder_path in enumerate(folder_paths):
            last_columns_data = []
            last_rows_data = []

            # 遍历指定顺序的文件
            for file_idx, filename in enumerate(file_order):
                full_filename = f"{filename}.xlsx"
                file_path = os.path.join(folder_path, full_filename)
                col_data, row_data = process_excel_file(file_path, file_idx)
                last_columns_data.extend(col_data)
                last_rows_data.extend(row_data)

            # 获取工作表名称
            sheet_name = algos[folder_idx]

            # 保存最后一列数据
            if last_columns_data:
                pd.concat(last_columns_data, axis=1).to_excel(plot_writer, sheet_name=sheet_name, index=False)
                print(f"最后一列数据已保存至 {plot_file} 的 {sheet_name} 工作簿")

            # 保存最后一行数据
            if last_rows_data:
                pd.concat(last_rows_data, axis=1).to_excel(data_writer, sheet_name=sheet_name, index=False)
                print(f"最后一行数据已保存至 {data_file} 的 {sheet_name} 工作簿")

            # 应用格式化到 plot_file
            for sheet_name in plot_writer.book.sheetnames:
                worksheet = plot_writer.book[sheet_name]
                for row in worksheet.rows:
                    for cell in row:
                        cell.border = border
                        cell.alignment = alignment
                        cell.font = title_font if cell.row == 1 else cell_font
                        cell.number_format = 'General' if cell.row == 1 else (
                            number_format if isinstance(cell.value, (int, float)) else 'General')

            # 应用格式化到 data_file
            for sheet_name in data_writer.book.sheetnames:
                worksheet = data_writer.book[sheet_name]
                for row in worksheet.rows:
                    for cell in row:
                        cell.border = border
                        cell.alignment = alignment
                        cell.font = title_font if cell.row == 1 else cell_font
                        cell.number_format = 'General' if cell.row == 1 else (
                            number_format if isinstance(cell.value, (int, float)) else 'General')

    print(f"处理完成！收敛图绘制数据保存至 {plot_file}，显著性分析数据保存至 {data_file}")


def main():
    """
    脚本入口函数
    """
    # 选择数据来源
    data = select_data_source()

    # 根据所选数据更新路径和文件
    folder_paths = [path.format(data=data) for path in folder_paths_template]
    data_file = data_file_template.format(data=data, file_prefix=file_prefix)
    plot_file = plot_file_template.format(data=data, file_prefix=file_prefix)

    setup_environment(f"{file_prefix}/Data", f"{file_prefix}/Plot")
    process_and_save_excel(data, folder_paths, data_file, plot_file)

    # 计算增量
    delta = 5000 if data[-2:] == "10" else 10000
    # 绘制收敛图和进行显著性分析
    convergence_plot(plot_file, plot_file.replace("_temp", ""), delta=delta)
    wilcoxon_rank_sum(data_file, data_file.replace("_temp", ""))


if __name__ == "__main__":
    main()
