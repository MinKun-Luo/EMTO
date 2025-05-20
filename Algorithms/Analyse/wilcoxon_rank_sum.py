# <-*--*--*--*- Coder -*--*--*--*--*->
# @Time: 2025/4/14 下午8:02
# @Author: wzb
# @Introduction: 使用Wilcoxon秩和检验比较不同算法性能，并输出结果到Excel文件（显著性分析）
# @Remind: 需注意文件命名格式

import numpy as np
import pandas as pd
import scipy.stats as ss
from openpyxl import Workbook
from openpyxl.styles import Font, Border, Side, Alignment

from Algorithms.Run.output import set_project_root

# 自动切换到项目根目录EMTO
project_root = set_project_root("EMTO")


def wilcoxon_rank_sum(file_path='', output_path=''):
    """
    使用Wilcoxon秩和检验比较不同算法性能，并输出结果到Excel文件（显著性分析）
    :param file_path: 输入文件路径
    :param output_path: 输出文件路径
    :return: Excel文件,显著性分析结果
    """
    # 读取文件
    excel_file = pd.ExcelFile(file_path)
    # 获取所有工作簿名称（问题名）
    sheet_names = excel_file.sheet_names

    # 读取第一个工作表，并将其每一列数据存入列表中（去除表头）
    df1 = pd.read_excel(file_path, sheet_name=sheet_names[0])
    columns_data1 = [df1[column].values for column in df1.columns]

    # 获取剩余工作表的名称并重新排序
    reordered_sheet_names = [sheet for sheet in sheet_names[1:] if sheet in sheet_names]

    results = []

    # 遍历每一个重新排序后的工作表
    for sheet_name in reordered_sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        columns_data2 = [df[column].values for column in df.columns]

        sheet_result = []
        result = np.zeros(3, dtype=int)

        # 对每一列数据进行秩和检验并判断p值
        for col1, col2 in zip(columns_data1, columns_data2):
            avg1 = "{:.2E}".format(np.mean(col2))
            p1 = ss.ranksums(col1, col2, alternative='two-sided')
            if p1.pvalue >= 0.05:
                result[1] += 1
                avg1 += '(≈)'
            else:
                p2 = ss.ranksums(col1, col2, alternative='greater')
                if p2.pvalue < 0.05:
                    result[2] += 1
                    avg1 += '(-)'
                else:
                    result[0] += 1
                    avg1 += '(+)'
            sheet_result.append(avg1)

        # 添加最终结果统计
        sheet_result.append(" / ".join(map(str, result.astype(int))))
        results.append(sheet_result)

    # 准备输出数据
    output = ["\t".join(sheet_names)]
    for i in range(len(columns_data1)):
        line = ["{:.2E}".format(np.mean(columns_data1[i]))]
        line.extend([results[j][i] for j in range(len(results))])
        output.append("\t".join(line))
    output.append("\t".join([" "] + [results[j][-1] for j in range(len(results))]))

    # 创建DataFrame
    output_df = pd.DataFrame([line.split('\t') for line in output], columns=sheet_names)

    # 使用openpyxl创建并美化Excel文件
    wb = Workbook()
    ws = wb.active

    # 写入数据
    for r_idx, row in enumerate(output_df.values, 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # 设置字体和边框
    font_normal = Font(name='Times New Roman', size=8)
    font_bold = Font(name='Times New Roman', size=8, bold=True)
    border = Border(left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin'))
    center_alignment = Alignment(horizontal='center', vertical='center')

    # 应用样式
    for row in ws.rows:
        for cell in row:
            cell.font = font_bold if cell.row == 1 else font_normal
            cell.border = border
            cell.alignment = center_alignment

    # 应用样式
    for row in ws.rows:
        for cell in row:
            cell.font = font_bold if cell.row == 1 else font_normal
            cell.border = border
            cell.alignment = center_alignment

    # 保存文件
    wb.save(output_path)

    # 打印至控制台
    for line in output:
        print(line)


if __name__ == "__main__":
    wilcoxon_rank_sum()
