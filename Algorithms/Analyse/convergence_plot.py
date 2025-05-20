# <-*--*--*--*- Coder -*--*--*--*--*->
# # @Time: 2025/4/14 下午8:02
# # @Author: wzb
# # @Introduction: 处理收敛图的绘制数据，问题作为工作簿，工作簿内包含不同算法不同进化阶段（根据评估次数均分为11个阶段）的最优值（收敛图绘制）
# # @Remind: 需注意文件命名格式

import pandas as pd
from openpyxl.styles import Font, Border, Side, Alignment

from Algorithms.Run.output import set_project_root

# 自动切换到项目根目录EMTO
project_root = set_project_root("EMTO")


def convergence_plot(file_path='', output_path='', delta=5000):
    """
    处理收敛图的绘制数据，问题作为工作簿，工作簿内包含不同算法不同进化阶段（根据评估次数均分为11个阶段）的最优值（收敛图绘制）
    :param file_path: 输入文件路径
    :param output_path: 输出文件路径
    :param delta: 收敛图首列数据增量（10万评估次数为5000，20万则为10000）
    :return: Excel文件,收敛图的绘制数据
    """

    # 读取所有工作簿
    xls = pd.ExcelFile(file_path)
    sheets = xls.sheet_names  # 获取所有工作簿名称（问题名）

    # 用于存储每个维度的数据
    dimension_data = {}

    # 遍历每个工作簿（问题）
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        # 遍历每一列（维度）
        for column in df.columns:
            if column not in dimension_data:
                dimension_data[column] = {}
            # 将该维度的数据存储，以问题名为键
            dimension_data[column][sheet] = df[column]

    # 创建新的 Excel 文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 遍历每个维度，生成对应的工作簿
        for dimension, data in dimension_data.items():
            # 创建一个新的 DataFrame，列名为问题名
            new_df = pd.DataFrame(data)
            # 计算行数
            num_rows = len(new_df)
            # 创建首列数据，用于绘图
            first_column = [i * delta for i in range(num_rows)]
            # 将列表转换为 pandas Series
            first_column_series = pd.Series(first_column)
            # 插入首列
            new_df.insert(0, 'FEs', first_column_series)
            # 将 DataFrame 写入新的工作簿，工作簿名为维度名
            new_df.to_excel(writer, sheet_name=dimension, index=False)

        # 应用格式化
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]

            # 定义边框样式
            border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )

            # 定义字体
            title_font = Font(name='Times New Roman', bold=True)
            cell_font = Font(name='Times New Roman')

            # 遍历所有单元格，应用边框、字体和对齐
            for row in worksheet.rows:
                for cell in row:
                    cell.border = border
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                    if cell.row == 1:  # 标题行 (例如 "P1-T1")
                        cell.font = title_font  # 加粗，Times New Roman
                        cell.number_format = 'General'  # 确保标题行显示为文本 (P1-T1)
                    else:  # 其他行
                        cell.font = cell_font  # 普通 Times New Roman
                        # 应用科学计数法，E后保留2位
                        if isinstance(cell.value, (int, float)) and cell.column != 1:  # 排除首列
                            cell.number_format = '0.00E+00'
                        elif cell.column == 1:  # 首列格式
                            cell.number_format = 'General'

    print(f"收敛图数据整理完成，已保存到 {output_path}")


if __name__ == "__main__":
    convergence_plot()
