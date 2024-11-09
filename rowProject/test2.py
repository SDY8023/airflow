import pandas as pd

# 读取XLSX文件
file_path = 'F:\Study\项目\道路项目\数据.xlsx'



# 读取XLSX文件
df = pd.read_excel(file_path,skiprows=range(1))
print(df["PQI"])
start_row = 1  # 从第2行开始读取
# 使用切片操作选择指定范围内的行，并将每列数据存储为列表
column_lists = {}
for col in range(len(df.columns)):
    column_data = list(df.iloc[start_row-1:, col])
    column_lists[f'{df.columns[col]}'] = column_data
print(column_lists)
# 输出每列数据的列表
for column, values in column_lists.items():
    print(f"{column}: {values}")