"""
模块说明
	作者：马世奇
	功能：计算主观评价最终得分与客观参数间的相关系数
	完成时间：2019-10-24
"""
import pandas as pd


files = {"idle": 13, "onst": 16}
filename = "data/idle_unpca.xlsx"					# 改这里
file = filename.strip("data/").strip("_unpca.xlsx")


df = pd.read_excel(filename,						# 读取excel文件
						sheet_name="Sheet1",		# 读取对应的sheet
						header=0, 					# 第0行为列索引
						index_col=0,				# 第0列为行索引
						names=None,					# 不定义DataFrame的名字
						encoding='utf8')

df2 = df[0:files[file]]
print("Spearman:\n", df.corr('spearman').loc[:, ["Score"]])
