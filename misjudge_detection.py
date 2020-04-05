import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

"""
模块说明
	作者：马世奇
	功能：计算各组主观评价结果的误判率
	完成时间：2019-10-23
"""

# FILE = "idle.xlsx"
# SAMPLE_NUM = 13
FILE = "const.xlsx"
SAMPLE_NUM = 16

wb = xlrd.open_workbook(FILE)
sheets = wb.sheet_names()		# 得到sheet名列表
combination_list = [c for c in combinations(range(1, SAMPLE_NUM+1), 3)]		# 计算组合数


# 规则转换
def rule(x):
	if x == 3:
		x = 0
	elif x == 2:
		x = -1
	return x


for sheet in range(len(sheets)):
	df = pd.read_excel(FILE,					# 读取excel文件
					sheet_name=sheet,			# 读取对应的sheet
					header=0, 					# 第0行为列索引
					index_col=0,				# 第0列为行索引
					names=None,					# 不定义DataFrame的名字
					encoding='utf8')
	deta_ijk = 0								# 初始化误判数
	for com_num in combination_list:
		(i, j, k) = com_num
		P_ijk = []
		for (a, b) in [(i, j), (j, k), (i, k)]:
			if np.isnan(df.loc[a, b]):
				p_ab = -rule(df.loc[b, a])
			else:
				p_ab = rule(df.loc[a, b])
			P_ijk.append(p_ab)
		deta = 0								# 没有误判，记为0；有误判记为1
		if (P_ijk[0] > 0 and P_ijk[1] >= 0) and P_ijk[2] <= 0:
			deta = 1
		elif (P_ijk[0] < 0 and P_ijk[1] <= 0) and P_ijk[2] >= 0:
			deta = 1
		elif P_ijk[0] == 0 and P_ijk[1] != P_ijk[2]:
			deta = 1
		deta_ijk += deta						# 误判数求和
	misjudge = deta_ijk/len(combination_list)
	print(sheets[sheet], len(combination_list))						# 表名
	print(deta_ijk)								# 误判数
	print(str(round(misjudge*100, 3))+' %')		# 误判率
	print('\n')
