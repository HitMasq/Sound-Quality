import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
模块说明
	作者：马世奇
	功能：计算各组主观评价结果的打分值
	完成时间：2019-10-24
"""

files = {"idle.xlsx": 13, "const.xlsx": 16}			# 怠速13个；匀速16个


def get_scores(file):
	"""
	:param file: 文件名
	:return: 文件中的sheet名，各评价者的打分结果
	"""
	results = []
	sheets = xlrd.open_workbook(file).sheet_names()	# 得到sheet名列表
	for sheet in range(len(sheets)):
		df = pd.read_excel(file,					# 读取excel文件
						sheet_name=sheet,			# 读取对应的sheet
						header=0, 					# 第0行为列索引
						index_col=0,				# 第0列为行索引
						names=None,					# 不定义DataFrame的名字
						encoding='utf8')
		df = df.sort_index(axis=1, ascending=True)	# 列排序
		df = df.sort_index(axis=0, ascending=True)	# 行排序
		arr_df = df.to_numpy()
		# arr_df[np.isnan(arr_df)] = 0.				# 空值nan为numpy.float64类型，将其替换为0
		# 规则转换
		arr_df[arr_df == 1.] = 6.					# A>B, -> 2
		arr_df[arr_df == 2.] = 4.					# A<B, -> 0
		arr_df[arr_df == 3.] = 5.					# A=B, -> 1
		arr_df[arr_df == 6.] = 2.
		arr_df[arr_df == 5.] = 1.
		arr_df[arr_df == 4.] = 0.
		arr = arr_df.copy()							# 复制二维数组，放在另一个地址，以确保不受转置转换的影响
		arr[np.isnan(arr)] = 0.

		t_arr = arr_df.transpose()					# 转置
		t_arr[t_arr == 2.] = 4.
		t_arr[t_arr == 0.] = 6.
		t_arr[t_arr == 6.] = 2.
		t_arr[t_arr == 4.] = 0.
		t_arr[np.isnan(t_arr)] = 0.

		add_arr = arr + t_arr						# 矩阵对应元素相加后列向求和
		result = (add_arr.sum(axis=0)+2)/2
		results.append(result)
	
	return sheets, np.array(results)


def get_total_scores(file):
	"""
	:param file: 文件名
	:return: 总的打分结果
	"""
	_, results = get_scores(file)
	score = np.mean(results, axis=0)
	return score


if __name__ == "__main__":
	input_file = "idle.xlsx"							# 只需这里修改idle或const
	sheets, results = get_scores(input_file)
	ind = np.arange(1, files[input_file]+1)
	df_scores = pd.DataFrame(results.T, index=ind, columns=sheets)
	df_scores.to_excel(input_file.strip('.xlsx') + '_score.xlsx')
