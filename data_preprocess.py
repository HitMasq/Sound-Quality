import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing


def standard_scaler(file_name):
	"""
	数据标准化函数
	:param 		file_name:		需要分析的文件
	:return:	x_train_scale	标准化训练集
				y_train			学习目标
				x_predict_scale	标准化预测集
	"""
	files = {"idle_": 13, "onst_": 16}
	file = file_name.strip("data/").strip("de.xlsx")
	df = pd.read_excel(file_name,
					   sheet_name="Sheet1",  	# 读取对应的sheet
					   header=0,  				# 第0行为列索引
					   index_col=0,  			# 第0列为行索引
					   names=None,  			# 不定义DataFrame的名字
					   encoding='utf8')
	data = df.to_numpy()
	x_train = data[:files[file], :-1]
	x_predict = data[files[file]:, :-1]
	std_scale = preprocessing.StandardScaler().fit(x_train)
	x_train_scale = std_scale.transform(x_train)				# 标准化训练集
	x_predict_scale = std_scale.transform(x_predict)			# 标准化预测集
	y_train = data[:files[file], -1]							# 学习目标

	return x_train_scale, y_train, x_predict_scale


def pro_min_max(file_name):
	"""
	数据归一化函数：
	:param 		file_name		需要分析的文件
	:return 	x_train_mm		归一化训练集
				y_train			学习目标，未归一化
				x_predict_mm	归一化预测集
	"""
	files = {"idle_": 13, "onst_": 16}
	file = file_name.strip("data/").strip("de.xlsx")
	df = pd.read_excel(file_name,
							sheet_name="Sheet1",		# 读取对应的sheet
							header=0, 					# 第0行为列索引
							index_col=0,				# 第0列为行索引
							names=None,					# 不定义DataFrame的名字
							encoding='utf8')
	data = df.to_numpy()
	x_train = data[:files[file], :-1]
	x_predict = data[files[file]:, :-1]
	min_max_scale = preprocessing.MinMaxScaler().fit(x_train)
	x_train_mm = min_max_scale.transform(x_train)
	x_predict_mm = min_max_scale.transform(x_predict)
	y_train = data[:files[file], -1]

	return x_train_mm, y_train, x_predict_mm


def pro_results(file):
	"""
	数据标准化函数：
	:param 	file: 			输入需要分析的数据文件
	:return x_scale:		正则化后的自变量
			y:				学习目标，未标准化，需要截取对应长度
	"""
	df = pd.read_excel(file,
							sheet_name="Sheet1",		# 读取对应的sheet
							header=0, 					# 第0行为列索引
							index_col=0,				# 第0列为行索引
							names=None,					# 不定义DataFrame的名字
							encoding='utf8')
	data = df.to_numpy()
	X = np.array(data[:, :-1])							# 特征值矩阵array
	y = data[:, -1]
	x_scale = preprocessing.scale(X)					# 正则化自变量

	return X, x_scale, y


def pca_results(file, n=4):
	"""
	PCA分析函数
	:param 	file: 			输入需要分析的数据文件
	:param 	n: 				主成分分析转换个数
	:return v_ratio:		方差贡献率
			x_components:	原始数据映射矩阵
			x_pca_scale		标准化后的主成分
	"""
	X, x_scale, y = pro_results(file)
	pca = PCA(n_components=n)
	pca.fit(X)
	v_ratio = pca.explained_variance_ratio_  		# 方差贡献率
	x_components = pca.components_  				# 原始变量映射矩阵
	x_pca = pca.fit_transform(X)  					# 主成分分析后的结果
	x_pca_scale = preprocessing.scale(x_pca)  		# 正则化主成分

	return v_ratio, x_components, x_pca_scale


if __name__ == "__main__":
	filename = "data/idle_unpca.xlsx"
	X_train_scale, Y_train, X_predict_scale = standard_scaler(filename)
	# print(X_train_scale)
	# print(X_predict_scale)
	X_train_mm, _, X_predict_mm = pro_min_max(filename)
	print(X_train_mm)

	# # 标准化处理
	# X, X_scale, y = pro_results(filename)
	# # 归一化处理
	# x_mm, y_ = pro_min_max(filename)
	# # 主成分分析
	# V_ratio, X_components, X_pca_scale = pca_results(filename, n=9)
	# df = pd.DataFrame(np.c_[V_ratio*100, X_components])
	# df.to_excel("save/idle_pca_reuslt.xlsx")
