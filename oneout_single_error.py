import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def cal_error_rbf(C, gamma, epsilon, train_data):
	svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
	Err = []
	for i in range(len(train_data)):
		x_test = train_data[i].reshape(1, 6)
		y_test = y_train[i]
		x = np.delete(train_data, i, axis=0)
		y = np.delete(y_train, i)
		y_predict = svr.fit(x, y).predict(x_test)
		y_err = np.abs(y_test-y_predict)/y_test
		Err.append(y_err)
	return np.mean(Err)


def cal_error_sig(C, gamma, Coef, epsilon, train_data):
	svr = SVR(kernel='sigmoid', C=C, gamma=gamma, coef0=Coef, epsilon=epsilon)
	Err = []
	for i in range(len(train_data)):
		x_test = train_data[i].reshape(1, 6)
		y_test = y_train[i]
		x = np.delete(train_data, i, axis=0)
		y = np.delete(y_train, i)
		y_predict = svr.fit(x, y).predict(x_test)
		y_err = np.abs(y_test-y_predict)/y_test
		Err.append(y_err)
	return np.mean(Err)


def cal_error_linear(C, epsilon, train_data):
	svr = SVR(kernel='linear', C=C, epsilon=epsilon)
	Err = []
	for i in range(len(train_data)):
		x_test = train_data[i].reshape(1, 7)
		y_test = y_train[i]
		x = np.delete(train_data, i, axis=0)
		y = np.delete(y_train, i)
		y_predict = svr.fit(x, y).predict(x_test)
		y_err = np.abs(y_test-y_predict)/y_test
		Err.append(y_err)
	return np.mean(Err)


def cal_error_poly(C, Gamma, Coef, Degree, epsilon, train_data):
	svr = SVR(kernel='sigmoid', C=C, gamma=Gamma, coef0=Coef, degree=Degree, epsilon=epsilon)
	Err = []
	for i in range(len(train_data)):
		x_test = train_data[i].reshape(1, 6)
		y_test = y_train[i]
		x = np.delete(train_data, i, axis=0)
		y = np.delete(y_train, i)
		y_predict = svr.fit(x, y).predict(x_test)
		y_err = np.abs(y_test-y_predict)/y_test
		Err.append(y_err)
	return np.mean(Err)


if __name__ == "__main__":
	import data_preprocess
	# 选择分析文件
	files = {"idle_": 13, "onst_": 16}
	filename = "data/idle_de.xlsx"  # 修改文件名
	file = filename.strip("data/").strip("de.xlsx")

	# 标准化数据导入
	x_train_scale, y_train, x_predict_scale = data_preprocess.standard_scaler(filename)
	# 归一化数据导入
	x_train_mm, y_, x_predict_mm = data_preprocess.pro_min_max(filename)

# const
# 	# linear 误差计算
# 	c_linear = 0.40405
# 	e_linear = 0.4545
# 	MAPE_linear = cal_error_linear(c_linear, e_linear, x_train_mm)
# 	print('linear: ', MAPE_linear)
#
# 	# rbf 误差计算
# 	c_rbf = 3.3133
# 	g_rbf = 1.4815
# 	e_rbf = 0.15016
# 	MAPE_rbf = cal_error_rbf(c_rbf, g_rbf, e_rbf, x_train_mm)
# 	print('rbf: ', MAPE_rbf)
#
# 	# sigmoid 误差计算
# 	c_sig = 0.61182
# 	g_sig = 3.035
# 	coef_sig = -3.8909
# 	e_sig = 0.066067
# 	MAPE_sig = cal_error_sig(c_sig, g_sig, coef_sig, e_sig, x_train_mm)
# 	print('sigmoid: ', MAPE_sig)
#
# 	poly 误差计算
# 	c_poly = 1.51516
# 	g_poly = 0.202
# 	coef_poly = -3.88889
# 	degree = 3
# 	e_poly = 0.0991
# 	MAPE_poly = cal_error_poly(c_poly, g_poly, coef_poly, degree, e_poly, x_train_mm)
# 	print('poly: ', MAPE_poly)

# idle
# 	# linear 误差计算
# 	c_linear = 0.40405
# 	e_linear = 0.4545
# 	MAPE_linear = cal_error_linear(c_linear, e_linear, x_train_mm)
# 	print('linear: ', MAPE_linear)
#
# 	# rbf 误差计算
# 	c_rbf = 3.3133
# 	g_rbf = 1.4815
# 	e_rbf = 0.15016
# 	MAPE_rbf = cal_error_rbf(c_rbf, g_rbf, e_rbf, x_train_mm)
# 	print('rbf: ', MAPE_rbf)
#
	# sigmoid 误差计算
	c_sig = 0.5095
	g_sig = 1.8018
	coef_sig = -2.77878
	e_sig = 0.12112
	MAPE_sig = cal_error_sig(c_sig, g_sig, coef_sig, e_sig, x_train_mm)
	print('sigmoid: ', MAPE_sig)
#
# 	# poly 误差计算
# 	c_poly = 1.51516
# 	g_poly = 0.202
# 	coef_poly = -3.88889
# 	degree = 3
# 	e_poly = 0.0991
# 	MAPE_poly = cal_error_poly(c_poly, g_poly, coef_poly, degree, e_poly, x_train_mm)
# 	print('poly: ', MAPE_poly)

	# mapes = []
	# for i in np.linspace(0.00001, 5, 100):
	# 	g = 0.101
	# 	e = 0
	# 	mape = cal_error_rbf(i, g, e, x_scale_train)
	# 	mapes.append(mape)
	# x = np.linspace(0.00001, 5, 100)
	# plt.plot(x, mapes)
	# plt.show()
