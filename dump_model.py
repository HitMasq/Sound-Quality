"""
模型验证，用于查看结果、保存模型
作者：马世奇
完成时间：2019.11.07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.externals import joblib

import data_preprocess

# 选择分析文件
files = {"idle_": 13, "onst_": 16}
filename = "data/onst_de.xlsx"  # 修改文件名
file = filename.strip("data/").strip("de.xlsx")

# 标准化数据导入
x_train_scale, y_train, x_predict_scale = data_preprocess.standard_scaler(filename)
# 归一化数据导入
x_train_mm, y_, x_predict_mm = data_preprocess.pro_min_max(filename)
plt.plot(y_train, label='score')

# rbf
# svr_rbf = SVR(kernel='rbf', C=11.122, gamma=0.0404, epsilon=0.022)
# svr_rbf.fit(x_pca_scale, y_train)
# y_rbf = svr_rbf.fit(x_pca_scale, y_train).predict(x_pca_scale)
# plt.plot(y_rbf, label='rbf')
# joblib.dump(svr_rbf, "save/onst_svr_rbf_c_11.122_g_0.0404_e_0.022.pkl")

# sigmoid
svr_sig = SVR(kernel='sigmoid', C=0.5051, gamma=1.8018, epsilon=0.1055, coef0=-2.7778)
svr_sig.fit(x_train_mm, y_train)
y_sig = svr_sig.fit(x_train_mm, y_train).predict(x_train_mm)
plt.plot(y_sig, label='sigmoid')
joblib.dump(svr_sig, "save/onst_svr_sig_c_0.5051_g_1.8018_e_0.1055_coef_-2.7778.pkl")

# linear
# svr_linear = SVR(kernel='linear', C=0.1, epsilon=1)
# svr_linear.fit(x_pca_scale, y_train)
# y_linear = svr_linear.fit(x_pca_scale, y_train).predict(x_pca_scale)
# plt.plot(y_linear, label='linear')
# # joblib.dump(svr_linear, "save/idle_svr_linear_c_0.219.pkl")

# # poly
# svr_poly = SVR(kernel='poly', C=1.424, gamma=0.066, coef0=4.125, degree=2, epsilon=1)
# svr_poly.fit(x_scale_train, y_train)
# y_poly = svr_poly.fit(x_scale_train, y_train).predict(x_scale_train)
# plt.plot(y_poly, label='poly')
# # joblib.dump(svr_poly, "save/idle_svr_poly_c_0.471_g_0.301_e_1_coef_0_degree_2.pkl")
#
# col = ['score', 'rbf', 'sigmoid', 'linear', 'poly']
# row = np.linspace(1, files[file], files[file])
# prediction = np.c_[y_train, y_rbf, y_sig, y_linear, y_poly]
#
# df = pd.DataFrame(prediction, index=row, columns=col)
# df.to_excel("save/" + file + "_plt.xlsx")

plt.show()
