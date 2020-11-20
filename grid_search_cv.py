import time
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import data_preprocess

files = {"idle_": 13, "onst_": 16}
filename = "data/idle_de.xlsx"							# 修改文件名
file = filename.strip("data/").strip("de.xlsx")

# 标准化数据导入
x_train_scale, y_train, x_predict_scale = data_preprocess.standard_scaler(filename)
# 归一化数据导入
x_train_mm, y_, x_predict_mm = data_preprocess.pro_min_max(filename)

# 网格划分
c_range = 		np.linspace(0.00001, 10, 100)
gamma_range = 	np.linspace(0.00001, 10, 100)
epsilon_range = np.linspace(0, 1, 10)
coef_range = 	np.linspace(-5, 5, 10)
degree_range = 	np.array([2, 3])
# 核函数参数选择
linear_key = {'C': c_range,
			  'epsilon': epsilon_range}

rbf_key = {'C': c_range,
		   'gamma': gamma_range,
		   'epsilon': epsilon_range}

sigmoid_key = {'C': c_range,
			   'gamma': gamma_range,
			   'coef0': coef_range}

poly_key = {'C': c_range,
			'gamma': gamma_range,
			'coef0': coef_range,
			'degree': degree_range}
# 网格搜索
t0 = time.time()													# 计时开始
svr = GridSearchCV(SVR(kernel='poly'),							# 改
					   cv=files[file],
					   param_grid=poly_key,						# 改
					   scoring='neg_mean_squared_error',
					   n_jobs=-1,
					   refit='neg_mean_squared_error')
# 数据预处理方式选择
svr_rbf = svr.fit(x_train_mm, y_train)							# 改

# 结果提取
score = -svr_rbf.best_score_
params = svr_rbf.best_params_
svr_time = time.time() - t0  										# 计时结束
# 结果显示
print("Score: ", score)
print("Params: ", params)
print("time: ", svr_time)
