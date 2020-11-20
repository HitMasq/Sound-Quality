import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import validation_curve

import data_preprocess

files = {"idle_": 13, "onst_": 16}
filename = "data/idle_de.xlsx"							# 修改文件名
file = filename.strip("data/").strip("de.xlsx")

# 标准化数据导入
x_train_scale, y_train, x_predict_scale = data_preprocess.standard_scaler(filename)
# 归一化数据导入
x_train_mm, y_, x_predict_mm = data_preprocess.pro_min_max(filename)

# 数据分析
svr_linear = SVR(kernel='linear', C=32.072, epsilon=0.40405)
svr_rbf = SVR(kernel='rbf', C=3.3133, gamma=1.4815, epsilon=0.15016)								# 其他参数固定
svr_sig = SVR(kernel='sigmoid', C=0.5095, gamma=1.8018, epsilon=0.12112, coef0=-2.77878)
svr_poly = SVR(kernel='poly', C=1.515, gamma=0.202, coef0=-3.889, degree=3, epsilon=0.0991)		# 其他参数固定

param_range = np.linspace(-5, 2, 1000)												# 参数区间
train_loss, test_loss = validation_curve(svr_sig, x_train_mm, y_train,					# 需要观察的模型
										 param_name='coef0',									# 需要观察的参数
										 param_range=param_range,
										 cv=files[file],									# 交叉验证
										 scoring='neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# 可视化
plt.plot(param_range, train_loss_mean, color='r', label="Training")
plt.plot(param_range, test_loss_mean, color='g', label="Cross_Validation")
plt.xlabel('C')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.show()
