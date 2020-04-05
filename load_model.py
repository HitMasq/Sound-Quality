"""
模型加载，用于预测
作者：马世奇
完成时间：2019.11.08
"""

import pandas as pd
from sklearn.externals import joblib

import data_preprocess

# 选择分析文件
files = {"idle_": 13, "onst_": 16}
filename = "data/idle_de.xlsx"  # 修改文件名
file = filename.strip("data/").strip("de.xlsx")

# 标准化数据导入
x_train_scale, y_train, x_predict_scale = data_preprocess.standard_scaler(filename)
# 归一化数据导入
x_train_mm, y_, x_predict_mm = data_preprocess.pro_min_max(filename)

# 导入保存的模型
model = "save/idle_svr_sig_c_0.5051_g_1.8018_e_0.1055_coef_-2.7778.pkl"
svr_rbf = joblib.load(model)
score = svr_rbf.predict(x_predict_mm)
df = pd.Series(score)
df.to_excel(model.strip(".pkl")+".xlsx")
