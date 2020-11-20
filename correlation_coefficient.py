import subjective_score as ss
import numpy as np
import pandas as pd


files = {"idle.xlsx": 13, "const.xlsx": 16}			# 怠速13个；匀速16个
file = "const.xlsx"									# 改这里！！！
sheets, scores = ss.get_scores(file)

ind = np.arange(1, files[file]+1)
col = sheets
df = pd.DataFrame(scores.T, index=ind, columns=col)
df.corr().to_excel(file.strip('.xlsx') + '_pearson_all.xlsx')	# 输出相关系数矩阵

pearson = (df.corr().sum(axis=1)-1.0)/(len(sheets)-1)
df_pearson = pd.DataFrame([])
df_pearson['pearson'] = pearson
df_pearson.to_excel(file.strip('.xlsx') + '_pearson.xlsx')		# 输出相关系数均值


# print("Pearson:\n", df.corr().mean(1))
# print("Kendall:\n", df.corr('kendall').mean(1), '\n')
# print("Spearman:\n", df.corr('spearman').mean(1))
