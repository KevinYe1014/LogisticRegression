from CalculateCParameter import printing_Kfold_scores, TrainLogisticReg, TrainLogisticRegWithProb
from Plot import plot_confusion_matrix, PlotConfusion, PlotConfusionWithThreshold
from sklearn.metrics import confusion_matrix
import joblib
import itertools
import numpy as np

# 加载数据
X_train_undersample = joblib.load( 'data/X_train_undersample')
X_test_undersample= joblib.load('data/X_test_undersample')
y_train_undersample = joblib.load('data/y_train_undersample')
y_test_undersample = joblib.load('data/y_test_undersample')

X_train = joblib.load('data/X_train')
X_test = joblib.load('data/X_test')
y_train = joblib.load('data/y_train')
y_test = joblib.load('data/y_test')

# best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

# # 在下采样样本上显示
# y_pred_undersample = TrainLogisticReg(X_train_undersample, y_train_undersample, best_c, X_test_undersample)
# # Compute the confusion matrix
# cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
# # 计算混淆矩阵，其中前面y_test_undersample是dataFrame 而 后面y_pred_undersample是np.array()
# np.set_printoptions(precision = 2)
# # 设置显示方式
# # region
# #精度为小数点后4位
# # np.set_printoptions(precision=4)
# # print(np.array([1.123456789]))  [1.1235]
#
# # #超过阈值就缩写
# # np.set_printoptions(threshold=4)
# # print(np.arange(10))  [0 1 2 … 7 8 9]
# #
# # #过小的结果会被压缩
# # np.set_printoptions(suppress=True)
# # x**2 - (x + eps)**2 array([-0., -0., 0., 0.])
# #
# # #用户设定数组元素的显示形式
# # np.set_printoptions(formatter={'all':lambda x: '囧: '+str(-x)})#所有元素应用一个lambda函数
# # x = np.arange(3)
# # array([囧: 0, 囧: -1, 囧: -2])
# # endregion
# print('Recall metirc in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# # plot non-normalized  confusion matrix
# PlotConfusion(cnf_matrix)
#
#
# # 在整个测试样本上显示
# y_pred = TrainLogisticReg(X_train_undersample, y_train_undersample, best_c, X_test)
# # Compute the confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# # 计算混淆矩阵，其中前面y_test_undersample是dataFrame 而 后面y_pred_undersample是np.array()
# np.set_printoptions(precision = 2)
# print('Recall metirc in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# PlotConfusion(cnf_matrix)


# 新的best_c
# # best_c = printing_Kfold_scores(X_train, y_train) # 10
# best_c = 10
# # 在整个测试样本上显示
# y_pred = TrainLogisticReg(X_train, y_train, best_c, X_test)
# # Compute the confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# # 计算混淆矩阵，其中前面y_test_undersample是dataFrame 而 后面y_pred_undersample是np.array()
# np.set_printoptions(precision = 2)
# print('Recall metirc in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
# PlotConfusion(cnf_matrix)

# 虽然一般阈值选择是0.5，但是有时候有特殊的要求，所以需要选择合适的阈值
y_pred_undersample_proba = TrainLogisticRegWithProb(X_train_undersample, y_train_undersample, 0.01, X_test_undersample )
thresholds = [i * 0.1 for i in range(1, 10)]
PlotConfusionWithThreshold(thresholds, y_test_undersample,  y_pred_undersample_proba)







