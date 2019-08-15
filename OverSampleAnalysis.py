# 上采样
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np, matplotlib.pyplot as plt
from Plot import plot_confusion_matrix, PlotConfusion
from sklearn.linear_model import LogisticRegression

credit_cards = pd.read_csv('data/creditcard.csv')
columns = credit_cards.columns
# the labels are in the last column ('Class') , simply remove it to obtain features columns
features_columns = columns.delete(len(columns) - 1)
features = credit_cards[features_columns]
labels = credit_cards['Class']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)

oversampler = SMOTE(random_state=0)
os_features, os_labels = oversampler.fit_sample(features_train, labels_train)
# len(os_labels[os_labels == 1])  227454

best_c = 10
lr = LogisticRegression(solver='liblinear', C=best_c, penalty='l1')
lr.fit(os_features, os_labels)
y_pred = lr.predict(features_test.values)
# compute confusion matrix
cnf_matrix = confusion_matrix(labels_test, y_pred)
np.set_printoptions(precision=2)

print('Recall metirc in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])) #0.9108910891089109
# # plot non-normalized  confusion matrix
PlotConfusion(cnf_matrix)
plt.show()

# 结论：如果能用上采样就尽量用上采样，因为数据越多越好。




