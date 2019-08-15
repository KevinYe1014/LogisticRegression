import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('../data/creditcard.csv')
# 查看数据head
# head = data.head()
# print(head)

count_classes = pd.value_counts(data['Class'], sort=True).sort_index()
# 画直方图的时候这个函数非常有用，其中sort是默认升序，按照第一个，sort_index默认也是第一个
# 0    284315
# 1       492
# 画直方图
# count_classes.plot(kind = 'bar')
# plt.title('Fraud class histogram')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

from sklearn.preprocessing import  StandardScaler
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
# fit() 用于计算训练数据的均值和方差
# transform() 很显然，它只是进行转换，只是把训练数据转换成标准的正态分布
# fit_transform() 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
# 例子
# scaler = preprocessing.StandardScaler().fit(X)
# scaler.transform(X)
# 注： 测试数据和预测数据的标准化的方式要和训练数据标准化的方式一样， 必须用同一个scaler来进行transform
data = data.drop(['Time', 'Amount'], axis= 1)
# dataNewHead = data.head()

X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

# number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# picking the indices of the normal class
normal_indices = data[data.Class == 0].index

# Out of the indices we picked , randomly select 'x' number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

#Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]


X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample =under_sample_data.ix[:, under_sample_data.columns == 'Class']

# showing ratio
# print('Percentage of normal transactions: ', len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
# print('Percentage of fraud transactions: ', len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
# print('Total number of transactions in resampled data: ', len(under_sample_data))

# Percentage of normal transactions:  0.5
# Percentage of fraud transactions:  0.5
# Total number of transactions in resampled data:  984



from sklearn.model_selection import train_test_split

# whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# print("Number transactions train dataset: ", len(X_train))
# print("Number transactions test dataset: ", len(X_test))
# print("Total number of transactions: ", len(X_train)+len(X_test))

# undersample dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = \
    train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)
# print("")
# print("Number transactions train dataset: ", len(X_train_undersample))
# print("Number transactions test dataset: ", len(X_test_undersample))
# print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))

# Number transactions train dataset:  199364
# Number transactions test dataset:  85443
# Total number of transactions:  284807
#
# Number transactions train dataset:  688
# Number transactions test dataset:  296
# Total number of transactions:  984

# collect data
joblib.dump(X_train_undersample, '../data/X_train_undersample')
joblib.dump(X_test_undersample, '../data/X_test_undersample')
joblib.dump(y_train_undersample, '../data/y_train_undersample')
joblib.dump(y_test_undersample, '../data/y_test_undersample')

joblib.dump(X_train, '../data/X_train')
joblib.dump(X_test, '../data/X_test')
joblib.dump(y_train, '../data/y_train')
joblib.dump(y_test, '../data/y_test')


