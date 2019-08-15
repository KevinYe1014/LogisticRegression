# ##一个简单的2折交叉验证
# from sklearn.model_selection import KFold
# import numpy as np
# X=np.array([[1,2],[3,4],[1,3],[3,5]])
# Y=np.array([1,2,3,4])
# KF=KFold(n_splits=2)  #建立4折交叉验证方法  查一下KFold函数的参数
# for train_index,test_index in KF.split(X):
#     print("TRAIN:",train_index,"TEST:",test_index)
#     X_train,X_test=X[train_index],X[test_index]
#     Y_train,Y_test=Y[train_index],Y[test_index]
#     print(X_train,X_test)
#     print(Y_train,Y_test)
# #小结：KFold这个包 划分k折交叉验证的时候，是以TEST集的顺序为主的，举例来说，如果划分4折交叉验证，那么TEST选取的顺序为[0].[1],[2],[3]。
#
# #提升
# import numpy as np
# from sklearn.model_selection import KFold
# #Sample=np.random.rand(50,15) #建立一个50行12列的随机数组
# Sam=np.array(np.random.randn(1000)) #1000个随机数
# New_sam=KFold(n_splits=5)
# for train_index,test_index in New_sam.split(Sam):  #对Sam数据建立5折交叉验证的划分
# #for test_index,train_index in New_sam.split(Sam):  #默认第一个参数是训练集，第二个参数是测试集
#     #print(train_index,test_index)
#     Sam_train,Sam_test=Sam[train_index],Sam[test_index]
#     print('训练集数量:',Sam_train.shape,'测试集数量:',Sam_test.shape)  #结果表明每次划分的数量
#
#
# #Stratified k-fold 按照百分比划分数据
# from sklearn.model_selection import StratifiedKFold
# import numpy as np
# m=np.array([[1,2],[3,5],[2,4],[5,7],[3,4],[2,7]])
# n=np.array([0,0,0,1,1,1])
# skf=StratifiedKFold(n_splits=3)
# for train_index,test_index in skf.split(m,n):
#     print("train",train_index,"test",test_index)
#     x_train,x_test=m[train_index],m[test_index]
# #Stratified k-fold 按照百分比划分数据
# from sklearn.model_selection import StratifiedKFold
# import numpy as np
# y1=np.array(range(10))
# y2=np.array(range(20,30))
# y3=np.array(np.random.randn(10))
# m=np.append(y1,y2) #生成1000个随机数
# m1=np.append(m,y3)
# n=[i//10 for i in range(30)]  #生成25个重复数据
#
# skf=StratifiedKFold(n_splits=5)
# for train_index,test_index in skf.split(m1,n):
#     print("train",train_index,"test",test_index)
#     x_train,x_test=m1[train_index],m1[test_index]


# import  numpy as np
# arr = np.arange(12).reshape(3, 4)
# a = arr.ravel()
# b = arr.flatten()
# c = arr.reshape(-1)
# d = arr.flatten()
# print(d)
import itertools

a = itertools.product(range(2), range(2))
print(list(a))