from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, classification_report
import pandas as pd
import numpy as np

def printing_Kfold_scores(X_train_data, y_train_data):
    # fold = KFold(len(y_train_data), 5, shuffle=False)

    fold = KFold(n_splits=5)
    # KFold解释
    # region
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
    # endregion

    # different C parameters
    c_param_range = [0.01, 0.1, 1, 10, 100]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns= ['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the K-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------')
        print('C parameter: ',c_param)
        print('-------------------------------')
        print('')

        recall_accs = []
        iteration = 0
        # for iteration, indices in enumerate(fold, start=1):
        for train_index,test_index in fold.split(X_train_data):
            # call the logistic regression model with a certain c parameter
            lr = LogisticRegression(solver='liblinear', C = c_param, penalty='l1',max_iter = 1000 )  # L1

            # use the training data to fit the model , In this case , we use the protion of the fold to train the model
            # with indices[0] , we then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(X_train_data.iloc[train_index, :].values, y_train_data.iloc[train_index, :].values.ravel())
            # 下面都是拉平的。
            #region
            # arr = np.arange(12).reshape(3, 4)
            # a = arr.ravel()
            # b = arr.flatten()
            # c = arr.reshape(-1)
            # d = arr.flatten()
            # endregion

            # predict values using the test indices in the training data
            y_pred_undersample = lr.predict(X_train_data.iloc[test_index, :].values)

            # calculate the recall score and append it to a list for recall scores representing the current c_params
            recall_acc = recall_score(y_train_data.iloc[test_index, :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            iteration += 1
            print('Iteration ', iteration , ': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[np.array(results_table['Mean recall score'].values).argmax()]['C_parameter']

    # Finally , we can check which C parameter is the best amongst the chosen .
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ',best_c)
    print('*********************************************************************************')

    return best_c

def TrainLogisticReg(x, y, c, x_test):
    lr = LogisticRegression(solver='liblinear', C=c, penalty='l1')
    lr.fit(x, y.values.ravel())
    y_pred_undersample = lr.predict(x_test.values)
    return y_pred_undersample

def TrainLogisticRegWithProb(x, y, c, x_test):
    lr = LogisticRegression(solver='liblinear', C=c, penalty='l1')
    lr.fit(x, y.values.ravel())
    y_pred_undersample = lr.predict_proba(x_test.values)
    return y_pred_undersample