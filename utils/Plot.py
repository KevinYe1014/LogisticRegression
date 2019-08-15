import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, title = 'Confusion matrix', cmap = plt.cm.Blues):
    '''
    this function prints and plots the confusion matrix
    :param cm:
    :param classes:
    :param title:
    :param cmap:
    :return:
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # a = itertools.product(range(2), range(2)) 产生 [(0, 0), (0, 1), (1, 0), (1, 1)]
        plt.text(j, i, cm[i, j], horizontalalignment = 'center', color = 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def PlotConfusion(cnf_matrix):
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()

def PlotConfusionWithThreshold(thresholds,y_test_undersample,  y_pred_undersample_proba):
    j = 1
    plt.figure(figsize=(10, 10))
    for i in thresholds:
        y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
        plt.subplot(3, 3, j)
        j += 1
        cnf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
        np.set_printoptions(precision= 2)
        print('Rcall metric in the testing dataset：', cnf_matrix[1, 1] / (cnf_matrix[1, 1] +cnf_matrix[1, 0]))

        # plot non-normalized confusion matrix
        class_names = [0, 1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s'%i)
    plt.show()

