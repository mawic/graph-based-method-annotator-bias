from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import helper.confusion_matrix_pretty_print as confusion_matrix_pretty_print
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def getProcessedArrays(y_true, y_pred, best_threshold=0.5):
    y_pred[y_pred >= best_threshold] = 1
    y_pred[y_pred < best_threshold] = 0   

    y_pred = np.array(y_pred).flatten().astype(int)

    return(y_true,y_pred)

def f1_score_overall(y_true, y_pred):
    
    class_scores = f1_score(y_true, y_pred, labels=[0,1], average=None)
    overall_score = f1_score(y_true, y_pred, labels=[0,1], average='weighted')
    matrix = confusion_matrix(y_true, y_pred)
    
    return(class_scores, overall_score, matrix)

def printEvaluationResults(y_true, y_pred, labels=None, best_threshold=0.5, only_overall=True):
    y_true, y_pred = getProcessedArrays(y_true, y_pred, best_threshold)

    class_scores, overall_score, confusion_matrix_plain = f1_score_overall(y_true, y_pred)
    print("Overall\t\tF1:",overall_score)
    i = 0
    for c in labels:
        print("  ",c,"\tF1:", class_scores[i])
        i += 1
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(y_true, y_pred,columns=labels, figsize=[4,4],fz=12)

    return(class_scores, overall_score, confusion_matrix_plain)

def getEvaluationResults(y_true, y_pred, labels=None, best_threshold=0.5, only_overall=True):
    y_true, y_pred = getProcessedArrays(y_true, y_pred, best_threshold)

    class_scores, overall_score, confusion_matrix_plain = f1_score_overall(y_true, y_pred)
    return(class_scores, overall_score, confusion_matrix_plain)
