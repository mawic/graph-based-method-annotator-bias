import io
import os
from contextlib import redirect_stdout
import pandas as pd
import ktrain
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from ktrain import text
from tqdm import tqdm


def preprocess(text):
    text = text.replace("NEWLINE_TOKEN"," ")
    text = text.replace("TAB_TOKEN"," ")

    return text

def loadData(number_of_groups,path_files,dic,random_state=0):
    col_x_train = []
    col_y_train = []
    col_x_dev = []
    col_y_dev = []
    col_x_test = []
    col_y_test = []
    
    for i in range(0,number_of_groups):
        df_train_raw = pd.read_pickle(path_files + str(number_of_groups) + "_" + str(i)+ '_TRAIN.pkl')
        df_test = pd.read_pickle(path_files + str(number_of_groups) + "_" + str(i)+ '_TEST.pkl')

        df_train_raw["attack"] = df_train_raw["attack"].astype(int)
        df_test["attack"] = df_test["attack"].astype(int)

        df_train_raw["attack"] = [dic.get(n, n) for n in df_train_raw["attack"] ]
        df_test["attack"] = [dic.get(n, n) for n in df_test["attack"] ]

        df_train = df_train_raw.sample(frac=0.9, random_state=random_state)
        df_vali = df_train_raw.drop(df_train.index)
        
        df_train['comment'] = df_train['comment'].apply(lambda x: preprocess(x))
        df_vali['comment'] = df_vali['comment'].apply(lambda x: preprocess(x))
        df_test['comment'] = df_test['comment'].apply(lambda x: preprocess(x))
        
        col_x_train.append(df_train["comment"].tolist())
        col_x_dev.append( df_vali["comment"].tolist())
        col_x_test.append(df_test["comment"].tolist())
        col_y_train.append(df_train["attack"].tolist())
        col_y_dev.append(df_vali["attack"].tolist())
        col_y_test.append(df_test["attack"].tolist())
    
        print(np.unique(df_train["attack"].tolist()))
        print(np.unique(df_test["attack"].tolist()))
        print(np.unique(df_vali["attack"].tolist()))
    
    list_of_lists = []
    columns = ['Group','x_train','y_train','x_dev','y_dev','x_test','y_test']
    for i in range(0,number_of_groups):
        list_of_lists.append(["Group "+str(i),len(col_x_train[i]),len(col_y_train[i]),len(col_x_dev[i]),len(col_y_dev[i]),len(col_x_test[i]),len(col_y_test[i]),])

    stat = pd.DataFrame(list_of_lists, columns=columns)
    
    
    return (col_x_train,col_y_train,col_x_dev,col_y_dev,col_x_test,col_y_test,stat)

def compare2Lists(a,b):
    print("Length list a:",len(a))
    print("Length list b:",len(b))
    if len(a) != len(b):
        print("Lists have different lengths")
    else:
        same = 0
        for i in range(0,len(a)):
            if a[i] == b[i]:
                same = same +1
        print("Shared between the lists:", round(same/len(a),2), same)
        
def train_model(i, col_f1_macro,col_f1_micro,col_f1_weighted,col_f1_class_0,col_f1_class_1,col_test_pred,col_precision,col_recall, model_name, number_tokens,batch, col_x_train,col_y_train,col_x_dev,col_y_dev,col_x_test,col_y_test):
    # set up model
    #target_names = list(set(col_y_train[i]))
    t = text.Transformer(model_name, maxlen=number_tokens, class_names=['ATTACK', 'OTHER'])
    #t = text.Transformer(MODEL_NAME, maxlen=100, classes=target_names)

    # preprocess
    trn = t.preprocess_train(col_x_train[i], col_y_train[i])
    val = t.preprocess_test(col_x_dev[i], col_y_dev[i])

    # set up learner
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch)

    #start training
    start = time.time()
    
    #estimate learning rate
    #learner.lr_find(show_plot=True, max_epochs=2)

    #learner.fit_onecycle(5e-6, 4)
    learner.fit_onecycle(5e-6, 2)
    #learner.autofit(5e-6, checkpoint_folder='./tmp/') 

    end = time.time()
    print("Run time:", (end - start)/60,"min")
       
    # evaluate
    predictor = ktrain.get_predictor(learner.model, preproc=t)
                            
    f1_macro = []
    f1_micro = []
    f1_weighted = [] 
    f1_class_0 = [] 
    f1_class_1 = [] 
    pred = []
    precision = []
    recall = []
                            
    y_true = col_y_test[0]
    y_pred = predictor.predict(col_x_test[0])
    pred.append(y_pred)

    f1_macro.append(f1_score(y_true, y_pred, average='macro'))
    f1_micro.append(f1_score(y_true, y_pred, average='micro'))
    f1_weighted.append(f1_score(y_true, y_pred, average='weighted'))
    f1_class_0.append(f1_score(y_true, y_pred, average=None))
    f1_class_1.append(f1_score(y_true, y_pred, average=None))
    precision.append(precision_score(y_true, y_pred, average=None))
    recall.append(f1_score(y_true, y_pred, average=None))
        
    for j in range(1,len(col_y_test)):
        y_true = col_y_test[j]
        #y_pred = predictor.predict(col_x_test[i])
        pred.append(y_pred)

        f1_macro.append(f1_score(y_true, y_pred, average='macro'))
        f1_micro.append(f1_score(y_true, y_pred, average='micro'))
        f1_weighted.append(f1_score(y_true, y_pred, average='weighted'))
        
        f1_class_0.append(f1_score(y_true, y_pred, average=None))
        f1_class_1.append(f1_score(y_true, y_pred, average=None))    
        
        precision.append(precision_score(y_true, y_pred, average=None))
        recall.append(f1_score(y_true, y_pred, average=None))
        
    #if i == 0:
    #    f1_macro.append(f1_score(y_true, y_pred, average='macro'))
    #    f1_micro.append(f1_score(y_true, y_pred, average='micro'))
    #    f1_weighted.append(f1_score(y_true, y_pred, average='weighted'))
    #    f1_class_0.append(f1_score(y_true, y_pred, average=None))
    #    f1_class_1.append(f1_score(y_true, y_pred, average=None))    
    #    precision.append(precision_score(y_true, y_pred, average=None))
    #    recall.append(recall_score(y_true, y_pred, average=None))
    #    pred.append(y_pred)
    #else:
    #    y_true = col_y_test[i]
    #    #y_pred = predictor.predict(col_x_test[i])
    #    pred.append(y_pred)
#
 #      f1_macro.append(f1_score(y_true, y_pred, average='macro'))
  #     f1_micro.append(f1_score(y_true, y_pred, average='micro'))
   #    f1_weighted.append(f1_score(y_true, y_pred, average='weighted'))
    #   
    #   f1_class_0.append(f1_score(y_true, y_pred, average=None))
    #   f1_class_1.append(f1_score(y_true, y_pred, average=None))    
     #   
     #   precision.append(precision_score(y_true, y_pred, average=None))
      #  recall.append(f1_score(y_true, y_pred, average=None))
                            
    col_f1_macro.append(f1_macro)
    col_f1_micro.append(f1_micro)
    col_f1_weighted.append(f1_weighted)
    col_f1_class_0.append(f1_class_0)
    col_f1_class_1.append(f1_class_1)
    col_test_pred.append(pred)
    col_precision.append(precision)
    col_recall.append(recall)
    
    #tst = t.preprocess_test(col_x_test[0], col_y_test[0])
    #learner.evaluate(tst, class_names=['ATTACK', 'OTHER'])
    
def getRelativeMatrix(matrix):
    mx = np.copy(matrix)
    base = mx[0][0]
    for i in range(0,len(mx)):
        for j in range(0,len(mx[0])):
            mx[i][j]=mx[i][j]-base
    return mx


def splitClassF1Scores(matrix):
    class_0 = []
    class_1 = []
    for i in range(0,len(matrix)):
        class_0.append([])
        class_1.append([])
        for j in range(0,len(matrix[0])):
            class_0[i].append(matrix[i][j][0])
            class_1[i].append(matrix[i][j][1])
    return(class_0,class_1)

def printHeatmap(matrix,type_name="Please add name",exp_name="Please add name",path="./",run=""):
    # Activating tex in all labels globally
    #sns.set(rc={'text.usetex': True})
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #plt.rc('text.latex', preamble=r'\sffamily')
    sns.set(font_scale=2)
    cmap = "RdBu"
    
    rel_matrix = getRelativeMatrix(matrix)
    rel_matrix [0][1] = np.nan
    
    # copy of relative matrix for annotation
    rel_matrix_annotation = []
    for i in range(0,len(rel_matrix)):
        row = []
        for j in range(0,len(rel_matrix[0])):
            if i == 0 and j == 0:
                row.append(r'\underline{' + '{:.2%}'.format(matrix[i][j]).replace("%",r"\%") + '}')
                #row.append( "{:.2%}".format(matrix[i][j]))
            else:
                row.append("{:+.2%}".format(rel_matrix[i][j]).replace("%","pp"))
        rel_matrix_annotation.append(row)
    rel_matrix_annotation = np.array(rel_matrix_annotation)
    
    min_val_1 = np.nanmin(rel_matrix)
    max_val_1 = np.nanmax(rel_matrix) 
    center_1 = rel_matrix[0][0]
    
    size = len(matrix)
    
    fig = plt.figure(figsize=(size-1,size))
    fig.suptitle(type_name + r" \- ", fontsize=28)
    #fig.suptitle(type_name + r" \- " + exp_name, fontsize=16)

    ax1 = plt.subplot2grid((1,1), (0, 0))
    x_axis_labels = ["0","resp."]
    
    #ax1 = sns.heatmap(rel_matrix, ax=ax1, annot=rel_matrix, fmt=".2%",vmin=min_val_1, vmax=max_val_1,cmap=cmap, center=center_1)
    ax1 = sns.heatmap(rel_matrix, ax=ax1, annot=False, fmt=".2",vmin=min_val_1, vmax=max_val_1,cmap=cmap, center=center_1, cbar=False, xticklabels=x_axis_labels)
    ax1 = sns.heatmap(rel_matrix, ax=ax1, annot=rel_matrix_annotation,vmin=min_val_1, vmax=max_val_1,cmap=cmap, center=center_1,fmt='', cbar=False, xticklabels=x_axis_labels)
    ax1.set(xlabel='Test set', ylabel='Classifier')
    fig.savefig(path+run+exp_name.replace(" ","")+"-"+type_name.replace(" ","")+"_" + str(len(matrix)) + "-groups2.pdf", bbox_inches='tight')
    fig.savefig(path+run+exp_name.replace(" ","")+"-"+type_name.replace(" ","")+"_" + str(len(matrix)) + "-groups2.png", bbox_inches='tight')
    
def plotMatrix(matrix_raw, type_name="Please add name",exp_name="Please add name",path="./",run="",relative=True):
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    sns.set(font_scale=1.5)
    matrix = np.empty([len(matrix_raw),len(matrix_raw)])
    matrix_2 = np.empty([len(matrix_raw),len(matrix_raw)])
    matrix_2[0][0] = matrix_raw[0][0]
    if relative:
        baseline = matrix_raw[0][0]
        for x in range(0,len(matrix_raw)):
            for y in range(0,len(matrix_raw[x])):
                matrix[x][y] = matrix_raw[x][y] - baseline

    # calculate averages
    avg_classifiers = []
    avg_testsets = []

    for i in range(0,len(matrix)):
        avg_classifiers.append(statistics.mean(matrix[i]))   

    for i in range(0,len(matrix)):
        list_val = []
        for j in range(0,len(matrix)):
            list_val.append(matrix[j][i])
        avg_testsets.append(statistics.mean(list_val))   


    size = len(matrix[0])
    min_val = np.amin(matrix)
    max_val = np.amax(matrix) 

    avg_classifiers = np.asarray(avg_classifiers).reshape(size,1)
    avg_testsets = np.asarray(avg_testsets).reshape(1,size)

    fig = plt.figure(figsize=(size+1,size+1))
    ax1 = plt.subplot2grid((size+1,size+1), (0,0), colspan=size, rowspan=size)
    ax2 = plt.subplot2grid((size+1,size+1), (size,0), colspan=size, rowspan=1)
    ax3 = plt.subplot2grid((size+1,size+1), (0,size), colspan=1, rowspan=size)

    cmap = "RdBu"
    center = matrix[0][0]

    sns.heatmap(matrix,mask=matrix < 1, ax=ax1,annot=True, fmt=".2%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap, center=center)
    sns.heatmap(matrix_2,mask=matrix_2 > 1, ax=ax1,annot=True, fmt=".2%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap, center=center)
    sns.heatmap(avg_testsets, ax=ax2, annot=True, fmt=".2%", cbar=False, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap, center=center)
    sns.heatmap(avg_classifiers, ax=ax3, annot=True, fmt=".2%", cbar=False, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap, center=center)

    
    ax1.set_title(type_name)
    ax1.xaxis.tick_top()
    ax1.set(xlabel='Test sets', ylabel='Classifiers')
    ax1.xaxis.set_label_coords(0.5, 1.13)

    ax2 = ax2.set(xlabel='', ylabel='AVG')
    ax3.set(xlabel='AVG', ylabel='')
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_coords(0.5, 1.13)
    
    fig.savefig(path+run+exp_name.replace(" ","")+"-"+type_name.replace(" ","")+"_" + str(len(matrix)) + "-groups2.pdf", bbox_inches='tight')
    fig.savefig(path+run+exp_name.replace(" ","")+"-"+type_name.replace(" ","")+"_" + str(len(matrix)) + "-groups2.png", bbox_inches='tight')
