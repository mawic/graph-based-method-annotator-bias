import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


def plotDistribtionsOfGroups(df_raw,xaxis,yaxis,values):
    # Data
    df = pd.pivot_table(df_raw, index=xaxis,values=values,columns=[yaxis],margins=False,aggfunc='count', fill_value=0)
    df_percentage = df.div(df.sum(axis=1), axis=0)

    r = df.index.unique().values
    names = df.index.unique().values
    categories = df_percentage.columns

    bars = []
    for cat in categories:
        bars.append(df_percentage[cat])

    # plot
    barWidth = 0.85
    color = sns.color_palette("Set2",len(categories))
    bottom = [0] * len(bars[0])
    cn = 0
    for bar,category in zip(bars,categories): 
        plt.bar(r, bar, bottom=bottom, color=color[cn], edgecolor='white', width=barWidth, label=category)
        cn = cn + 1
        
        bottom = [sum(x) for x in zip(bar, bottom)]
        for index, value in enumerate(bar):
            plt.text(index, bottom[index]-value/2, "{:.1%}".format(value), horizontalalignment='center', verticalalignment='center')
        


    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel(xaxis)
    plt.title(yaxis)
    
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    # Show graphic
    plt.show()

def plotBoxplot(df_raw,xaxis,yaxis):
    boxplot = df_raw.boxplot(by=xaxis,column=yaxis, showfliers=False, showmeans=True)
    title_boxplot = 'yaxis'
    plt.title( title_boxplot )
    plt.suptitle('')
    plt.show()

def plotMatrix(matrix, name="Please add name", relative=True):
    if relative:
        baseline = matrix[0][0]
        for x in range(0,len(matrix)):
            for y in range(0,len(matrix[x])):
                matrix[x][y] = matrix[x][y] - baseline

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

    sns.heatmap(matrix, ax=ax1,annot=True, fmt=".2%",vmin=min_val, vmax=max_val, cbar=False,cmap=cmap, center=center)
    sns.heatmap(avg_testsets, ax=ax2, annot=True, fmt=".2%", cbar=False, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap, center=center)
    sns.heatmap(avg_classifiers, ax=ax3, annot=True, fmt=".2%", cbar=True, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap, center=center)

    ax1.xaxis.tick_top()
    ax1.set(xlabel='Test sets', ylabel='Classifiers')

    if relative:
        ax1.set_title(name + ' - Delta to baseline', fontsize=16)
    else:
        ax1.set_title(name, fontsize=16)
    ax1.xaxis.set_label_coords(0.5, 1.1)

    ax2 = ax2.set(xlabel='', ylabel='AVG')
    ax3.set(xlabel='AVG', ylabel='')
    ax3.xaxis.set_label_coords(0.5, 1.1)
