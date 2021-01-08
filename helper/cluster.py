import pandas as pd
import numpy as np
import math
import time
import copy
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import networkx as nx
import networkx.algorithms.community as nxcom
import collections
import community as community_louvain
import seaborn as sns
import statistics
import krippendorff


def getCommunityGroups(G,method="louvian",random_state=0):
    groups = []
    if method == 'louvian':
        partition = community_louvain.best_partition(G, weight="weight",random_state=random_state,resolution=1)
        cnt = collections.Counter()
        for i in partition.values():
            cnt[i] += 1
        for part in cnt:
            groups.append([])
        for key in partition:
            groups[partition[key]].append(key)

    if method == 'greedy':
        groups_frozensets = list(nxcom.greedy_modularity_communities(G, weight="weight"))
        groups = [list(x) for x in groups_frozensets]
        
    return groups

def getStatsOfGroups(group_list):
    output = ""
    output += "Number of communities:\t"+str(len(group_list))+"\n"
    output += "Communities by size:\n"
    
    sizes = []
    for i in range(0,len(group_list)):
        sizes.append(len(group_list[i]))
        
    sizes.sort(reverse=True)
    
    for size in sizes:
        output += "\t" + str(size) + "\n"
    
    return output

def getExtendedStatsOfGroups(group_list, df_annotations):
    list_of_lists = []
    for group in group_list:
        selected = df_annotations.loc[df_annotations['worker_id'].isin(group)]
        list_of_lists.append([len(group), len(selected), len(selected.groupby('rev_id').nunique())])

    return pd.DataFrame(list_of_lists, columns=['#Annotators','#Annotations','#Comments'])

def removeSmallGroup(group_list_input,min_size_group):
    group_list = copy.deepcopy(group_list_input)
    for i in range(len(group_list)-1, -1,-1):
        length = len(group_list[i])
        if length < min_size_group:
            group_list.pop(i)
            #print("Deleted group with index", i,"and size of",length)
    return group_list

def getGroupSpecificDataSlices(group_list,df_annotations,df_comments):
    # store texts and annotations according to communites

    # data for leave out approach
    group_dfs_leave_out = []

    # data for select only approach
    group_dfs_each = []

    for group in group_list:
        # select annotations from group
        selected = df_annotations.loc[~df_annotations['worker_id'].isin(group)]
        selected_agg = selected.groupby('rev_id').mean()
        len_0 = len(selected_agg)
        #selected_agg = selected_agg[selected_agg['attack'] != 0.5]
        selected_agg[selected_agg['attack'] >= 0.5] = 1
        selected_agg[selected_agg['attack'] < 0.5] = 0
        #print("Number of indifferent (0.5) comments:", len_0-len(selected_agg), "with number of users:", len(group))

        # join with text
        merged = pd.merge(df_comments, selected_agg, how='right', on=['rev_id', 'rev_id'])
        # filter columns
        merged = merged[['rev_id','comment','year','logged_in','ns','sample','split','attack']]

        group_dfs_leave_out.append(merged)

    for group in group_list:
        # select annotations from group
        if len(group) != 0:
            selected = df_annotations.loc[df_annotations['worker_id'].isin(group)]
        else:
            selected = df_annotations
        selected_agg = selected.groupby('rev_id').mean()
        len_0 = len(selected_agg)
        #selected_agg = selected_agg[selected_agg['attack'] != 0.5]
        selected_agg[selected_agg['attack'] >= 0.5] = 1
        selected_agg[selected_agg['attack'] < 0.5] = 0
        #print("Number of indifferent (0.5) comments:", len_0-len(selected_agg), "with number of users:", len(group))

        # join with text
        merged = pd.merge(df_comments, selected_agg, how='right', on=['rev_id', 'rev_id'])
        # filter columns
        merged = merged[['rev_id','comment','year','logged_in','ns','sample','split','attack']]

        group_dfs_each.append(merged)

    return (group_dfs_each,group_dfs_leave_out)

def getSharedComments(group_dfs_each):
    shared_comments = group_dfs_each[0]
    for i in range(1,len(group_dfs_each)):
        shared_comments = pd.merge(shared_comments,group_dfs_each[i],left_on='rev_id',right_on='rev_id',how="inner")
    return shared_comments

def getNotContainedComments(group_dfs_each,df_comments):
    not_contained_comments = df_comments.copy()
    for i in range(1,len(group_dfs_each)):
        not_contained_comments = not_contained_comments.loc[~not_contained_comments['rev_id'].isin(group_dfs_each[i]['rev_id'])]
    return not_contained_comments


def storeFullySharedOnlyDataSlices(group_dfs_each,shared_comments,path_store,random_state=0):
    train_set = shared_comments.sample(frac=0.80, random_state=random_state)
    test_set = shared_comments.drop(train_set.index)
    
    print("Fully shared comments only:\t\t", len(train_set), "/",len(test_set))
    
    cn = 0
    for group in group_dfs_each:
        
        train = group[group.rev_id.isin(train_set['rev_id'])].copy()
        train.to_pickle(path_store+"_Fully_Shared_Only_"+str(len(group_dfs_each)) + "_" +str(cn)+"_TRAIN.pkl")
        
        ratio_train = len(train[train['attack'] == 1])/len(train)
        
        test = group[group.rev_id.isin(test_set['rev_id'])].copy()
        test.to_pickle(path_store+"_Fully_Shared_Only_"+str(len(group_dfs_each)) + "_" +str(cn)+"_TEST.pkl")
        
        ratio_test = len(test[test['attack'] == 1])/len(test)
        print("    ",str(cn),"-",len(train),"(",ratio_train,")","/", len(test),"(",ratio_test,")")
        cn = cn +1
    
def storeFullySharedLeaveOutDataSlices(group_dfs_leave_out,shared_comments,path_store,random_state=0):
    train_set = shared_comments.sample(frac=0.80, random_state=random_state)
    test_set = shared_comments.drop(train_set.index)
    
    print("Fully shared comments leave out:\t", len(train_set), "/",len(test_set))
    
    cn = 0
    for group in group_dfs_leave_out:
        
        train = group[group.rev_id.isin(train_set['rev_id'])].copy()
        train.to_pickle(path_store+"_Fully_Shared_Leave_out_"+str(len(group_dfs_leave_out)) + "_" +str(cn)+"_TRAIN.pkl")
        
        ratio_train = len(train[train['attack'] == 1])/len(train)
        
        test = group[group.rev_id.isin(test_set['rev_id'])].copy()
        test.to_pickle(path_store+"_Fully_Shared_Leave_out_"+str(len(group_dfs_leave_out)) + "_" +str(cn)+"_TEST.pkl")
        
        ratio_test = len(test[test['attack'] == 1])/len(test)
        print("    ",str(cn),"-",len(train),"(",ratio_train,")","/", len(test),"(",ratio_test,")")
        cn = cn +1

def storeTestSharedOnlyDataSlices(group_dfs_each,shared_comments,path_store,random_state=0):
    min_size = 200000
    for group in group_dfs_each: 
        if len(group) < min_size:
            min_size = len(group)
    
    test_size = len(shared_comments)
    if test_size > min_size*0.2:
        test_size = round(min_size*0.2)
    
    test_set = shared_comments.sample(n=test_size, random_state=random_state)
    #test_set = shared_comments.drop(train_set.index)
    
    print("Shared test comments only:\t\t", int(min_size*0.8), "/",len(test_set))
    
    cn = 0
    for group in group_dfs_each:
        
        train = group[~group.rev_id.isin(test_set['rev_id'])].copy()
        train = train.sample(n=int(min_size*0.8), random_state=random_state)
        
        ratio_train = len(train[train['attack'] == 1])/len(train)
        train.to_pickle(path_store+"_Test_Shared_Only_"+str(len(group_dfs_each)) + "_" +str(cn)+"_TRAIN.pkl")
        
        test = group[group.rev_id.isin(test_set['rev_id'])].copy()
        test.to_pickle(path_store+"_Test_Shared_Only_"+str(len(group_dfs_each)) + "_" +str(cn)+"_TEST.pkl")
        
        ratio_test = len(test[test['attack'] == 1])/len(test)
        print("    ",str(cn),"-",len(train),"(",ratio_train,")","/", len(test),"(",ratio_test,")")
        cn = cn +1
    
def storeTestSharedLeaveOutDataSlices(group_dfs_leave_out,shared_comments_leave_out,size_leave_out,path_store,random_state=0):
    shared_comments = shared_comments_leave_out.sample(n=size_leave_out, random_state=random_state)
    train_set = shared_comments.sample(frac=0.8, random_state=random_state)
    test_set = shared_comments.drop(train_set.index)
    
    print("Shared test comments leave out:\t\t", len(train_set), "/",len(test_set))
    
    cn = 0
    for group in group_dfs_leave_out:
        
        train = group[group.rev_id.isin(train_set['rev_id'])].copy()
        train.to_pickle(path_store+"_Test_Shared_Leave_out_"+str(len(group_dfs_leave_out)) + "_" +str(cn)+"_TRAIN.pkl")
        
        ratio_train = len(train[train['attack'] == 1])/len(train)
        
        test = group[group.rev_id.isin(test_set['rev_id'])].copy()
        test.to_pickle(path_store+"_Fully_Shared_Leave_out_"+str(len(group_dfs_leave_out)) + "_" +str(cn)+"_TEST.pkl")
        
        ratio_test = len(test[test['attack'] == 1])/len(test)
        print("    ",str(cn),"-",len(train),"(",ratio_train,")","/", len(test),"(",ratio_test,")")
        cn = cn +1  


def getInterraterReliabilityWithinGroup(group_list,df_annotations,shared_comments):
    alphas = []
    
    # pivot data frame
    df_annotations_pivot = pd.pivot_table(df_annotations, values='attack', index=['rev_id'], columns=['worker_id'], aggfunc=np.sum)
    
    # baseline group
    df_annotations_pivot_baseline_group = df_annotations_pivot[df_annotations_pivot.index.isin(shared_comments['rev_id'])]
    annotations_of_group = []
    for rater in df_annotations_pivot_baseline_group.columns:
        annotation = df_annotations_pivot_baseline_group[int(rater)].tolist()
        # add only with at least one annotation
        if np.count_nonzero(~np.isnan(annotation)) != 0:
            annotations_of_group.append(annotation)
    print(len(df_annotations_pivot_baseline_group))
    print(len(annotations_of_group))
    alphas.append(krippendorff.alpha(reliability_data=annotations_of_group, level_of_measurement='nominal'))
    
    # other groups
    for i in range(1,len(group_list)):
        annotations_of_group = []
        # select only annotations from selected group
        for rater in group_list[i]:
            annotations_of_group.append(df_annotations_pivot_baseline_group[int(rater)].tolist())
        # calculate krippendorffs alpha for sleected group
        if len(annotations_of_group) == 0:
            alphas.append(0.6)
        else:
            alphas.append(krippendorff.alpha(reliability_data=annotations_of_group, level_of_measurement='nominal'))
    return alphas

def getInterraterReliabilityBetweenGroups(group_list,df_annotations,shared_comments):
    alphas = np.empty([len(group_list), len(group_list)])
    
    # pivot data frame
    df_annotations_pivot = pd.pivot_table(df_annotations, values='attack', index=['rev_id'], columns=['worker_id'], aggfunc=np.sum)
    
    # baseline group
    df_annotations_pivot_baseline_group = df_annotations_pivot[df_annotations_pivot.index.isin(shared_comments['rev_id'])]
    
    for i in range(0,len(group_list)):
        for j in range(i+1,len(group_list)):
            annotations_of_group = []
            if i == 0:
                for rater in df_annotations_pivot_baseline_group.columns:
                    annotation = df_annotations_pivot_baseline_group[int(rater)].tolist()
                    # add only with at least one annotation
                    if np.count_nonzero(~np.isnan(annotation)) != 0:
                        annotations_of_group.append(annotation)
            else:
                for rater in group_list[i]:
                    annotations_of_group.append(df_annotations_pivot_baseline_group[int(rater)].tolist()) 
            
            for rater in group_list[j]:
                annotations_of_group.append(df_annotations_pivot_baseline_group[int(rater)].tolist()) 
    
            alpha = krippendorff.alpha(reliability_data=annotations_of_group, level_of_measurement='nominal')
            alphas[i][j] = alpha
            alphas[j][i] = alpha
    
    return alphas

def getInterraterReliability(group_dfs_each,shared_comments,group_list,df_annotations,selected_print_name,selected_type,path_results,man_min=None,man_max=None):
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    sns.set(font_scale=1.5)
    # get interrater reliabilities
    
    interrater_reliability = getInterraterReliabilityBetweenGroups(group_list,df_annotations,shared_comments)
    ingroup = np.array(getInterraterReliabilityWithinGroup(group_list,df_annotations,shared_comments)).flatten()
    ingroup = np.reshape(ingroup, (len(ingroup),-1))
    
    avg = []
    
    for i in range(0,len(group_list)):
        interrater_reliability[i][i] = np.nan
        avg.append([np.nanmean(interrater_reliability[i])])

    cmap = sns.cubehelix_palette(as_cmap=True)
    size=len(interrater_reliability)
    print(size)
    fig = plt.figure()

    off_1 = 2
    off_2 = 0

    ax0 = plt.subplot2grid((size+off_1,size+off_2), (0,0), colspan=1, rowspan=size+off_1)
    ax1 = plt.subplot2grid((size+off_1,size+off_2), (0,1), colspan=size-2, rowspan=size+off_1)
    ax3 = plt.subplot2grid((size+off_1,size+off_2), (0,size-1), colspan=1, rowspan=size+off_1)


    min_val_1 = np.nanmin(interrater_reliability)
    max_val_1 = np.nanmax(interrater_reliability) 

    min_val_2 = np.nanmin(ingroup)
    max_val_2 = np.nanmax(ingroup) 
    
    min_val = np.nanmin([min_val_1,min_val_2])
    max_val = np.nanmax([max_val_1,max_val_2]) 
   
    if man_max is not None:
        max_val = man_max 
   
    if man_min is not None:
        min_val = man_min

    sns.heatmap(ingroup, ax=ax0, annot=True, fmt=".1%", cbar=False, xticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap)
    sns.heatmap(interrater_reliability, ax=ax1,annot=True, fmt=".1%", yticklabels=False, vmin=min_val, vmax=max_val, cbar=False,cmap=cmap)
    sns.heatmap(avg, ax=ax3, annot=True, fmt=".1%", cbar=True, xticklabels=False, yticklabels=False,vmin=min_val, vmax=max_val,cmap=cmap)

    ax0.set_title("In")
    ax0.set(xlabel='group', ylabel='Group')
    ax0.xaxis.set_label_coords(0.5, 1.20)

    ax1.set_title("Between")
    ax1.set(xlabel='groups', ylabel='')
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_coords(0.5, 1.20)

    ax3.set(xlabel='AVG', ylabel='')
    ax3.xaxis.set_label_coords(0.5, 1.20)

    # vertical line
    if size > 6:
        pos_line = 0.228
    else:
        pos_line = 0.248

    l1 = lines.Line2D([pos_line, pos_line], [0.1, 1.1],color="grey", transform=fig.transFigure, figure=fig)
    fig.lines.extend([l1])
    
    fig.savefig(path_results+"interrater_"+selected_type+"_"+str(len(group_dfs_each))+"-groups.pdf", bbox_inches='tight')
    fig.savefig(path_results+"interrater_"+selected_type+"_"+str(len(group_dfs_each))+"-groups.png", bbox_inches='tight')

    return (ingroup,interrater_reliability,avg)