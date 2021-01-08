import pandas as pd
import krippendorff
import numpy as np
import math
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

def normalize(to_be_normalized, min_val, max_val,offset=0):
    return (to_be_normalized - min_val)/(max_val-min_val) + offset

def normalizeMatrix(matrix, offset=0):
    new_matrix = matrix.copy()
    matrix_wo_zeros = np.delete(new_matrix, np.argwhere(new_matrix == 0))
    min_val = np.min(matrix_wo_zeros)
    max_val = np.max(matrix_wo_zeros)
    
    for i in range(0,len(new_matrix)):
        for j in range(0,len(new_matrix[0])):
            if new_matrix[i][j] > 0:
                new_matrix[i][j] = normalize(new_matrix[i][j], min_val, max_val,offset)
    return new_matrix

def compute_d(x, y):
    """
    If b is increased or c is decreased, 
    then compute_d(0, 1) increases
    
    """
    pr1 = 10
    pr2 = 0.25 # 0.5
    deg = 5 # 15
    in_rad = np.deg2rad(deg)
    
    #print("in_rad: ", in_rad)
    cos_deg = np.cos(in_rad) 
    sin_deg = np.sin(in_rad)

    Q = np.array([[pr1, 0], [0, pr2]])
    
    R = np.array([[cos_deg, - sin_deg],[sin_deg, cos_deg]])
    #print("R: ", R)
    
    # actual computation:
    # 1. translation:
    p = np.array([x-1, y-1])
    
    # 2. rotation:    
    p_rotated = (R.dot(p))#.reshape(2,)
    #print("p_rotated: ", p_rotated)
    #print(p_rotated)
    # 3. Quadrik computation
    tmp = Q.dot(p_rotated)  
    output = p_rotated.dot(tmp)
    
    return(output)

def heuristic_d(overlap,agreement):
    matrix = np.zeros(shape=(20,11))

    len_r = len(matrix)
    len_c = len(matrix[0])

    value_0_1 = 0.7
    value_100_1 = 1.1
    value_0_20 = 0.5
    value_100_20 = 1.5

    matrix[0][0] = value_0_1
    matrix[0][len_c-1] = value_100_1
    matrix[len_r-1][0] = value_0_20
    matrix[len_r-1][len_c-1] = value_100_20

    # first and last row
    for i in range(1,len_c-1):
        matrix[0][i] = matrix[0][0] + (matrix[0][len_c-1]-matrix[0][0])*i/(len_c-1)
        matrix[len_r-1][i] = matrix[len_r-1][0] + (matrix[len_r-1][len_c-1]-matrix[len_r-1][0])*i/(len_c-1)

    for r in range(1,len_r-1):
        for c in range(0,len_c):
            matrix[r][c] = matrix[0][c] + (matrix[len_r-1][c]-matrix[0][c])*r/(len_r-1)
            
    over = int(overlap)-1
    agree = int(round(agreement*10))
    return matrix[over][agree]

def getWeightMatrix(df,min_overlap=1):
    df_matrix = df.to_numpy() 
    number_workers = np.size(df_matrix,1)
    # empty distance matrix
    distance_matrix = np.zeros((number_workers,number_workers))
    list_over = []
    for i in tqdm(range(0,number_workers)):
    ##for i in range(0,number_workers):
        #print(i)
        for j in range(i+1,number_workers):
            agreed = 0
            disagreed = 0
            weight = 0
            
            overlap = df_matrix[i]+df_matrix[j]
            overlap = overlap[~np.isnan(overlap)]
            list_over.append(len(overlap))
            for k in overlap:
                if k == 1:
                    disagreed += 1
                else:
                    agreed += 1
                    
            if len(overlap) >= min_overlap:
                # call distance function
                agreement_rate = agreed/(agreed+disagreed)
                weight = agreement_rate + 0.5
            
            distance_matrix[i,j] = weight
            distance_matrix[j,i] = weight
    
    return (distance_matrix,list_over)

def getWeightHeuristicMatrix(df,min_overlap=1):
    df_matrix = df.to_numpy() 
    number_workers = np.size(df_matrix,1)
    # empty distance matrix
    distance_matrix = np.zeros((number_workers,number_workers))
    list_over = []
    for i in tqdm(range(0,number_workers)):
        for j in range(i+1,number_workers):
            agreed = 0
            disagreed = 0
            weight = 0
            
            overlap = df_matrix[i]+df_matrix[j]
            overlap = overlap[~np.isnan(overlap)]
            list_over.append(len(overlap))
            for k in overlap:
                if k == 1:
                    disagreed += 1
                else:
                    agreed += 1
                    
            if len(overlap) >= min_overlap:
                # call distance function
                agreement_rate = agreed/(agreed+disagreed)
                weight = heuristic_d(len(overlap), agreement_rate)

            distance_matrix[i,j] = weight
            distance_matrix[j,i] = weight
    
    return (distance_matrix,list_over)

def getWeightCohensKappaMatrix(df,min_overlap=1):
    df_matrix = df.to_numpy() 
    number_workers = np.size(df_matrix,1)
    # empty distance matrix
    distance_matrix = np.zeros((number_workers,number_workers))
    list_over = []
    for i in tqdm(range(0,number_workers)):
        for j in range(i+1,number_workers):
            weight = 0
            annotator_1 = df_matrix[i]
            annotator_2 = df_matrix[j]
            
            annotator_1_cleaned = annotator_1[~np.isnan(annotator_2)]  
            annotator_2_cleaned = annotator_2[~np.isnan(annotator_1)]  
            
            annotator_1_cleaned = annotator_1_cleaned[~np.isnan(annotator_1_cleaned)]  
            annotator_2_cleaned = annotator_2_cleaned[~np.isnan(annotator_2_cleaned)]  
            
            len_overlap = len(annotator_2_cleaned)
            list_over.append(len_overlap)
                    
            if len_overlap >= min_overlap:
                try:
                    cohen = cohen_kappa_score(annotator_1_cleaned,annotator_2_cleaned)
                except RuntimeWarning:
                    print(annotator_1_cleaned)
                    print(annotator_2_cleaned)
                if cohen < 2:
                    weight = 0.5+ (cohen +1)/(2)                   

            distance_matrix[i,j] = weight
            distance_matrix[j,i] = weight
    
    return (distance_matrix,list_over)

def getWeightKrippendorffMatrix(df,min_overlap=1):
    df_matrix = df.to_numpy() 
    number_workers = np.size(df_matrix,1)
    # empty distance matrix
    distance_matrix = np.zeros((number_workers,number_workers))
    list_over = []
    for i in tqdm(range(0,number_workers)):
        for j in range(i+1,number_workers):
            weight = 0
            annotator_1 = df_matrix[i]
            annotator_2 = df_matrix[j]
            
            annotator_1_cleaned = annotator_1[~np.isnan(annotator_2)]  
            annotator_2_cleaned = annotator_2[~np.isnan(annotator_1)]  
            
            annotator_1_cleaned = annotator_1_cleaned[~np.isnan(annotator_1_cleaned)]  
            annotator_2_cleaned = annotator_2_cleaned[~np.isnan(annotator_2_cleaned)]  
            
            len_overlap = len(annotator_2_cleaned)
            list_over.append(len_overlap)
                    
            if len_overlap >= min_overlap:
                try:
                    kd_value = krippendorff.alpha(reliability_data=[annotator_1_cleaned,annotator_2_cleaned],                               
                                          level_of_measurement='nominal')
                except RuntimeWarning:
                    print(annotator_1_cleaned)
                    print(annotator_2_cleaned)
                if kd_value < 2:
                    weight = 0.5+ (kd_value +1)/(2)
                    #weight = 0.5+ kd_value 
                                   

            distance_matrix[i,j] = weight
            distance_matrix[j,i] = weight
    
    return (distance_matrix,list_over)