#!/usr/bin/env python
# coding: utf-8
   
import numpy as np
import pandas as pd
import os
import ast
import glob 
import random

from time import process_time
from scipy.spatial import distance
from sklearn.metrics import jaccard_score

from ast import literal_eval
import sys
from functools import cmp_to_key
from itertools import combinations 

from sklearn.impute import SimpleImputer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer 

import warnings
warnings.filterwarnings("ignore")

def any_nans(a):
    for x in a:
        if np.isnan(x).any(): 
            return True
    return False

def compare(item1, item2):
    if item1 < item2:
        return -1
    elif item1 > item2:
        return 1
    else:
        return 0

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    
    return float(intersection) / union


def simpleImputation(data, objQuery):
    for atr in range(len(data.columns)):
        data.iloc[:,atr] = data.iloc[:,atr].apply(lambda x: np.array(x))
        
    for atr in range(len(data.columns)):     

        for tup in range(len(data)):

            if(np.isnan(data.iloc[tup,atr]).all()): 

                data.iloc[tup,atr] = np.sum(data.iloc[:,atr])/len(data) 

        if(np.isnan(objQuery.iloc[atr]).all()): objQuery.iloc[atr] = data.iloc[atr,atr]/len(data) 

    return data, objQuery

def knnImputation(data, objQuery):
    for atr in range(len(data.columns)):
        data.iloc[:,atr] = data.iloc[:,atr].apply(lambda x: np.array(x))
        
    imputer_KNN = KNNImputer(n_neighbors=len(data)*.2)

    for atr in range(len(data.columns)):  

        if(any_nans(data.iloc[:,atr])):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = (imputer_KNN.fit_transform(null_data))
            for tup in range(len(data)): data.iloc[tup,atr] = imputed_data[tup]

        if(np.isnan(objQuery.iloc[atr]).all()):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = (imputer_KNN.fit_transform(null_data))
            for tup in range(len(data)): 
                objQuery.iloc[atr] = imputed_data[tup]         
                
    return data, objQuery  

def regressorImputation(data, objQuery):
    
    for atr in range(len(data.columns)):
        data.iloc[:,atr] = data.iloc[:,atr].apply(lambda x: np.array(x))
        
    imputer_Reg = IterativeImputer(random_state=0, estimator=KNeighborsRegressor(n_neighbors=3))
    
    for atr in range(len(data.columns)):  

        if(any_nans(data.iloc[:,atr])):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = imputer_Reg.fit_transform(null_data)

            for tup in range(len(data)): data.iloc[tup,atr] = imputed_data[tup]

        if(np.isnan(objQuery.iloc[atr]).all()):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = imputer_Reg.fit_transform(null_data)

            for tup in range(len(data)): 
                objQuery.iloc[atr] = imputed_data[tup]
                
    return data, objQuery

def decisionTreeImputation(data, objQuery):
    for atr in range(len(data.columns)):
        data.iloc[:,atr] = data.iloc[:,atr].apply(lambda x: np.array(x))
    
    imputer_Reg = IterativeImputer(random_state=0, estimator=    DecisionTreeRegressor(max_features='sqrt', random_state=0))
    
    for atr in range(len(data.columns)):  

        if(any_nans(data.iloc[:,atr])):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = imputer_Reg.fit_transform(null_data)

            for tup in range(len(data)): data.iloc[tup,atr] = imputed_data[tup]

        if(np.isnan(objQuery.iloc[atr]).all()):
            len_vector = 0
            for tup in range(len(data)):
                if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

            null_data = np.empty([len(data),len_vector])

            for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

            imputed_data = imputer_Reg.fit_transform(null_data)

            for tup in range(len(data)): 
                objQuery.iloc[atr] = imputed_data[tup]


    return data, objQuery


def Similarity_Queries_Weight(matrixAttributes, matrixComp, diameters, k, objQuery):
    
    matrixResultSetGlobal = pd.DataFrame(index = matrixAttributes.index, columns = np.arange(0, len(matrixAttributes.columns)))
    KnnResultSetGlobal = pd.DataFrame(index = matrixAttributes.index, columns = ['ID_OBJ', 'Tuple_Distance'])

    finalResultSetLocal = []

    attributes_compatible = []
    weights_sq = []
    attribute_nan = []

    for atr in range(len(matrixAttributes.columns)):

        for i in range(len(objQuery)):
            if(np.isnan(objQuery.iloc[i]).any()): 
                attribute_nan += [i]

        if(atr in attribute_nan):
            attributes_compatible += matrixComp.loc[attribute_nan.index(atr), 'Comp_attributes']
            weights_sq += matrixComp.loc[attribute_nan.index(atr),'Fact_norm']
   
    for atr in range(len(matrixAttributes.columns)):
        matrixResultSetLocal = pd.DataFrame(index=matrixAttributes.index, columns = [atr])   
   
        #weigh objquery attr_i (case one)
        if(any_nans(objQuery) == True and any_nans(matrixAttributes.iloc[:,atr]) == False): 
    
            if(atr in attributes_compatible):

                for tup in range(len(matrixAttributes)):
                    
                    if(np.isnan(objQuery.iloc[atr]).all() != True):

                        dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])  
                                        
                        matrixResultSetLocal.loc[tup,atr] = (1 - weights_sq[attributes_compatible.index(atr)]) * (dist/ diameters.iloc[0,atr])          

            else:

                if(atr not in attribute_nan):

                    for tup in range(len(matrixAttributes)): 
                        if(np.isnan(objQuery.iloc[atr]).all() != True and np.isnan(matrixAttributes.iloc[tup,atr]).all() != True):

                            dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])

                            if( diameters.iloc[0,atr]  == 0):  diameters.iloc[0,atr] = 1
                            
                            matrixResultSetLocal.loc[tup,atr] = dist/ diameters.iloc[0,atr]              

        #weigh tuple with missing (case two)
        elif(any_nans(matrixAttributes.iloc[:,atr]) == True and any_nans(objQuery) == False):
            if(atr in attributes_compatible):

                for tup in range(len(matrixAttributes)):

                    dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])  
                    
                    if( diameters.iloc[0,atr]  == 0):  diameters.iloc[0,atr] = 1

                    matrixResultSetLocal.loc[tup,atr] = (1 - weights_sq[attributes_compatible.index(atr)]) * (dist/ diameters.iloc[0,atr])          

            else:

                if(atr not in attribute_nan):
                    
                    for tup in range(len(matrixAttributes)):
                        if(np.isnan(objQuery.iloc[atr]).all() != True and np.isnan(matrixAttributes.iloc[tup,atr]).all() != True):

                            dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])

                            matrixResultSetLocal.loc[tup,atr] = dist/ diameters.iloc[0,atr]              

        
        #weigh tuple and objQuery with missing (case three)
        elif(any_nans(objQuery) == True and any_nans(matrixAttributes.iloc[:,atr]) == True):
            if(atr in attributes_compatible):

                for tup in range(len(matrixAttributes)):
                    if(np.isnan(objQuery.iloc[atr]).all() != True and np.isnan(matrixAttributes.iloc[tup,atr]).all() != True):

                        dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])  

                        matrixResultSetLocal.loc[tup,atr] = (1 - weights_sq[attributes_compatible.index(atr)]) * (dist/ diameters.iloc[0,atr])          

            else:

                if(atr not in attribute_nan):

                    for tup in range(len(matrixAttributes)):
                        if(np.isnan(objQuery.iloc[atr]).all() != True and np.isnan(matrixAttributes.iloc[tup,atr]).all() != True):

                            dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])
                            
                            matrixResultSetLocal.loc[tup,atr] = dist/ diameters.iloc[0,atr]              

        else: # none missing (case four)
            
            for tup in range(len(matrixAttributes)):
                
                dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])

                if( diameters.iloc[0,atr]  == 0):  diameters.iloc[0,atr] = 1
                matrixResultSetLocal.loc[tup,atr] = dist/ diameters.iloc[0,atr]         
        
        finalResultSetLocal += [matrixResultSetLocal.values]
        
    ############################# Create Global ResultSet #############################
    
    for i in range(len(finalResultSetLocal)):
        matrixResultSetGlobal.loc[:,i] = finalResultSetLocal[i] 
    
    for i in range(len(matrixResultSetGlobal)):
        KnnResultSetGlobal.iloc[i] = (i,matrixResultSetGlobal.iloc[i].sum(axis=0))
     
    KnnResultSetGlobal = KnnResultSetGlobal.sort_values(by='Tuple_Distance')
    
    return KnnResultSetGlobal.iloc[1:k+1,0].values
            
def Similarity_Queries(matrixAttributes, diameters, k, objQuery):
    
    matrixResultSetGlobal = pd.DataFrame(index = matrixAttributes.index, columns = np.arange(0, len(matrixAttributes.columns)))
    KnnResultSetGlobal = pd.DataFrame(index = matrixAttributes.index, columns = ['ID_OBJ', 'Tuple_Distance'])
    
    finalResultSetLocal = []   
    
    for atr in range(len(matrixAttributes.columns)):
        
        matrixResultSetLocal = pd.DataFrame(index=matrixAttributes.index, columns = [atr]) 
        
        if(any_nans(objQuery) != True and any_nans(matrixAttributes.iloc[:,atr]) != True):

            for tup in range(len(matrixAttributes)):
                dist = distance.euclidean(objQuery.iloc[atr], matrixAttributes.iloc[tup,atr])
               
                matrixResultSetLocal.loc[tup,atr] = dist/ diameters.iloc[0,atr]

        else: atr+=1
        
        finalResultSetLocal += [matrixResultSetLocal.values]        
   
    for i in range(len(finalResultSetLocal)):
        matrixResultSetGlobal.loc[:,i] = finalResultSetLocal[i] 
        

    for i in range(len(matrixResultSetGlobal)):
        KnnResultSetGlobal.iloc[i] = (i,matrixResultSetGlobal.iloc[i].sum(axis=0))
     
    KnnResultSetGlobal = KnnResultSetGlobal.sort_values(by='Tuple_Distance')
    
    return KnnResultSetGlobal.iloc[1:k+1,0].values

def makeImputation(data, type_imp, objQuery):
        
    for atr in range(len(data.columns)):
        data.iloc[:,atr] = data.iloc[:,atr].apply(lambda x: np.array(x))
        
    if(type_imp == 'simple'):

        for atr in range(len(data.columns)):     
            
            for tup in range(len(data)):
                
                if(np.isnan(data.iloc[tup,atr]).all()): 
                
                    data.iloc[tup,atr] = np.sum(data.iloc[:,atr])/len(data)
               
        
    elif(type_imp == 'knni'):

        imputer_KNN = KNNImputer(n_neighbors=len(data)*.2)

        for atr in range(len(data.columns)):  

            if(any_nans(data.iloc[:,atr])):
                len_vector = 0
                for tup in range(len(data)):
                    if(np.isnan(data.iloc[tup,atr]).all() == False): len_vector = len(data.iloc[tup,atr])

                null_data = np.empty([len(data),len_vector])

                for tup in range(len(data)): null_data[tup] = data.iloc[tup,atr]

                imputed_data = (imputer_KNN.fit_transform(null_data))
                for tup in range(len(data)): data.iloc[tup,atr] = imputed_data[tup]
             
                
    return data, objQuery