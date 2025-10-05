import pandas as pd 
import numpy as np
import math
import ast

import glob
import os
import random
from time import process_time

from scipy.stats import pearsonr, spearmanr
from scipy.spatial import distance

import warnings
warnings.filterwarnings("ignore")

import sys
from functools import cmp_to_key
from  sklearn.preprocessing import normalize
from ast import literal_eval

def compare(item1, item2):
    # Função utilitária de comparação entre dois valores.
    # Retorna -1 se item1 < item2, 1 se item1 > item2 e 0 se iguais.
    # Usada para ordenações compatíveis com cmp_to_key quando necessário.
    if item1 < item2:
        return -1
    elif item1 > item2:
        return 1
    else:
        return 0




def getDiametersDistances(matrixDistances):
    # Calcula o "diâmetro" (máximo) das distâncias armazenadas em cada célula de
    # `matrixDistances` e retorna um DataFrame com um valor por coluna (atributo).
    # Algumas células podem ser NaN ou float (não-iteráveis). Essa versão trata
    # esses casos de forma robusta: tenta obter max(cell) quando possível e
    # usa NaN caso contrário; depois calcula o máximo por coluna ignorando NaNs.

    diametersMatrix = pd.DataFrame(index=matrixDistances.index, columns=matrixDistances.columns)
    finalDiameters = pd.DataFrame(index=[0], columns=np.arange(0, len(matrixDistances.columns)))

    for tup in range(len(matrixDistances)):
        for col in range(len(matrixDistances.columns)):
            cell = matrixDistances.iloc[tup, col]
            try:
                diam_val = max(cell)
            except Exception:
                diam_val = np.nan
            diametersMatrix.iloc[tup, col] = diam_val

    # Para cada coluna calcula o máximo ignorando NaNs
    for col in range(len(diametersMatrix.columns)):
        col_series = pd.to_numeric(diametersMatrix.iloc[:, col], errors='coerce')
        finalDiameters.iloc[0, col] = col_series.max()

    return finalDiameters



def generateDistanceMaps(data, function, dir_path):
    # Gera um DataFrame onde cada célula contém a lista de distâncias entre o
    # vetor do objeto (linha) e todos os outros vetores válidos da mesma coluna.
    # Parâmetros:
    # - data: DataFrame N x M com cada célula sendo um vetor numpy/lista ou NaN.
    # - function: string 'euclidean'|'manhattan'|'chebyshev' indicando a métrica.
    # - dir_path: caminho onde será salvo o arquivo de diâmetros.
    # Retorna: DataFrame de listas de distâncias (mesmo formato de `data`, mas
    # cada célula é a lista de distâncias do objeto com os demais).

    distanceMatrixValues = pd.DataFrame(index=data.index, columns=data.columns)

    #loops (tup x objs) to iterate each object of dataframe
    for atr in range(len(data.columns)):
        for tup in range(len(data)):
#             # Verify if the choose obj is a missing data
#             # IF NOT missing data, continue the process...
#             # IF YES, choose the next object of dataframe
            if(np.isnan(data.iloc[tup, atr]).any() == False):

                obj = data.iloc[tup, atr]
                objDistance = []

                for i in range(len(data)):

                    #  Verify if the iterate object is missing data
                    if(np.isnan(data.iloc[i,atr]).any()):
                           continue
                    else:
                        if (function == 'euclidean'):
                            # create equation to calculate matrix with euclidean
                            objDistance += [distance.euclidean(obj, data.iloc[i,atr])]

                        elif (function == 'chebyshev'):
                            # create equation to calculate matrix with chebyshev
                            objDistance += [distance.chebyshev(obj, data.iloc[i,atr])]

                        elif (function == 'manhattan'):
                            # create equation to calculate matrix with manhattan/cityblock
                            objDistance += [distance.cityblock(obj, data.iloc[i,atr])]
                                
            else:
                objDistance = np.nan
        
            distanceMatrixValues.iloc[tup, atr] = objDistance
    
    diametersMatrix = getDiametersDistances(distanceMatrixValues)
    
    diametersMatrix.to_pickle(dir_path+'matrix_diameters.pkl')
            
    return distanceMatrixValues


def corDiS(data, correlation , dir_path):
    # Calcula uma matriz de correlação entre atributos usando as distâncias
    # (cada entrada da matriz 'data' é uma lista/array de distâncias por tupla).
    # Para cada par de atributos (i,j) a função coleta coeficientes de
    # correlação (Pearson ou Spearman) entre vetores correspondentes por linha
    # e faz a média (por linha) para popular a matriz de correlação.
    # Salva o resultado em pickle em `dir_path`.

    corrMatrix = pd.DataFrame(index = data.columns,columns=data.columns, dtype=float)
        
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            p = []
            for k in range(len(data)):
                if(np.isnan(data.iloc[k,j]).any() == False and np.isnan(data.iloc[k,i]).any() == False):

                    if(correlation == 'pearson'):p += [pearsonr(data.iloc[k,i], data.iloc[k,j])[0]]
                        
                    elif(correlation == 'spearman'): p += [spearmanr(data.iloc[k,i], data.iloc[k,j])[0]]
                        
            corrMatrix.iloc[i,j] = (np.sum(p))/len(data)

    corrMatrix.to_pickle(dir_path+'matrix_{}.pkl'.format(correlation))
        
    return corrMatrix


# In[5]:


def normalizeCompatibleAttributes (matrixComp):
    # Normaliza fatores de compatibilidade em cada linha da tabela de
    # compatibilidade `matrixComp`. Espera que a coluna 'Fact_attributes'
    # contenha listas de pesos; essa função divide cada peso pela soma
    # total e armazena a lista normalizada em 'Fact_norm'. Retorna o DataFrame
    # atualizado (com 'Fact_norm' convertido de string para lista via literal_eval).

    # manipulate each columns(attribute) of patient tuple
    for i in range(len(matrixComp)):

        #iterate between each exam            
        fatct_comp = matrixComp.loc[i]['Fact_attributes']
        sum_fatc = sum(matrixComp.iloc[i]['Fact_attributes'])
        
        new_factor = []
        for j in range(len(fatct_comp)):
            new_factor += [fatct_comp[j]/sum_fatc] 
     
        matrixComp.loc[i, 'Fact_norm'] =  str(new_factor)
    matrixComp['Fact_norm'] = matrixComp['Fact_norm'].apply(literal_eval)

    return matrixComp
    
def findCompatibleAttributes(data, matrixCorr, threshold, k_attributes, dir_path, correlation):   
    # Para cada atributo (linha em matrixCorr) encontra até k_attributes outros
    # atributos cuja correlação é >= threshold. Gera um DataFrame que mapeia
    # cada atributo para a lista de atributos compatíveis e seus fatores
    # (Fact_attributes). Normaliza os fatores e salva o resultado em pickle
    # dentro de uma subpasta baseada no tipo de correlação.

    # Path of output directory
    matrixCompatibleAtr = pd.DataFrame(columns=['Threshold','K_Value','Comp_attributes','Fact_attributes'])

    # Path of matrixTCompatibleAttributes 
    output_path = dir_path+'matrixCompatibility_{}/'.format(correlation)
    try:
        # Create target Directory
        os.mkdir(output_path)
    except FileExistsError:
        pass

    for row in range(len(matrixCorr)): # iterate rows in matrixThreshold 
        list_comp_atr  = []
        list_factc = []
        for col in range(len(matrixCorr.columns)):  # iterate columns in matrixThreshold 
            if(matrixCorr.iloc[row,col] >= threshold and (row is not col) and (len(list_comp_atr) < k_attributes)):

                list_comp_atr  += [col]
                list_factc += [matrixCorr.iloc[row,col]]

        matrixCompatibleAtr.loc[len(matrixCompatibleAtr)] = (threshold, k_attributes,list_comp_atr,list_factc)

    matrixCompatibleAtr = normalizeCompatibleAttributes(matrixCompatibleAtr)

    matrixCompatibleAtr.to_pickle(output_path+'matrix_CompatibleAtr_k_'+str(k_attributes)+'_th_'+str(threshold)+'.pkl')    


# In[6]:


def processingSOLID(input_path, output_path, distance_function, correlation, threshold, max_K):
    
    output_path = output_path+'CorDis/'
    
    try:
        # Create target Directory
        os.mkdir(output_path)
    except FileExistsError:
        pass
        
    train_data = pd.read_pickle(input_path+'train_data.pkl')

    ## METHOD OF DISTANCE MAPS #
    matrixValues = generateDistanceMaps(train_data, distance_function, output_path)
    
    ## METHOD OF CORRELATION MAPS #
    matrixCorr = corDiS(matrixValues, correlation, output_path)
    
    for k_attributes in range(1,max_K+1,2):

        ## METHOD OF FIND HIGLY CORRELATED ATTRIBUTES #
        findCompatibleAttributes(train_data, matrixCorr, threshold, k_attributes, output_path, correlation)  


## CoDiS METHO DESCRIPTION
## Fixed Parameters
## Arg1 : Input_path: 'String Format'
## Arg2 : Output_path: 'String Format'
## Arg3 : Distance Function: 'String Format' -> Euclidean, Manhattan and Chebyshev are avaliable
## Arg4 : Correlation Measure: 'String Format' -> Pearson or Spearman are avaliable
## Arg5 :Threshold Correlation: 'Float Format' -> (0,1]
## Arg6 :K_attributes: Integer Format -> Amount of neighboors attributes compatible

## To run python CorDiS.py Arg1 Arg2 Arg3 Arg4 Arg5 Arg6
def main(argv):
    input_path = ''
    output_path = ''
    df = ''
    correlation = ''
    threshold = 0
    k_attributes = 0

    if(len(argv) == 7):
        if(argv[1] is not ''):
            input_path = argv[1]
            
        if(argv[2] is not ''):
            output_path = argv[2]
              
        if(argv[3] is not None):
            df = argv[3]
            
        if(argv[4] is not None):
            correlation = argv[4]

        if(argv[5] is not None):
            threshold = float(argv[5])
                          
        if(argv[6] is not None):
            max_knn = int(argv[6])
        
        processingSOLID(input_path, output_path, distance_function, correlation, max_knn) # type: ignore

    else:
        print('Parameters Not Found!...')
        sys.exit()
    
if __name__ == "__main__": 
    main(sys.argv)

