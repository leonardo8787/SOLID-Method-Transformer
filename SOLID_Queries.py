#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import matplotlib.pyplot as plt
from Submodules import Functions

def any_nans(a):
    # Verifica se algum elemento do iterável `a` (cada elemento é esperado ser
    # um vetor numpy/lista) contém NaNs. Retorna True caso algum possua NaNs.
    # Útil para decidir se um atributo/coluna possui valores faltantes.
    for x in a:
        if np.isnan(x).any(): 
            return True
    return False

def compare(item1, item2):
    # Função de comparação padrão que retorna -1/0/1. Útil para compatibilidade
    # com APIs que aceitam uma função de comparação (cmp_to_key).
    if item1 < item2:
        return -1
    elif item1 > item2:
        return 1
    else:
        return 0

def jaccard_similarity(list1, list2):
    # Calcula a similaridade de Jaccard entre duas listas (|A ∩ B| / |A ∪ B|).
    # Útil para medir sobreposição entre conjuntos de índices/atributos.
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# In[4]:


def Test_SOLID_Queries(matrixAttributesMissing, matrixComp, diameters, k, id_objQuery):
    # Função de teste que executa uma consulta SOLID ponderada (Similarity_Queries_Weight)
    # usando um `matrixAttributesMissing` (dataset com valores faltantes) e o id
    # do objeto-consulta. Retorna o conjunto de resultados ponderados (IDs dos
    # vizinhos mais próximos). Observação: espera que `matrixAttributesComplete`
    # esteja disponível no escopo superior se necessário (ainda que aqui não seja
    # usado ativamente para a consulta em si).
    result_Weight = []

    objQuery = matrixAttributesComplete.loc[id_objQuery]
    
    # Define objeQuery Missing (Same tuple of Complete)
    objQueryMissing = matrixAttributesMissing.loc[id_objQuery].copy()
    
    #     # SOLID Similarity Query
    result_Weight = Functions.Similarity_Queries_Weight(matrixAttributesMissing.copy(), matrixComp, diameters, k, objQueryMissing.copy())

    return result_Weight


# In[5]:


def Test_ObjQueryMissing(matrixAttributesComplete, missing,matrixComp, diameters, k, id_objQuery):
    # Gera uma versão do objeto-consulta com valores faltantes (percentual `missing`)
    # a partir de um objeto completo e executa a consulta SOLID para verificar o
    # comportamento do sistema diante de atributos ausentes. Retorna os vizinhos
    # mais próximos encontrados.
    result_Weight = []

    objQuery = matrixAttributesComplete.loc[id_objQuery]
    
   
    # Define objeQuery Missing (Same tuple of Complete)
    objQueryMissing = matrixAttributesComplete.loc[id_objQuery].copy()

    pos = np.random.randint(0,len(objQueryMissing),round(len(objQueryMissing)*(missing/100))+1)
    
    for atr in range(len(objQueryMissing)):
        if(atr in pos):
            objQueryMissing.iloc[atr] = np.nan
            
    #     # SOLID Similarity Query
    result_Weight = Queries.Similarity_Queries_Weight(matrixAttributesComplete.copy(), matrixComp, diameters, k, objQueryMissing.copy())

    return result_Weight


# ## SOLID QUERY DESCRIPTION
# ## Fixed Parameters
# ## Arg1 = Input_path: 'String Format'
# ## Arg2 = Output_path: 'String Format'
# ## Arg3 = K Neartest Neighbors to retrieve: Double Format [0~Infinite Value] 
# ## Arg4 = Object Query: -> Integer Format [0~ Len of Tuples of Dataset -1] 

# ## To run python SOLID_Queries.py Arg1 Arg2 Arg3 Arg4

def main(argv):
    input_path = ''
    output_path = ''
    k = 0
    objQuery = 0
    
    # Função main que serve como interface de linha de comando. Lê argumentos
    # (input_path, output_path, k, objQuery) e tenta carregar os pickles
    # necessários para executar uma consulta SOLID. Observação: atualmente o
    # código espera os três pickles no mesmo caminho e chama uma função
    # `SOLID(...)` que não está definida aqui — portanto é necessário ajustar os
    # nomes dos arquivos e o fluxo para uso direto.
    if(len(argv) == 5):
        if(argv[1] is not ''):
            input_path = argv[1]
        if(argv[2] is not ''):
            output_path = argv[2]
        if(argv[3] is not None):
            k = int(argv[3])
        if(argv[4] is not None):
            objQuery = int(argv[4])

        data_missing = pd.read_pickle(input_path)
        matrixComp = pd.read_pickle(input_path)
        diameters = pd.read_pickle(input_path)

        SOLID(data_missing, matrixComp, diameters, k, objQuery)
    else:
        print('Parameters Not Found!...')
        sys.exit()   
        
if __name__ == "__main__": 
    main(sys.argv)

