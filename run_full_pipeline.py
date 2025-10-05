#!/usr/bin/env python3
"""
Orquestrador para executar o pipeline completo:
 1) converte um CSV em train_data.pkl (usa função interna similar à do run_solid_pipeline)
 2) executa CorDiS.processingSOLID para gerar mapas e compatibilidades
 3) executa uma consulta SOLID (Similarity_Queries_Weight) e salva os vizinhos

Uso:
  python run_full_pipeline.py <csv-file> <output-dir> [distance] [correlation] [threshold] [max_K] [obj_idx] [k_query] [comp_k]

Exemplo:
  python run_full_pipeline.py 1-DS-MSTSpine.csv ./work_mstspine euclidean pearson 0.8 5 0 5 1

"""
import os
import sys
import pandas as pd
import numpy as np
from ast import literal_eval


def parse_cell(cell):
    if pd.isna(cell) or (isinstance(cell, str) and cell.strip() == ''):
        return np.nan
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.array(cell)
    if isinstance(cell, str):
        s = cell.strip()
        try:
            v = literal_eval(s)
            if isinstance(v, (list, tuple)):
                return np.array(v)
            else:
                return np.array([v])
        except Exception:
            for sep in [';', ',']:
                if sep in s:
                    parts = [p.strip() for p in s.split(sep) if p.strip() != '']
                    try:
                        nums = [float(p) for p in parts]
                        return np.array(nums)
                    except Exception:
                        break
            try:
                return np.array([float(s)])
            except Exception:
                return np.nan
    if isinstance(cell, (int, float, np.floating, np.integer)):
        return np.array([cell])
    return np.nan


def csv_to_train_pickle(csv_path, out_dir):
    df = pd.read_csv(csv_path, dtype=str)
    out = pd.DataFrame(index=df.index, columns=df.columns)
    for col in df.columns:
        for idx, val in df[col].items():
            out.at[idx, col] = parse_cell(val)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, 'train_data.pkl')
    out.to_pickle(pkl_path)
    return pkl_path


def find_compatibility_file(comp_dir, comp_k, threshold):
    # tenta caminho explícito
    target = os.path.join(comp_dir, f'matrix_CompatibleAtr_k_{comp_k}_th_{threshold}.pkl')
    if os.path.exists(target):
        return target
    # fallback: pega primeiro pkl da pasta
    if not os.path.isdir(comp_dir):
        return None
    for f in os.listdir(comp_dir):
        if f.endswith('.pkl'):
            return os.path.join(comp_dir, f)
    return None


def main(argv):
    if len(argv) < 3:
        print('Uso: python run_full_pipeline.py <csv-file> <output-dir> [distance] [correlation] [threshold] [max_K] [obj_idx] [k_query] [comp_k]')
        sys.exit(1)

    csv_file = argv[1]
    output_dir = argv[2]
    distance = argv[3] if len(argv) > 3 else 'euclidean'
    correlation = argv[4] if len(argv) > 4 else 'pearson'
    threshold = float(argv[5]) if len(argv) > 5 else 0.8
    max_K = int(argv[6]) if len(argv) > 6 else 5
    obj_idx = int(argv[7]) if len(argv) > 7 else 0
    k_query = int(argv[8]) if len(argv) > 8 else 5
    comp_k = int(argv[9]) if len(argv) > 9 else 1

    # preparar diretório
    os.makedirs(output_dir, exist_ok=True)

    print('Convertendo CSV -> train_data.pkl')
    csv_to_train_pickle(csv_file, output_dir)

    # garantir barra para compatibilidade com CorDiS
    input_path = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep
    out_path = output_dir if output_dir.endswith(os.sep) else output_dir + os.sep

    # executar CorDiS
    try:
        import CorDiS
    except Exception as e:
        print('Não foi possível importar CorDiS:', e)
        sys.exit(1)

    print('Executando CorDiS.processingSOLID...')
    CorDiS.processingSOLID(input_path, out_path, distance, correlation, threshold, max_K)

    cordis_dir = os.path.join(out_path, 'CorDis')
    # carregar resultados gerados
    train_pkl = os.path.join(input_path, 'train_data.pkl')
    diameters_pkl = os.path.join(cordis_dir, 'matrix_diameters.pkl')

    if not os.path.exists(train_pkl):
        print('train_data.pkl não encontrado em', train_pkl)
        sys.exit(1)
    if not os.path.exists(diameters_pkl):
        print('matrix_diameters.pkl não encontrado em', diameters_pkl)
        sys.exit(1)

    train = pd.read_pickle(train_pkl)
    diameters = pd.read_pickle(diameters_pkl)

    comp_dir = os.path.join(cordis_dir, f'matrixCompatibility_{correlation}')
    comp_file = find_compatibility_file(comp_dir, comp_k, threshold)
    if comp_file is None:
        print('Nenhum arquivo de compatibilidade encontrado em', comp_dir)
        sys.exit(1)

    matrixComp = pd.read_pickle(comp_file)

    # executar consulta SOLID ponderada
    try:
        from Submodules import Functions
    except Exception as e:
        print('Não foi possível importar Submodules.Functions:', e)
        sys.exit(1)

    if obj_idx not in train.index:
        print('obj_idx não encontrado no dataset, usando índice 0')
        obj_idx = train.index[0]

    objQuery = train.loc[obj_idx].copy()

    print(f'Executando consulta SOLID para obj {obj_idx} (k_query={k_query})')
    neighbors = Functions.Similarity_Queries_Weight(train.copy(), matrixComp, diameters, k_query, objQuery.copy())

    # salvar resultados
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    out_csv = os.path.join(results_dir, f'neighbors_{dataset_name}_obj{obj_idx}_k{str(k_query)}.csv')
    pd.DataFrame({'neighbor_id': neighbors}).to_csv(out_csv, index=False)

    print('Consulta finalizada. Resultados salvos em', out_csv)


if __name__ == '__main__':
    main(sys.argv)
