#!/usr/bin/env python3
"""
view_pkl.py

Ferramenta pequena para inspecionar arquivos .pkl (pickles) usados no projeto.

Funcionalidades:
- mostra tipo do objeto serializado e resumo (DataFrame: shape/dtypes/head)
- para dict/list mostra tamanho e primeiros elementos
- permite salvar DataFrame em CSV (--to-csv)
- permite inspecionar uma célula específica de um DataFrame (--cell 0,2)
- opção segura para disassembler do pickle sem desserializar (--disasm)

Uso:
  python view_pkl.py caminho/para/arquivo.pkl [--rows N] [--to-csv out.csv] [--cell R,C] [--disasm]

Segurança: carregar pickles executa código embutido — abra SOMENTE arquivos confiáveis.
Se não confiar no arquivo, use --disasm para ver os opcodes com pickletools.
"""
import argparse
import pandas as pd
import pickle
import pickletools
import os
import sys
import textwrap


def disasm(path, limit=200):
    print('--- pickletools disassembly (primeiras linhas) ---')
    try:
        out = pickletools.dis(open(path, 'rb').read(), annotate=False)
        # pickletools.dis returns None but prints to stdout; fallback: use command
    except Exception:
        # fallback: run module
        import subprocess
        subprocess.run([sys.executable, '-m', 'pickletools', path])


def main():
    parser = argparse.ArgumentParser(description='Visualizador de arquivos .pkl')
    parser.add_argument('pkl', help='caminho para o arquivo .pkl')
    parser.add_argument('--rows', type=int, default=10, help='quantas linhas mostrar para DataFrame (default 10)')
    parser.add_argument('--to-csv', help='salvar DataFrame em CSV (caminho de saída)')
    parser.add_argument('--cell', help='mostrar célula do DataFrame no formato R,C (ex: 0,2)')
    parser.add_argument('--disasm', action='store_true', help='disassembler do pickle (não desserializa)')

    args = parser.parse_args()
    path = args.pkl

    if not os.path.exists(path):
        print('Arquivo não encontrado:', path)
        sys.exit(1)

    if args.disasm:
        disasm(path)
        sys.exit(0)

    print('Atenção: carregar pickles pode executar código. Só abra arquivos confiáveis.')

    # Primeiro tente pandas (útil para DataFrame/Series)
    obj = None
    try:
        obj = pd.read_pickle(path)
        loaded_with = 'pandas.read_pickle'
    except Exception as e:
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            loaded_with = 'pickle.load'
        except Exception as e2:
            print('Falha ao carregar pickle:', e)
            print('Tentativa com pickle.load também falhou:', e2)
            print('\nUse --disasm para inspecionar sem desserializar.')
            sys.exit(1)

    print(f'Carregado com: {loaded_with}')
    print('Tipo:', type(obj))

    # DataFrame
    if isinstance(obj, pd.DataFrame):
        print('DataFrame shape:', obj.shape)
        print('\nDtypes:')
        print(obj.dtypes)
        print('\nHead:')
        print(obj.head(args.rows))
        if args.cell:
            try:
                r, c = [int(x) for x in args.cell.split(',')]
                val = obj.iloc[r, c]
                print(f'\nCélula ({r},{c}): type={type(val)}')
                print(val)
            except Exception as e:
                print('Erro ao acessar célula:', e)
        if args.to_csv:
            try:
                obj.to_csv(args.to_csv, index=False)
                print('DataFrame salvo em', args.to_csv)
            except Exception as e:
                print('Falha ao salvar CSV:', e)
        return

    # Series
    if isinstance(obj, pd.Series):
        print('Series length:', len(obj))
        print(obj.head(args.rows))
        return

    # Dict
    if isinstance(obj, dict):
        print('dict com', len(obj), 'chaves. chaves (amostra):')
        for i, k in enumerate(list(obj.keys())[:20]):
            print(i, k, type(obj[k]))
        return

    # List/Tuple
    if isinstance(obj, (list, tuple)):
        print('Lista/Tupla len =', len(obj))
        print('Primeiros itens:')
        for i, v in enumerate(obj[:min(len(obj), 20) ]):
            print(i, type(v), repr(v)[:200])
        return

    # Outros
    try:
        print('Repr (até 1000 chars):')
        print(textwrap.shorten(repr(obj), width=1000))
    except Exception as e:
        print('Não foi possível imprimir o objeto:', e)


if __name__ == '__main__':
    main()
