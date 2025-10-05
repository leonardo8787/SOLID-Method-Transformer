#!/usr/bin/env bash
# Script para criar/atualizar ambiente virtual e executar o pipeline completo
set -euo pipefail

VENV_DIR=".venv"
REQ_FILE="requirements.txt"

usage() {
  cat <<EOF
Uso: $0 <csv-file> <output-dir> [distance] [correlation] [threshold] [max_K] [obj_idx] [k_query] [comp_k]

Exemplo:
  $0 1-DS-MSTSpine.csv ./work_mstspine euclidean pearson 0.8 5 0 5 1

O script:
 - cria um venv em ${VENV_DIR} (se não existir)
 - ativa o venv e atualiza pip
 - instala/atualiza pacotes do ${REQ_FILE}
 - executa run_full_pipeline.py passando todos os argumentos
EOF
}

if [ "$#" -lt 2 ]; then
  echo "Argumentos insuficientes." >&2
  usage
  exit 1
fi

CSV="$1"
OUTDIR="$2"

echo "CSV: $CSV"
echo "Output dir: $OUTDIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "Criando virtualenv em $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtualenv já existe em $VENV_DIR. Usando ambiente existente (será atualizado)."
fi

echo "Ativando virtualenv..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Atualizando pip..."
pip install --upgrade pip

if [ -f "$REQ_FILE" ]; then
  echo "Instalando/atualizando dependências de $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "$REQ_FILE não encontrado. Instalando dependências básicas..."
  pip install pandas numpy scipy scikit-learn matplotlib
fi

echo "Executando run_full_pipeline.py..."
python3 run_full_pipeline.py "$@"

echo "Processo finalizado. Saída em ${OUTDIR}/results (se aplicável)."
