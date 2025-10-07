"""
clip_embed.py

Utilitário para extrair embeddings de imagens usando CLIP (OpenAI) e salvar
os embeddings em arquivos numpy/pickle na pasta atual.

Funções principais:
- build_model(device='cpu') -> (model, preprocess)
- embed_images(image_paths, model, preprocess, device, batch_size=32) -> np.ndarray
- save_embeddings(embeddings, paths, out_dir, base_name)

Uso básico:
from clip_embed import build_model, embed_images, save_embeddings
model, preprocess = build_model('cuda')
emb, paths = embed_images(list_of_image_paths, model, preprocess, 'cuda')
save_embeddings(emb, paths, './embeddings', 'mstspine')

Observações:
- Requer PyTorch e CLIP instalados (ver requirements.txt)
- Pode usar CPU (lento) ou GPU (recomendado quando disponível)
"""

import os
import math
import sys
import numpy as np
from PIL import Image
import torch


def build_model(device='cpu', model_name='ViT-B/32'):
    """Carrega o modelo CLIP e o transform preprocess.

    Retorna (model, preprocess). Use device 'cuda' ou 'cpu'.
    """
    try:
        import clip
    except Exception as e:
        raise RuntimeError('CLIP não está instalado. Execute: pip install git+https://github.com/openai/CLIP.git')

    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def _load_image(path, preprocess=None):
    try:
        img = Image.open(path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f'Falha ao abrir imagem {path}: {e}')

    if preprocess is not None:
        return preprocess(img)
    else:
        # sem preprocess, apenas converter para tensor
        import torchvision.transforms as T
        t = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
        return t(img)


def embed_images(image_paths, model, preprocess, device='cpu', batch_size=32):
    """Computa embeddings para uma lista de caminhos de imagens.

    Retorna (embeddings, valid_paths)
    - embeddings: np.ndarray shape (N, D)
    - valid_paths: lista de caminhos que foram carregados com sucesso
    """
    device = torch.device(device)
    all_embeddings = []
    valid_paths = []

    n = len(image_paths)
    if n == 0:
        return np.zeros((0, model.visual.output_dim)), []

    # processa em batches
    for i in range(0, n, batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(_load_image(p, preprocess))
                valid_paths.append(p)
            except Exception as e:
                print('Aviso: falhou ao carregar', p, '->', e)
        if len(imgs) == 0:
            continue
        x = torch.stack(imgs).to(device)

        with torch.no_grad():
            # normalizar embeddings
            emb = model.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy()
            all_embeddings.append(emb)

    if len(all_embeddings) == 0:
        return np.zeros((0, model.visual.output_dim)), []

    embeddings = np.vstack(all_embeddings)
    return embeddings, valid_paths


def save_embeddings(embeddings, paths, out_dir, base_name='emb'):
    """Salva embeddings e paths em out_dir.

    Cria arquivos:
    - out_dir/base_name_embeddings.npy
    - out_dir/base_name_paths.txt
    - out_dir/base_name_embeddings.pkl (pickle do array)
    """
    os.makedirs(out_dir, exist_ok=True)
    npy = os.path.join(out_dir, f'{base_name}_embeddings.npy')
    pth = os.path.join(out_dir, f'{base_name}_paths.txt')
    pkl = os.path.join(out_dir, f'{base_name}_embeddings.pkl')

    np.save(npy, embeddings)
    with open(pth, 'w') as f:
        for p in paths:
            f.write(p + '\n')

    # salvar pickle como fallback
    try:
        import pickle
        with open(pkl, 'wb') as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        print('Aviso: falha ao salvar pickle:', e)

    print('Embeddings salvos em:', npy)
    print('Paths salvos em:', pth)
    print('Embeddings (pickle) salvo em:', pkl)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gerar embeddings CLIP para imagens em uma pasta')
    parser.add_argument('images_dir', help='diretório contendo imagens (jpg/png)')
    parser.add_argument('--out', default='./embeddings', help='diretório de saída')
    parser.add_argument('--device', default='cpu', help='cuda ou cpu')
    parser.add_argument('--batch', type=int, default=32, help='tamanho do batch')
    parser.add_argument('--model', default='ViT-B/32', help='modelo CLIP')
    parser.add_argument('--ext', default='jpg,png', help='extensões separadas por vírgula')
    parser.add_argument('--name', default='images', help='base name para arquivos de saída')
    args = parser.parse_args()

    imgs = []
    exts = [e.strip().lower() for e in args.ext.split(',')]
    for root, _, files in os.walk(args.images_dir):
        for f in files:
            if any(f.lower().endswith('.' + e) for e in exts):
                imgs.append(os.path.join(root, f))

    if len(imgs) == 0:
        print('Nenhuma imagem encontrada em', args.images_dir)
        sys.exit(1)

    model, preprocess = build_model(args.device, args.model)
    emb, paths = embed_images(imgs, model, preprocess, args.device, args.batch)
    save_embeddings(emb, paths, args.out, args.name)