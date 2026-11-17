"""Build K fixed anchors for the multi-anchor UFF variant (2026-11 revision).

Anchors = spherical k-means centroids of the L2-normalised *pretrained* CLIP text features of the observed
(noisy) training captions. K = 1 reduces to the text-mean anchor of make_text_mean_ufr.py. No ground-truth
noise labels are used. Each anchor set is saved as a [K, 512] unit-norm tensor in fp16 (the UFR format);
main_uff.py routes every image to its most similar anchor when the loaded tensor has more than one row.

Usage: python revision/make_text_kmeans_ufr.py <dataset_root> <noise_ratio> <out_prefix> <K1,K2,...> [seed]
       -> <out_prefix>_K<k>.pt for every k
"""
import os
import sys

import clip
import torch
import torch.nn.functional as F


def text_features(root, noise, bs=2048):
    split = os.path.join(root, 'annotations', 'scan_split')
    name = 'train_caps.txt' if float(noise) == 0 else f'{noise}_noise_train_caps.txt'
    with open(os.path.join(split, name), encoding='utf-8') as f:
        caps = [line.strip() for line in f]
    model, _ = clip.clip.load('ViT-B/32', device='cuda', jit=False)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(caps), bs):
            tok = clip.clip.tokenize(caps[i:i + bs], truncate=True).cuda()
            feats.append(F.normalize(model.encode_text(tok).float(), dim=1))
    print(f'{len(caps)} captions from {name}')
    return torch.cat(feats)


def spherical_kmeans(x, k, seed, iters=50):
    g = torch.Generator(device='cpu').manual_seed(seed)
    c = x[torch.randperm(x.size(0), generator=g)[:k].to(x.device)].clone()
    for it in range(iters):
        assign = (x @ c.T).argmax(1)
        new = torch.zeros_like(c).index_add_(0, assign, x)
        empty = new.norm(dim=1) == 0
        if empty.any():  # re-seed empty clusters with random points
            new[empty] = x[torch.randint(x.size(0), (int(empty.sum()),), generator=g).to(x.device)]
        new = F.normalize(new, dim=1)
        shift = (1 - (new * c).sum(1)).max().item()
        c = new
        if shift < 1e-6:
            break
    sizes = torch.bincount((x @ c.T).argmax(1), minlength=k)
    obj = (x @ c.T).max(1).values.mean().item()
    return c, sizes, obj, it + 1


def main():
    root, noise, prefix, ks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    x = text_features(root, noise)
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    for k in [int(v) for v in ks.split(',')]:
        c, sizes, obj, n_it = spherical_kmeans(x, k, seed)
        out = f'{prefix}_K{k}.pt'
        torch.save(c.half().cpu(), out)
        print(f'K={k}: {n_it} iterations, mean cos to centroid {obj:.4f}, cluster size min/median/max '
              f'{sizes.min().item()}/{sizes.median().item()}/{sizes.max().item()} -> {out}')


if __name__ == '__main__':
    main()
