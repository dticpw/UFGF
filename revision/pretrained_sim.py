"""Similarity of observed training pairs under the pretrained CLIP (before any fine-tuning).

Inference only; gives the "before training" panel of the similarity-distribution figure.
For every caption of the noisy training file: cosine(img, observed caption) with pretrained ViT-B/32,
ground-truth noise flag (line differs from train_caps.txt), AUROC of the cosine for clean vs noisy,
a histogram of the cosine for clean and noisy pairs (bins -0.1..0.6, step 0.01), and the
precision/recall of the in-batch top-match clean mask on random batches of size --bs.

Usage: python revision/pretrained_sim.py <dataset_root> <noise_ratio> <out_dir> [--bs 256] [--seed 99]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clip  # noqa: E402
from diag import _auroc, _read_lines  # noqa: E402


class Imgs(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.pre = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.pre(Image.open(self.paths[i]).convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('noise', type=float)
    ap.add_argument('out')
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--seed', type=int, default=99)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    split = os.path.join(a.root, 'annotations', 'scan_split')
    clean = _read_lines(os.path.join(split, 'train_caps.txt'))
    noisy = _read_lines(os.path.join(split, f'{a.noise}_noise_train_caps.txt'))
    ids = [int(x) for x in _read_lines(os.path.join(split, 'train_ids.txt')) for _ in range(5)]  # 5 captions/image, as in data.py
    names = _read_lines(os.path.join(split, 'image_name.txt'))
    assert len(clean) == len(noisy) == len(ids), (len(clean), len(noisy), len(ids))
    gt_noisy = np.array([c != n for c, n in zip(clean, noisy)])
    print(f'{len(ids)} captions, {gt_noisy.sum()} noisy ({100 * gt_noisy.mean():.2f}%)', flush=True)

    dev = 'cuda'
    model, pre = clip.clip.load('ViT-B/32', device=dev, jit=False)
    model.eval()
    uniq = sorted(set(ids))
    pos = {u: k for k, u in enumerate(uniq)}
    paths = [os.path.join(a.root, 'images', names[u]) for u in uniq]
    feats = []
    with torch.no_grad():
        for x in DataLoader(Imgs(paths, pre), batch_size=256, num_workers=8):
            f = model.encode_image(x.to(dev)).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        img = torch.cat(feats)
        txt = []
        for s in range(0, len(noisy), 1024):
            tok = clip.clip.tokenize(noisy[s:s + 1024], truncate=True).to(dev)
            txt.append(torch.nn.functional.normalize(model.encode_text(tok).float(), dim=-1).cpu())
        txt = torch.cat(txt)
    img_c = img[[pos[i] for i in ids]]
    sim = (img_c * txt).sum(-1).numpy()

    # in-batch top-match clean mask (same rule as training) on random batches
    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(len(ids))
    clean_mask = np.zeros(len(ids), dtype=bool)
    for s in range(0, len(perm) - a.bs + 1, a.bs):  # drop_last like the training loader
        b = perm[s:s + a.bs]
        m = img_c[b] @ txt[b].T
        clean_mask[b] = (m.argmax(1).numpy() == np.arange(len(b)))
    used = np.zeros(len(ids), dtype=bool)
    used[perm[:(len(perm) // a.bs) * a.bs]] = True
    g, c = gt_noisy[used], clean_mask[used]
    flagged = ~c
    prec = float((flagged & g).sum() / max(flagged.sum(), 1))
    rec = float((flagged & g).sum() / max(g.sum(), 1))

    bins = np.round(np.arange(-0.1, 0.6 + 1e-9, 0.01), 4)
    hc, _ = np.histogram(sim[~gt_noisy], bins=bins)
    hn, _ = np.histogram(sim[gt_noisy], bins=bins)
    res = {
        'n': int(len(ids)), 'n_noisy': int(gt_noisy.sum()),
        'auroc_ori_sim': float(_auroc(sim, ~gt_noisy)),
        'mean_clean': float(sim[~gt_noisy].mean()), 'std_clean': float(sim[~gt_noisy].std()),
        'mean_noisy': float(sim[gt_noisy].mean()), 'std_noisy': float(sim[gt_noisy].std()),
        'mask_bs': a.bs, 'mask_seed': a.seed,
        'mask_precision_noisy': prec, 'mask_recall_noisy': rec,
        'mask_f1_noisy': 2 * prec * rec / max(prec + rec, 1e-12),
        'clean_flagged_rate': float((flagged & ~g).sum() / max((~g).sum(), 1)),
        'hist': {'bins': bins.tolist(), 'clean': hc.tolist(), 'noisy': hn.tolist(), 'range': [-0.1, 0.6]},
    }
    np.save(os.path.join(a.out, 'ori_sim.npy'), sim.astype(np.float32))
    np.save(os.path.join(a.out, 'gt_noisy.npy'), gt_noisy)
    with open(os.path.join(a.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'hist'}, indent=1))


if __name__ == '__main__':
    main()
