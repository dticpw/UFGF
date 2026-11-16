"""Build an alternative fixed anchor for the UFF ablation (2026-11 revision).

Anchor = mean of L2-normalised *pretrained* CLIP text features over the observed (noisy) training
captions, re-normalised to unit length, saved as a [1, 512] tensor in the same dtype as the UFG UFR.
No UFG training is involved, so running UFF with this file isolates whether the UFG-produced UFR matters.

Usage: python revision/make_text_mean_ufr.py <dataset_root> <noise_ratio> <out.pt> [batch]
"""
import os
import sys

import clip
import torch


def main():
    root, noise, out = sys.argv[1], sys.argv[2], sys.argv[3]
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
    split = os.path.join(root, 'annotations', 'scan_split')
    name = 'train_caps.txt' if float(noise) == 0 else f'{noise}_noise_train_caps.txt'
    with open(os.path.join(split, name), encoding='utf-8') as f:
        caps = [line.strip() for line in f]
    model, _ = clip.clip.load('ViT-B/32', device='cuda', jit=False)
    model.eval()
    total = None
    with torch.no_grad():
        for i in range(0, len(caps), bs):
            tok = clip.clip.tokenize(caps[i:i + bs], truncate=True).cuda()
            feat = model.encode_text(tok).float()
            feat = feat / feat.norm(dim=1, keepdim=True)
            s = feat.sum(0)
            total = s if total is None else total + s
    mean = total / len(caps)
    print(f'{len(caps)} captions from {name}; norm of raw mean = {mean.norm().item():.4f}')
    ufr = (mean / mean.norm()).unsqueeze(0).half().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(ufr, out)
    print('saved', out, tuple(ufr.shape), ufr.dtype)


if __name__ == '__main__':
    main()
