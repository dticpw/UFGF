"""Generate alternative noisy-caption files for the 2026-11 revision.

Modes (all keep the clean caption order; caption i is "noisy" iff it differs from train_caps.txt[i]):
  shuffle  - identical procedure to data.py (random subset, captions permuted among that subset);
             numpy seed fixed so the file is reproducible.
  semantic - for a random subset, caption i is replaced by a caption of a *different* image drawn
             uniformly from the top-k most similar captions under pretrained CLIP ViT-B/32 text features
             (semantically plausible mismatches; k=10 so it is not the single hardest negative).
  partial  - for a random subset, ~frac of the non-stopword tokens (at least one) are replaced by random
             frequent words from the training vocabulary (intra-caption corruption, e.g. wrong objects).

Usage:
  python make_noise.py shuffle  <clean_caps> <out> <ratio> [seed]
  python make_noise.py semantic <clean_caps> <out> <ratio> [seed] [num_anns=5] [k=10]
  python make_noise.py partial  <clean_caps> <out> <ratio> [seed] [frac=0.3]
"""
import collections
import random
import re
import sys

import numpy as np

STOP = set('a an the of on in at to with and or is are was were be been being it its this that these those '
           'for from by as into onto over under near while his her their there some two three one four '
           'he she they them who which has have had'.split())


def read(p):
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f]


def write(p, caps):
    with open(p, 'w', encoding='utf-8') as f:
        for c in caps:
            f.write(c + '\n')


def pick(n, ratio, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:int(ratio * n)], rng


def shuffle_mode(caps, ratio, seed):
    np.random.seed(seed)  # mirror data.py exactly
    idx = np.arange(len(caps)); np.random.shuffle(idx)
    k = int(ratio * len(caps))
    sc = np.array(caps)[idx[:k]]; np.random.shuffle(sc)
    out = np.array(caps); out[idx[:k]] = sc
    return list(out)


def semantic_mode(caps, ratio, seed, num_anns=5, k=10):
    import clip
    import torch
    sel, rng = pick(len(caps), ratio, seed)
    model, _ = clip.clip.load('ViT-B/32', device='cuda', jit=False)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(caps), 2048):
            f = model.encode_text(clip.clip.tokenize(caps[i:i + 2048], truncate=True).cuda()).float()
            feats.append(f / f.norm(dim=1, keepdim=True))
    F = torch.cat(feats)
    img = torch.arange(len(caps), device='cuda') // num_anns
    out = list(caps)
    sel_t = torch.as_tensor(sel, device='cuda')
    with torch.no_grad():
        for s in range(0, len(sel_t), 1024):
            q = sel_t[s:s + 1024]
            sim = F[q] @ F.T
            sim[img[q][:, None] == img[None, :]] = -2  # exclude captions of the same image
            top = sim.topk(k, dim=1).indices.cpu().numpy()
            for row, qi in enumerate(q.cpu().numpy()):
                out[qi] = caps[top[row, rng.randint(k)]]
    return out


def partial_mode(caps, ratio, seed, frac=0.3):
    sel, rng = pick(len(caps), ratio, seed)
    tok = lambda c: re.findall(r"[A-Za-z']+|[^A-Za-z'\s]", c)
    cnt = collections.Counter(w.lower() for c in caps for w in tok(c) if w.isalpha() and w.lower() not in STOP)
    vocab = [w for w, _ in cnt.most_common(5000)]
    out = list(caps)
    for i in sel:
        words = caps[i].split(' ')
        cand = [j for j, w in enumerate(words) if re.sub(r'\W', '', w).lower() not in STOP and re.sub(r'\W', '', w)]
        if not cand:
            continue
        m = max(1, int(round(frac * len(cand))))
        for j in rng.choice(cand, size=min(m, len(cand)), replace=False):
            core = re.sub(r'\W', '', words[j]).lower()
            new = core
            while new == core:
                new = vocab[rng.randint(len(vocab))]
            words[j] = new + (words[j][-1] if words[j] and not words[j][-1].isalnum() else '')
        out[i] = ' '.join(words)
    return out


def main():
    mode, src, dst, ratio = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 99
    caps = read(src)
    if mode == 'shuffle':
        out = shuffle_mode(caps, ratio, seed)
    elif mode == 'semantic':
        out = semantic_mode(caps, ratio, seed, *(int(x) for x in sys.argv[6:8]))
    elif mode == 'partial':
        out = partial_mode(caps, ratio, seed, *(float(x) for x in sys.argv[6:7]))
    else:
        raise SystemExit('unknown mode ' + mode)
    write(dst, out)
    diff = sum(a != b for a, b in zip(caps, out))
    print(f'{mode}: {len(out)} captions, {diff} differ ({100.0 * diff / len(out):.2f}%) -> {dst}')


if __name__ == '__main__':
    main()
