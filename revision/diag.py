"""Revision diagnostics for UFG/UFF runs (added 2026-11 for the NEUNET-D-26-06233 revision).

Records, without changing the training computation:
  * per-sample, per-epoch: cosine(img, observed text), cosine(img, UFR), clean_mask, replace_mask
    (indexed by caption index, so they can be joined with the ground-truth noise flag);
  * per-step: loss components, learning rates, mask counts;
  * per-epoch: dev-set embedding statistics (feature variance, effective rank, mean pairwise cosine).

Ground-truth noise flag: caption i is noisy iff the line in <ratio>_noise_train_caps.txt differs from
train_caps.txt (the noise files were produced by shuffling a subset of captions).
"""
import json
import os

import numpy as np
import torch


def _read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]


class Diag:
    def __init__(self, run_dir, dataset_root, noise_ratio, stage, logger):
        self.dir = os.path.join(run_dir, 'diag')
        os.makedirs(self.dir, exist_ok=True)
        self.stage = stage
        self.logger = logger
        split = os.path.join(dataset_root, 'annotations', 'scan_split')
        clean = _read_lines(os.path.join(split, 'train_caps.txt'))
        if noise_ratio == 0:
            noisy_caps = clean
        else:
            noisy_caps = _read_lines(os.path.join(split, f'{noise_ratio}_noise_train_caps.txt'))
        assert len(clean) == len(noisy_caps), 'caption files differ in length'
        self.gt_noisy = np.array([a != b for a, b in zip(clean, noisy_caps)], dtype=bool)
        self.n = len(clean)
        np.save(os.path.join(self.dir, 'gt_noisy.npy'), self.gt_noisy)
        logger.info('[diag] %s: %d captions, %d differ from clean (%.2f%%)',
                    stage, self.n, int(self.gt_noisy.sum()), 100.0 * self.gt_noisy.mean())
        self.steps = []
        self._new_epoch()

    def _new_epoch(self):
        self.ori_sim = np.full(self.n, np.nan, dtype=np.float32)
        self.ufr_sim = np.full(self.n, np.nan, dtype=np.float32)
        self.clean = np.zeros(self.n, dtype=bool)
        self.replace = np.zeros(self.n, dtype=bool)
        self.seen = np.zeros(self.n, dtype=bool)

    @torch.no_grad()
    def record_batch(self, epoch, step, cap_idx, ori_sim, ufr_sim, clean_mask, replace_mask, losses, lrs):
        i = cap_idx.cpu().numpy()
        self.ori_sim[i] = ori_sim.float().cpu().numpy()
        self.ufr_sim[i] = ufr_sim.float().cpu().numpy()
        c = clean_mask.cpu().numpy()
        r = replace_mask.cpu().numpy()
        self.clean[i] = c
        self.replace[i] = r
        self.seen[i] = True
        g = self.gt_noisy[i]
        rec = {'epoch': epoch, 'step': step, 'B': int(len(i)),
               'n_clean': int(c.sum()), 'n_replace': int(r.sum()),
               'n_kept_flagged': int((~c & ~r).sum()),
               'n_gt_noisy': int(g.sum()),
               'clean_and_gt_noisy': int((c & g).sum()),
               'replace_and_gt_noisy': int((r & g).sum()),
               'kept_flagged_and_gt_noisy': int((~c & ~r & g).sum()),
               'lr': [float(x) for x in lrs]}
        rec.update({k: float(v) for k, v in losses.items()})
        self.steps.append(rec)

    def end_epoch(self, epoch):
        s = self.seen
        np.savez_compressed(os.path.join(self.dir, f'{self.stage}_epoch{epoch + 1}_samples.npz'),
                            ori_sim=self.ori_sim, ufr_sim=self.ufr_sim, clean=self.clean,
                            replace=self.replace, seen=s)
        g, c, r = self.gt_noisy[s], self.clean[s], self.replace[s]
        # "clean_mask" as a detector of clean pairs
        tp = int((c & ~g).sum()); fp = int((c & g).sum()); fn = int((~c & ~g).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        summary = {
            'stage': self.stage, 'epoch': epoch + 1, 'seen': int(s.sum()),
            'clean_precision': prec, 'clean_recall': rec,
            'clean_f1': 2 * prec * rec / max(prec + rec, 1e-12),
            'auroc_ori_sim': _auroc(self.ori_sim[s], ~g),
            'replace_rate_among_gt_noisy': float(r[g].mean()) if g.any() else float('nan'),
            'replace_rate_among_gt_clean': float(r[~g].mean()) if (~g).any() else float('nan'),
            'kept_flagged_frac': float((~c & ~r).mean()),
            'kept_flagged_gt_noisy_frac': float((~c & ~r & g).mean()),
        }
        with open(os.path.join(self.dir, 'epoch_summary.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary) + '\n')
        with open(os.path.join(self.dir, f'{self.stage}_steps.jsonl'), 'a', encoding='utf-8') as f:
            for rec_ in self.steps:
                f.write(json.dumps(rec_) + '\n')
        self.steps = []
        self.logger.info('[diag] %s', json.dumps(summary))
        self._new_epoch()

    @torch.no_grad()
    def embed_stats(self, model, dataloader, epoch):
        model.eval()
        feats_i, feats_t = [], []
        for images, texts, _ in dataloader:
            fi = model.encode_image(images.cuda()).float()
            ft = model.encode_text(texts.cuda()).float()
            feats_i.append(fi / fi.norm(dim=1, keepdim=True))
            feats_t.append(ft / ft.norm(dim=1, keepdim=True))
        num_anns = dataloader.dataset.num_anns
        fi = torch.cat(feats_i)[::num_anns]
        ft = torch.cat(feats_t)
        out = {'stage': self.stage, 'epoch': epoch + 1}
        for name, x in (('img', fi), ('txt', ft)):
            xc = x - x.mean(0, keepdim=True)
            sv = torch.linalg.svdvals(xc)
            p = sv ** 2 / (sv ** 2).sum()
            out[f'{name}_effective_rank'] = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
            out[f'{name}_total_variance'] = float(xc.pow(2).sum(1).mean())
            sims = x @ x.T
            m = x.shape[0]
            out[f'{name}_mean_pairwise_cos'] = float((sims.sum() - m) / (m * (m - 1)))
        with open(os.path.join(self.dir, 'embed_stats.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(out) + '\n')
        self.logger.info('[diag] %s', json.dumps(out))
        model.train()


def _auroc(score, positive):
    """AUROC of `score` for separating positive (clean) from negative (noisy); rank-based."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
