# UFGF: Universal Feature Generation and Filling for Robust Cross-Modal Matching

Code for image–text retrieval under noisy correspondence with a CLIP ViT-B/32 backbone.
Suspected noisy pairs are not dropped or re-weighted; their text features are replaced ("filled")
by a single shared anchor, the Universal Feature Representation (UFR).

Training has two independent runs:

1. **UFG** (`revision/main_ufg.py`): fine-tunes CLIP from the pretrained weights, learns a shared projection,
   and exports the batch-mean projected text feature as the UFR (`ufr/universal_pseudo_text3.pt`,
   taken at the first batch of epoch 4).
2. **UFF** (`revision/main_uff.py`): restarts from the pretrained CLIP weights with a new optimizer, keeps the UFR
   fixed, and replaces the text feature of every pair that is not detected as clean. The final retrieval model
   is the UFF checkpoint selected on the validation set.

## Environment

Tested with Python 3.10, PyTorch 2.2.2 (CUDA 12.1 wheels) and one RTX 4090 D (24 GB; one run uses about 13 GB).

```bash
pip install -r requirements-t200.txt          # includes OpenAI CLIP at commit d05afc4
CLIP_DIR=$(python -c "import clip, os; print(os.path.dirname(clip.__file__))")
cp model.py criterions.py "$CLIP_DIR"/          # the training code imports clip.criterions and needs
                                                # the modified CLIP forward that also returns features
```

`model.py` is a modified copy of OpenAI CLIP's `clip/model.py` (MIT License, see `LICENSE`).
The pretrained `ViT-B/32` weights are downloaded by `clip.load` on first use.

## Data

We use the SCAN splits of Flickr30K and MS-COCO and the CC120K subset of Conceptual Captions, in the format of
[NPC](https://github.com/ZhangXu0963/NPC) (annotations and images are available from the links in its README).
`--dataset_root` points to one dataset directory:

```
<dataset_root>/
├── images/                         # Flickr30K: original file names; CC120K: <id>.jpg
│                                   # MS-COCO: COCO_2014_<12-digit id>.jpg (rename train2014/val2014 files)
└── annotations/scan_split/
    ├── train_ids.txt  train_caps.txt
    ├── dev_ids.txt    dev_caps.txt
    ├── test_ids.txt   test_caps.txt
    ├── image_name.txt                # Flickr30K only
    └── <ratio>_noise_train_caps.txt  # e.g. 0.4_noise_train_caps.txt
```

Flickr30K and MS-COCO use 5 captions per image (`--num_anns 5`); CC120K uses 1 (`--num_anns 1`) and its
noise is real (`--noise_ratio 0`, the observed captions are used as is).

### Synthetic noise

A noisy-caption file keeps the clean caption order; caption *i* is noisy iff it differs from `train_caps.txt[i]`.
If `<ratio>_noise_train_caps.txt` is missing, `data.py` creates it by shuffling the captions of a random subset.
`revision/make_noise.py` reproduces this exactly (with seed 99 it regenerates our 40% Flickr30K file byte for byte)
and also provides the two extra noise types used in the revision:

```bash
python revision/make_noise.py shuffle  <root>/annotations/scan_split/train_caps.txt out.txt 0.4 99
python revision/make_noise.py semantic <train_caps.txt> out.txt 0.4 99   # caption of another image, random among CLIP top-10
python revision/make_noise.py partial  <train_caps.txt> out.txt 0.4 99   # ~30% of content words replaced
```

## Training and evaluation

Run from the repository root. Outputs go to `$UFGF_RUNS/<run_name>/` (default `/root/autodl-tmp/runs`; set
`UFGF_RUNS` and `UFGF_PY` for your machine).

```bash
# UFG -> UFR -> UFF, then test the validation-selected UFG and UFF checkpoints
bash revision/run_two_stage.sh f30k /data/Flickr30K 0.4 99 f30k_n04_s99
# same-backbone baseline: plain CLIP fine-tuning on the same noisy data (no replacement)
bash revision/run_vanilla.sh f30k /data/Flickr30K 0.4 99 f30k_n04_vanilla_s99 <any_ufr.pt>
# UFF only, with a given fixed anchor
bash revision/run_uff_only.sh f30k /data/Flickr30K 0.4 99 f30k_n04_uffonly_s99 <anchor.pt>
# anchor = mean pretrained CLIP text feature of the training captions (no UFG)
python revision/make_text_mean_ufr.py /data/Flickr30K 0.4 textmean.pt
# K fixed anchors = spherical k-means centroids of the same text features (k-means seed 0);
# when the anchor file has K > 1 rows, main_uff.py routes each image to its most similar anchor
python revision/make_text_kmeans_ufr.py /data/Flickr30K 0.4 kmeans 16,128,1024 0
```

Default recipe (the settings of the paper, used for all runs): batch 256, 5 epochs per stage, lr 1e-7 for CLIP and
1e-5 for the projection, weight decay 0.2, warm-up 500 steps, λ1 = 2, λ2 = 0.8, σ = 0.2. These are the defaults of
`params.py` and `run_two_stage.sh`. Optional environment variables of `run_two_stage.sh`:
`UFGF_BS`, `UFGF_SIGMA`, `UFGF_NANNS`, `UFGF_LAMBDA1`, `UFGF_LAMBDA2`, `UFGF_PROJ_LR`, `UFGF_UFF_RULE=three`, and
`UFGF_ORACLE=1` (UFF only: use the ground-truth noise labels of the synthetic noise file instead of the predicted
clean mask). `revision/run_drop.sh` runs a drop baseline that removes the flagged pairs from the batch instead of
filling them (same clean mask and schedule as UFF).

Seeds used in the paper: training seed 99 for all single-run results, and seeds 1, 2 and 99 for the three-seed
results on Flickr30K with 40% shuffled and semantic noise; noise files generated by `make_noise.py` (80% shuffled,
semantic and intra-caption noise) use seed 99.

Test only: `python revision/main_ufg.py --eval --resume <ckpt> --dataset f30k --dataset_root <root> --num_anns 5`
(MS-COCO additionally reports the five-fold 1K protocol).

### Diagnostics

Each training run writes `diag/` next to its log: per-epoch clean-mask precision/recall/F1 and AUROC against the
known synthetic noise, replacement rates among noisy and clean pairs, per-sample similarities
(`*_samples.npz`), per-step losses and learning rates (`*_steps.jsonl`), and validation-set embedding statistics
(effective rank, variance, mean pairwise cosine) in `embed_stats.jsonl`.

`revision/pretrained_sim.py <dataset_root> <noise_ratio> <out_dir>` computes the same similarity statistics for
pretrained CLIP before any training (cosine of every observed pair, AUROC for clean versus noisy pairs, and the
precision/recall of the in-batch clean mask), which gives the "before training" panel of the similarity figure.

## License

MIT, see `LICENSE`.
