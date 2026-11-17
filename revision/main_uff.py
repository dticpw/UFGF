import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import clip
from clip.criterions import TotalLoss
from torch import optim
from util import set_seed_logger, get_logger, feature_plot, visualize_and_save
from params import parse_args
from scheduler import cosine_lr
from eval import evaluate, evaluate_1K
from data import prepare_dataloader
import copy
from torch import autograd
from multihead import MultiHeadExternalAttention
from diag import Diag  # revision diagnostics
from einops import rearrange
from collections import deque



global logger

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def set_requires_grad(net: nn.Module, mode=True):
    for p in net.parameters():
        p.requires_grad_(mode)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class SlidingWindow:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        
    def add(self, value):
        self.values.append(value)
    
    def get_n_mean(self, last_n=100):
        if not self.values:
            return 0
        if last_n is None or last_n >= len(self.values):
            return sum(self.values) / len(self.values)
        return sum(list(self.values)[-last_n:]) / last_n
    
    def clear(self):
        self.values.clear()


def main():
    global logger
    args = parse_args()

    seed = set_seed_logger(args)
    # dir_path = os.path.join(args.checkpoint_path, args.experiments)
    dir_path = os.environ.get('UFGF_RUN_DIR', 'runs/default')  # revision: output dir from env
    os.makedirs(dir_path, exist_ok=True)
    logger = get_logger(os.path.join(dir_path, "log.txt"))

    ### 实在是太长了这些，而且在sh里也直接能看到
    # logger.info("Effective parameters:")
    # for key in sorted(args.__dict__):
    #     logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model_clip, preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training
    # siamese_model_clip, siamese_preprocess = clip.clip.load(args.vision_model, device=device, jit=False) #Must set jit=False for training
    clip.clip.load(args.vision_model, device=device, jit=False)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model_clip
        # siamese_model = siamese_model_clip
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model from {}".format(args.resume))
    else:
        model = model_clip
        # siamese_model = siamese_model_clip
        logger.info("Model Initialized!")

    model = model.cuda()
    # siamese_model = siamese_model.cuda()
    fea_dim = 512
    crossAttn = MultiHeadExternalAttention(fea_dim,fea_dim,8).cuda().half()

    if args.eval:
        train_dataloader = None
        train_length = 0
        args.epochs = 0
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'test')
        test_dataloader, test_length = dataloader['test']
        if args.dataset == 'coco':
            eval_Rank = evaluate(args, model, test_dataloader, logger)
            eval_Rank_1K = evaluate_1K(args, model, test_dataloader, logger) # Only for MSCOCO 1K testing
        else:
            eval_Rank = evaluate(args, model, test_dataloader, logger)
    else:
        dataloader = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'train')
        train_dataloader, train_length = dataloader['train'] # 里面有一堆东西，cap_token,captions,image_name,images_id,mbank_i,mbank_t都有
        dataloader_dev = prepare_dataloader(args, args.dataset_root, preprocess, logger, 'dev')
        dev_dataloader, dev_length = dataloader_dev['dev']

    loss = TotalLoss().cuda()
    total_steps = train_length * args.epochs
    ufr_path = os.environ.get('UFGF_UFR', "/I2T/NPC-master/logs/f30k0_4/uni_txt/universal_pseudo_text3.pt")  # revision: UFR path from env
    universal_pseudo_text = torch.load(ufr_path).cuda()
    logger.info("Loaded fixed UFR from %s, shape %s", ufr_path, tuple(universal_pseudo_text.shape))
    diag = None
    if not args.eval:
        diag = Diag(dir_path, args.dataset_root, args.noise_ratio, 'uff', logger)



    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 
    # revision: the archived 4-arg cosine_lr(optimizer, base_lr, warmup, steps) no longer exists in scheduler.py.
    # With this single param group, the 3-arg version scaling initial_lr gives the identical lr schedule.
    for g in optimizer.param_groups:
        g["initial_lr"] = args.lr
    scheduler = cosine_lr(optimizer, args.warmup, total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    clip_sim_threshold = 0.2
    
    for epoch in range(args.epochs):

        # NPC training
        # train(args, model, siamese_model, crossAttn, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, siamese_scheduler, optimizer, siamese_optimizer, loss)
        train(args, model, universal_pseudo_text, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, optimizer, loss, diag)
        diag.end_epoch(epoch)
        
        save_path = os.path.join(dir_path, f"epoch{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "name": args.name,
                "state_dict": model.state_dict(),
                # revision: optimizer state not saved (training resume is never used; saves ~2x disk)
            },
            save_path,
        )
        logger.info("Saved checkpoint {} (epoch {})".format(save_path, epoch + 1))

        ## Run on val dataset for selecting best model.
        logger.info("Eval on val dataset")
        eval_Rank = evaluate(args, model, dev_dataloader, logger)
        diag.embed_stats(model, dev_dataloader, epoch)

        if best_score <= eval_Rank:
            best_score = eval_Rank
            best_output_model_file = save_path
        logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

# def train(args, model, siamese_model, crossAttn, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, siamese_scheduler, optimizer, siamese_optimizer, loss):
def train(args, model, universal_pseudo_text, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, optimizer, loss, diag=None):

    model.train()
    # siamese_model.train()

    # 初始化滑动窗口 # 每个epoch都会重新构建
    sample_score_max_window = SlidingWindow(args.display)
    sample_score_mean_window = SlidingWindow(args.display)
    ori_score_mean_window = SlidingWindow(args.display)
    ori_score_max_window = SlidingWindow(args.display)
    
    noisy_count = 0
    replaced_count = 0
    clip_sim_threshold = float(os.environ.get("UFGF_SIGMA", 0.2))  # revision: sigma override for the gated rule (default 0.2)
    replaced = 0
    route_hist = torch.zeros(universal_pseudo_text.size(0), dtype=torch.long)  # revision: anchor usage (multi-anchor only)
    logit_scale = torch.tensor(100.0, device='cuda')
    fea_dim = 512

    stored_I = [] # 可视化相关参数
    stored_T = []
    stored_PI = []
    stored_PT = []
    store_count = 0
    visual_count = 10

    loss_I = 0.0
    loss_T = 0.0


    for idx, batch in enumerate(train_dataloader) : # 因为是直接导入的没有随机数所以每次从同一个batch开始训练

        step = train_length * epoch + idx
        scheduler(step)
        # siamese_scheduler(step) # step? # cosine_lr(step)

        images, texts, cap_idx, img_ids = batch # (128,3,224,224),(128,77) # get_item决定的哇
        images = images.cuda()
        texts = texts.cuda()
        # mbankI_imgs = mbankI_imgs.cuda()
        # mbankI_txts = mbankI_txts.cuda()
        # mbankT_txts = mbankT_txts.cuda()
        # mbankT_imgs = mbankT_imgs.cuda()

        #### loss
        _, _, ImgFea_b_grad, TxtFea_b_grad = model(images, texts) # (bs,bs) #### 0208 在这里！！！
        # with torch.no_grad(): # 训练到下一轮的参数与本轮的在同一个该batch上对比得到 confidence weight w # 使用loss之间的对比
        #     _, _, ImgFea_MBI_s_nograd, TxtFea_MBI_s_nograd = model(mbankI_imgs, mbankI_txts) 
        
        with torch.no_grad(): ### 计算置信度的本质是"评估"而非"学习",梯度不从这里流向attnT生成模块
            ori_sim = F.cosine_similarity(ImgFea_b_grad, TxtFea_b_grad) 
            ori_mat = ImgFea_b_grad @ TxtFea_b_grad.T # 在这里搞一下双向
            clean_mask = torch.arange(ImgFea_b_grad.size(0)).cuda() == torch.max(ori_mat, dim=-1)[1] # 匹配的话便是clean
            if os.environ.get("UFGF_ORACLE") == "1":  # revision: ground-truth clean mask (upper bound for detection)
                clean_mask = torch.from_numpy(~diag.gt_noisy[cap_idx.cpu().numpy()]).to(clean_mask.device)
            noisy_count = noisy_count + torch.sum(~clean_mask).item() 

        ## 全部替换不太好，还得检验一下
        with torch.no_grad(): ### 计算置信度的本质是"评估"而非"学习",梯度不从这里流向attnT生成模块
            anchor_b = universal_pseudo_text  # [1, d]: one shared anchor (default, unchanged behaviour)
            if universal_pseudo_text.size(0) > 1:  # revision: K fixed anchors, each image routed to its most similar anchor
                route = (F.normalize(ImgFea_b_grad.float(), dim=-1)
                         @ F.normalize(universal_pseudo_text.float(), dim=-1).T).argmax(dim=1)
                anchor_b = universal_pseudo_text[route]  # [B, d]
                route_hist += torch.bincount(route[~clean_mask], minlength=universal_pseudo_text.size(0)).cpu()
            clip_sim = F.cosine_similarity(ImgFea_b_grad, anchor_b)
        #### 量化生成样本的品质 # 更新滑动窗口
        sample_score_max_window.add(torch.max(clip_sim).item())
        sample_score_mean_window.add(torch.mean(clip_sim).item())
        ori_score_mean_window.add(torch.mean(ori_sim).item())
        ori_score_max_window.add(torch.max(ori_sim).item())
        
        # 4.定位替换
        replace_mask = (~clean_mask) # 这clip_sim_max是否有问题#没有问题，显示的是最大值的均值，还是有比2.0大的
        if os.environ.get("UFGF_UFF_RULE") == "three":  # revision ablation: UFG-style rule (non-clean, > sigma, better than own text)
            replace_mask = (clip_sim > ori_sim) & (clip_sim > clip_sim_threshold) & (~clean_mask)
        # replace_mask = ((clip_sim > ori_sim)) & (~clean_mask)
        texts_new_fea = torch.where(replace_mask.unsqueeze(1), anchor_b.clone(), TxtFea_b_grad) # 如果~clean_mask(是噪声)，但没找到合适的伪样本(clip_sim > ori_sim)，那还是不替换
        vanilla = os.environ.get("UFGF_MODE") == "vanilla"  # revision: plain CLIP fine-tuning baseline
        if vanilla:  # no replacement; every pair keeps its observed (possibly noisy) caption
            replace_mask = torch.zeros_like(clean_mask)
            texts_new_fea = TxtFea_b_grad
        # 对替换的样本计数
        replaced = replaced + torch.sum(replace_mask).item() ## 能给出来的 55
        replaced_count = replaced_count + torch.sum(replace_mask).item()


        #### loss 
        batch_i2t_sim_new, batch_t2i_sim_new, _, _ = model(ImgFea_b_grad, texts_new_fea) # (bs,bs) # 确实啊，前面的用不到
        batch_i2t_ori_loss, batch_t2i_ori_loss = loss(batch_i2t_sim_new, batch_t2i_sim_new, 'batch')
        loss_mask = torch.ones_like(clean_mask) if vanilla else (replace_mask | clean_mask)
        batch_loss = torch.mean(loss_mask * (batch_i2t_ori_loss + batch_t2i_ori_loss)/2)            
        
        # batch_loss = torch.mean((batch_i2t_ori_loss + batch_t2i_ori_loss)/2)
        final_loss = batch_loss #+ loss_I + loss_T# + feaMatchingLoss
        if diag is not None:
            diag.record_batch(epoch, step, cap_idx, ori_sim, clip_sim, clean_mask, replace_mask,
                              {'batch_loss': batch_loss.item(),
                               'ce_mean': ((batch_i2t_ori_loss + batch_t2i_ori_loss) / 2).item()},
                              [g['lr'] for g in optimizer.param_groups])

        # # # ##可视化
        # # # # 确保你的区分化 和 高质量
        # if (idx % (800//visual_count) == 0) and store_count < visual_count:
        #     stored_I.append(ImgFea_b_grad[0])
        #     stored_T.append((TxtFea_b_grad)[0])
        #     stored_PI.append((universal_pseudo_text))
        #     stored_PT.append((universal_pseudo_text))
        #     store_count += 1
        # if (store_count == visual_count):
        #     stacked_I = torch.stack(stored_I, dim=0)
        #     stacked_T = torch.stack(stored_T, dim=0)
        #     stacked_PI = torch.stack(stored_PI, dim=0)
        #     stacked_PT = torch.stack(stored_PT, dim=0)
        #     tensor_IPT = torch.cat([stacked_I,stacked_T,stacked_PI,stacked_PT],dim=0)
        #     # tensor_TPTc = torch.cat([stacked_T,stacked_PTc],dim=0)

        #     viusal_I = tensor_IPT @ tensor_IPT.t()
        #     # visual_tensor = torch.cat([viusal_I,viusal_T],dim=0)
        #     # ranks_vis = np.clip(ranks_vis, 0, 11)
        #     visualize_and_save(viusal_I, '/I2T/NPC-master/logs/f30k0_4/plots0506/tmp/img_txt0520/')
        #     # visualize_and_save(viusal_T, '/I2T/NPC-master/logs/f30k0_4/plots0506/tmp/TPTc/')
        #     logger.info("visualize completed!")
        #     store_count += 1

        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2) ## 还没看
        optimizer.step()

        # if epoch == 2:  # 注意：第3个epoch的编号是2
        #     # attnT is shape (bs, dim) — 取平均生成通用伪文本
        #     universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
        #     torch.save(universal_pseudo_text, "/I2T/NPC-master/logs/f30k0_4/uni_txt/universal_pseudo_text.pt")
        #     print("通用伪文本向量已保存为 '/I2T/NPC-master/logs/f30k0_4/uni_txt/universal_pseudo_text.pt'")
        #     sys.exit(0)
        if (idx % (args.display) == 0 and idx!=0):
            # logger.info("Epoch: %d/%d, step:%d/%d, lr: %.2e/%.2e, batch_loss: %.3f, loss_I: %.3f, loss_T: %.3f, sim_tou: %.3f, smean/max/omean/max: %.3f/%.3f/%.3f/%.3f, R_C/N_C: %d/%d", 
            #             epoch + 1, args.epochs, idx, len(train_dataloader), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            #             batch_loss, loss_I, loss_T, clip_sim_threshold, sample_score_mean_window.get_n_mean(), sample_score_max_window.get_n_mean(),
            #             ori_score_mean_window.get_n_mean(), ori_score_max_window.get_n_mean(), replaced_count, noisy_count)
            logger.info("Epoch: %d/%d, step:%d/%d, batch_loss: %.3f, loss_I: %.3f, loss_T: %.3f, sim_tou: %.3f, smean/max/omean/max: %.3f/%.3f/%.3f/%.3f, R_C/N_C: %d/%d", 
                        epoch + 1, args.epochs, idx, len(train_dataloader),
                        batch_loss, loss_I, loss_T, clip_sim_threshold, sample_score_mean_window.get_n_mean(), sample_score_max_window.get_n_mean(),
                        ori_score_mean_window.get_n_mean(), ori_score_max_window.get_n_mean(), replaced_count, noisy_count)
            
            noisy_count = 0
            replaced_count = 0

    if universal_pseudo_text.size(0) > 1:  # revision: how the flagged pairs were spread over the anchors
        h = route_hist.float()
        p_ = h / h.sum().clamp(min=1)
        logger.info("[anchors] epoch %d: K=%d, used=%d, largest share=%.3f, effective number=%.1f",
                    epoch + 1, h.numel(), int((h > 0).sum()), p_.max().item(),
                    torch.exp(-(p_[p_ > 0] * p_[p_ > 0].log()).sum()).item())
    return

if __name__ == '__main__':
    main()

