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
    os.makedirs(os.path.join(dir_path, 'ufr'), exist_ok=True)
    global UFR_DIR
    UFR_DIR = os.path.join(dir_path, 'ufr')
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
    diag = None
    if not args.eval:
        diag = Diag(dir_path, args.dataset_root, args.noise_ratio, 'ufg', logger)
        diag.embed_stats(model, dev_dataloader, -1)  # pretrained CLIP reference

    #### learning rate
    high_lr_params = []

    for name, param in crossAttn.named_parameters(): # 就那四个linear
        high_lr_params.append(param)
    # 获取其他参数（假设model包含除crossAttn之外的所有参数）
    base_params = [p for p in model.parameters() if not any(p is hp for hp in high_lr_params)] # 如果crossAttn是独立模块需要额外处理
    ####

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay) 
    # scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    proj_lr = float(os.environ.get("UFGF_PROJ_LR", 1e-5))  # lr of the UFR projection (paper: 1e-5)
    optimizer = optim.AdamW(
        [
            {"params": high_lr_params, "lr": proj_lr, "initial_lr": proj_lr},  #
            # {"params": high_lr_params, "lr": 5e-6, "initial_lr": 5e-6},  #
            {"params": base_params, "lr": args.lr, "initial_lr": args.lr}      # 基础学习率
        ],
        betas=(args.beta1, args.beta2), 
        eps=args.eps, 
        weight_decay=args.weight_decay
    ) 
    scheduler = cosine_lr(optimizer, args.warmup, total_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", total_steps)

    best_score = 0
    clip_sim_threshold = float(os.environ.get("UFGF_SIGMA", "0.2"))  # revision: sigma sensitivity
    
    for epoch in range(args.epochs):

        # NPC training
        # train(args, model, siamese_model, crossAttn, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, siamese_scheduler, optimizer, siamese_optimizer, loss)
        train(args, model, crossAttn, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, optimizer, loss, diag)
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
def train(args, model, crossAttn, clip_sim_threshold, train_dataloader, train_length, epoch, scheduler, optimizer, loss, diag=None):

    model.train()
    # siamese_model.train()

    # 初始化滑动窗口 # 每个epoch都会重新构建
    sample_score_max_window = SlidingWindow(args.display)
    sample_score_mean_window = SlidingWindow(args.display)
    ori_score_mean_window = SlidingWindow(args.display)
    ori_score_max_window = SlidingWindow(args.display)
    
    noisy_count = 0
    replaced_count = 0
    clip_sim_threshold = float(os.environ.get("UFGF_SIGMA", "0.2"))  # revision: sigma sensitivity
    logit_scale = torch.tensor(100.0, device='cuda')
    fea_dim = 512

    stored_I = [] # 可视化相关参数
    stored_T = []
    stored_PI = []
    stored_PT = []
    store_count = 0
    visual_count = 10


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
            noisy_count = noisy_count + torch.sum(~clean_mask).item() 

        #### 伪文本生成
        # 1.获得重构的图像和文本 # 一个batch产一个weight (8,128,128)
        # attnI, attnT = crossAttn(ImgFea_b_grad.detach(), ImgFea_MBI_s_nograd.detach(), TxtFea_MBI_s_nograd.detach()) # 其实后两个的detach()在冗余
        # attnI, attnT = crossAttn(ImgFea_MBI_s_nograd.detach(), ImgFea_b_grad.detach(), TxtFea_b_grad.detach())
        attnI, attnT = crossAttn(ImgFea_b_grad.detach(), TxtFea_b_grad.detach())
        attnI = l2norm(attnI, -1)
        attnT = l2norm(attnT, -1)


        # # 2.重构loss
        lambda1 = args.lambda1
        lambda2 = args.lambda2
        loss_l2I = (torch.sum((attnI - ImgFea_b_grad.detach())**2,dim=-1))**0.5
        loss_I = torch.mean(loss_l2I, dim=-1) * (lambda1) * (lambda2)
        ### 文本模态重构损失
        loss_l2T = (torch.sum((attnT[clean_mask] - (TxtFea_b_grad.detach())[clean_mask])**2,dim=-1))**0.5 ## 只用给不用重构的那一部分打loss就好
        loss_T = torch.mean(loss_l2T, dim=-1) * (lambda1) * (1-lambda2)

        ## 全部替换不太好，还得检验一下
        with torch.no_grad(): ### 计算置信度的本质是"评估"而非"学习",梯度不从这里流向attnT生成模块
            clip_sim = F.cosine_similarity(ImgFea_b_grad, attnT)
        #### 量化生成样本的品质 # 更新滑动窗口
        sample_score_max_window.add(torch.max(clip_sim).item())
        sample_score_mean_window.add(torch.mean(clip_sim).item())
        ori_score_mean_window.add(torch.mean(ori_sim).item())
        ori_score_max_window.add(torch.max(ori_sim).item())
        
        # 4.定位替换
        replace_mask = ((clip_sim > ori_sim) & (clip_sim > clip_sim_threshold)) & (~clean_mask) # 这clip_sim_max是否有问题#没有问题，显示的是最大值的均值，还是有比2.0大的
        # replace_mask = ((clip_sim > ori_sim)) & (~clean_mask)
        texts_new_fea = torch.where(replace_mask.unsqueeze(1), attnT, TxtFea_b_grad) # 如果~clean_mask(是噪声)，但没找到合适的伪样本(clip_sim > ori_sim)，那还是不替换
        # 对替换的样本计数
        replaced_count = replaced_count + torch.sum(replace_mask).item()


        #### loss 
        batch_i2t_sim_new, batch_t2i_sim_new, _, _ = model(ImgFea_b_grad, texts_new_fea) # (bs,bs) # 确实啊，前面的用不到
        batch_i2t_ori_loss, batch_t2i_ori_loss = loss(batch_i2t_sim_new, batch_t2i_sim_new, 'batch')
        batch_loss = torch.mean((replace_mask | clean_mask) * (batch_i2t_ori_loss + batch_t2i_ori_loss)/2)            
        
        ### 那些被替换的地方的图像没有被优化到啊，
        # 同一个clean_mask, 不过replace_mask部分要重新算
        final_loss = batch_loss + loss_T + loss_I
        if diag is not None:
            diag.record_batch(epoch, step, cap_idx, ori_sim, clip_sim, clean_mask, replace_mask,
                              {'batch_loss': batch_loss.item(), 'loss_I': loss_I.item(), 'loss_T': loss_T.item(),
                               'ce_mean': ((batch_i2t_ori_loss + batch_t2i_ori_loss) / 2).item()},
                              [g['lr'] for g in optimizer.param_groups])
        

        # ##可视化
        # # 确保你的区分化 和 高质量
        # if (idx % (800//visual_count) == 0) and store_count < visual_count:
        #     stored_I.append(ImgFea_b_grad[0])
        #     stored_T.append((TxtFea_b_grad)[0])
        #     stored_PI.append((attnI[0]))
        #     stored_PT.append((attnT)[0])
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

        # if idx == 100:
        #     sys.exit(0)
        if epoch == 1 and idx == 0:  #
            # attnT is shape (bs, dim) — 取平均生成通用伪文本
            universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
            torch.save(universal_pseudo_text, os.path.join(UFR_DIR, "universal_pseudo_text1.pt"))
            logger.info("UFR saved: %s", os.path.join(UFR_DIR, "universal_pseudo_text1.pt"))
        if epoch == 2 and idx == 0:  #
            universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
            torch.save(universal_pseudo_text, os.path.join(UFR_DIR, "universal_pseudo_text2.pt"))
            logger.info("UFR saved: %s", os.path.join(UFR_DIR, "universal_pseudo_text2.pt"))
        if epoch == 3 and idx == 0:  #
            universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
            torch.save(universal_pseudo_text, os.path.join(UFR_DIR, "universal_pseudo_text3.pt"))
            logger.info("UFR saved: %s", os.path.join(UFR_DIR, "universal_pseudo_text3.pt"))
        if epoch == 4 and idx == 0:  #
            universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
            torch.save(universal_pseudo_text, os.path.join(UFR_DIR, "universal_pseudo_text4.pt"))
            logger.info("UFR saved: %s", os.path.join(UFR_DIR, "universal_pseudo_text4.pt"))
        if epoch == 5 and idx == 0:  #
            universal_pseudo_text = attnT.mean(dim=0, keepdim=True).detach().cpu()  # (1, dim)
            torch.save(universal_pseudo_text, os.path.join(UFR_DIR, "universal_pseudo_text5.pt"))
            logger.info("UFR saved: %s", os.path.join(UFR_DIR, "universal_pseudo_text5.pt"))
            # sys.exit(0)
        if (idx % (args.display) == 0 and idx!=0):
            logger.info("lam1/lam2: %.1f/%.2f, Epoch: %d/%d, step:%d/%d, lr: %.2e/%.2e, batch_loss: %.3f, loss_I: %.3f, loss_T: %.3f, su/sm/ou/om: %.3f/%.3f/%.3f/%.3f, R_C/N_C: %d/%d", 
                        lambda1, lambda2, epoch + 1, args.epochs, idx, len(train_dataloader), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                        batch_loss, loss_I, loss_T, sample_score_mean_window.get_n_mean(), sample_score_max_window.get_n_mean(),
                        ori_score_mean_window.get_n_mean(), ori_score_max_window.get_n_mean(), replaced_count, noisy_count)
            noisy_count = 0
            replaced_count = 0
            
    return

if __name__ == '__main__':
    main()

