import torch
from tqdm import tqdm
import numpy as np
from util import visualize_and_save

def evaluate(args, model, dataloader, logger):
    # 可视化的话找 dataloader.dataset.captions 和 dataloader.dataset.image_name
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch # getitem定的啊
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids] # （1000，512）
        text_features = torch.cat(text_features, dim=0) # （5000，512）

        sim_matrix = []
        
        for idx, image_feat in tqdm(enumerate(image_features)):
            sim_line = image_feat @ text_features.t()
            sim_matrix.append(sim_line.unsqueeze(0).cpu())
        
        sim_matrix = torch.cat(sim_matrix, dim=0) # (1000,5000)? 这么快？
        label = torch.eye(num_imgs).unsqueeze(-1).repeat(1,1,num_anns).view(-1, num_ids)
        # universal_pseudo_text = torch.load("/I2T/NPC-master/logs/f30k0_4/uni_txt/universal_pseudo_text04.pt").cuda()
        # sim_mat_uni = image_feat @ universal_pseudo_text.t()
        results = metric_compute(sim_matrix, label, logger)
    return results['mean_R1']

def evaluate_1K(args, model, dataloader, logger):
    logger.info("Testing MSCOCO 1K.")
    model.eval()
    with torch.no_grad():
        image_features = []
        text_features = []
        num_anns = dataloader.dataset.num_anns
        num_ids = len(dataloader.dataset)
        num_imgs = dataloader.dataset.img_length
        for idx, batch in enumerate(dataloader):

            images, texts, img_id = batch 
            
            images = images.cuda()
            texts = texts.cuda()

            batch_image_features = model.encode_image(images)
            batch_text_features = model.encode_text(texts)

            batch_image_features = batch_image_features / batch_image_features.norm(dim=1, keepdim=True)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=1, keepdim=True)

            image_features.append(batch_image_features)
            text_features.append(batch_text_features)

            if idx % args.display == 0:
                logger.info("step:%d/%d", idx, len(dataloader))

        images_ids = torch.arange(0, num_ids, num_anns).cuda()
        image_features = torch.cat(image_features, dim=0)[images_ids]
        text_features = torch.cat(text_features, dim=0)

        sim_matrix = []
        i2t_R1 = 0
        i2t_R5 = 0
        i2t_R10 = 0
        t2i_R1 = 0 
        t2i_R5 = 0
        t2i_R10 = 0
        for i in range(5):
            img_embs = image_features[i * 1000 : (i + 1) * 1000]
            cap_embs = text_features[i * 5000 : (i + 1) * 5000]

            sim_matrix = img_embs @ cap_embs.t()
            
            label = torch.eye(1000).unsqueeze(-1).repeat(1,1,num_anns).view(-1, int(num_ids/5)).cuda()
            results = metric_compute(sim_matrix, label, logger)
            i2t_R1 = i2t_R1 + results['i2t_R@1']
            i2t_R5 = i2t_R5 + results['i2t_R@5']
            i2t_R10 = i2t_R10 + results['i2t_R@10']
            t2i_R1 = t2i_R1 + results['t2i_R@1']
            t2i_R5 = t2i_R5 + results['t2i_R@5']
            t2i_R10 = t2i_R10 + results['t2i_R@10']
        logger.info("Image-to-Text Average:")
        logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(i2t_R1/5, i2t_R5/5, i2t_R10/5))
        logger.info("Text-to-Image Average:")
        logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(t2i_R1/5, t2i_R5/5, t2i_R10/5))
        mean_R1 = (i2t_R1/5 + t2i_R1/5)/2
        Rsum = i2t_R1/5+i2t_R5/5+i2t_R10/5+t2i_R1/5+t2i_R5/5+t2i_R10/5
        logger.info("Mean R1: {:.2f} - Rsum: {:.2f}".format(mean_R1, Rsum))
    return mean_R1

def metric_compute(sim_matrix, label, logger):
    '''
    sim_matrix: (1000,5000)
    label:      
    '''
    results = {}
    
    # Image-to-Text
    i2t_rank_matrix = (-sim_matrix).argsort().argsort() + 1 # (1000,5000)
    i2t_gt_rk_matrix = label * i2t_rank_matrix
    i2t_gt_rk_matrix[i2t_gt_rk_matrix==0] = 1e9
    i2t_min_rank = i2t_gt_rk_matrix.min(1).values # (1000)
    # ranks_vis = i2t_min_rank[:500].reshape(-1,25) #### 
    # ranks_vis = np.clip(ranks_vis, 0, 11) ## 
    # rank_label = i2t_rank_matrix[label.bool()].reshape(-1,5) ## 不对啊这个只知道排到哪里去了但不知道前五排了啥啊。
    # rank_detail = torch.topk(i2t_rank_matrix, k=5, dim=1, largest=False).indices
    # tensor_cpu = rank_label.cpu()
    # torch.set_printoptions(precision=4, sci_mode=False)  # 可选：控制打印格式
    # with open('/I2T/NPC-master/logs/f30k0_4/plots_0620case/tensor.txt', 'w') as f:
    #     f.write(str(tensor_cpu))
    # tensor_cpu_detail = rank_detail.cpu()
    # torch.set_printoptions(precision=4, sci_mode=False)  # 可选：控制打印格式
    # with open('/I2T/NPC-master/logs/f30k0_4/plots_0620case/tensor_detail.txt', 'w') as f:
    #     f.write(str(tensor_cpu_detail))
    # # torch.save(rank_label, '/I2T/NPC-master/logs/f30k0_4/plots_0620case/rank_label.pt')
    # # torch.save(rank_detail, '/I2T/NPC-master/logs/f30k0_4/plots_0620case/rank_detail.pt')
    # # 
    # # visualize_and_save(ranks_vis, '/I2T/NPC-master/logs/f30k0_4/plots_0620case')
    ### case
    results['i2t_R@1'] = 100 * torch.where(i2t_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['i2t_R@5'] = 100 * torch.where(i2t_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['i2t_R@10'] = 100 * torch.where(i2t_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Image-to-Text:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['i2t_R@1'], results['i2t_R@5'], results['i2t_R@10']))
    
    # Text-to-Image
    t2i_rank_matrix = (-sim_matrix.T).argsort().argsort() + 1
    t2i_gt_rk_matrix = label.T * t2i_rank_matrix
    t2i_gt_rk_matrix[t2i_gt_rk_matrix==0] = 1e9
    t2i_min_rank = t2i_gt_rk_matrix.min(1).values # (5000)
    ### case
    # ranks_vis = i2t_min_rank[:500].reshape(-1,25)
    # ranks_vis = np.clip(ranks_vis, 0, 11) ## 
    # rank_label_t2i = t2i_rank_matrix[label.T.bool()] ## 这里不用reshape，因为每行只有一个查询
    # rank_detail_t2i = torch.topk(t2i_rank_matrix, k=5, dim=1, largest=False).indices
    # # torch.save(rank_label_t2i, '/I2T/NPC-master/logs/f30k0_4/plots_0620case/rank_label_t2i.pt')
    # # torch.save(rank_detail_t2i, '/I2T/NPC-master/logs/f30k0_4/plots_0620case/rank_detail_t2i.pt')
    # # visualize_and_save(ranks_vis, '/I2T/NPC-master/logs/f30k0_4/plots_0620case')
    
    results['t2i_R@1'] = 100 * torch.where(t2i_min_rank <= 1, 1, 0).type(torch.float32).mean()
    results['t2i_R@5'] = 100 * torch.where(t2i_min_rank <= 5, 1, 0).type(torch.float32).mean()
    results['t2i_R@10'] = 100 * torch.where(t2i_min_rank <= 10, 1, 0).type(torch.float32).mean()

    logger.info("Text-to-Image:")
    logger.info('\t>>>  R@1: {:.2f} - R@5: {:.2f} - R@10: {:.2f}'.
                format(results['t2i_R@1'], results['t2i_R@5'], results['t2i_R@10']))
    
    results['mean_R1'] = (results['i2t_R@1'] + results['t2i_R@1']) / 2
    results['Rsum'] = (results['i2t_R@1'] + results['t2i_R@1'] + results['i2t_R@5'] + results['t2i_R@5'] + results['i2t_R@10'] + results['t2i_R@10'])

    logger.info("Mean R1: {:.2f} - Rsum: {:.2f}".format(results['mean_R1'], results['Rsum']))
    
    return results
    