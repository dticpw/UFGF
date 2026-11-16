import numpy as np

# Reference https://github.com/OFA-Sys/Chinese-CLIP/blob/master/cn_clip/training/scheduler.py

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def assign_multi_lr(optimizer, scale_factor):
    """按组应用学习率缩放因子"""
    for param_group in optimizer.param_groups:
        base_lr = param_group["initial_lr"]
        param_group["lr"] = base_lr * scale_factor

def cosine_lr(optimizer, warmup_steps, total_steps):
    def _lr_adjuster(step):
        if step < warmup_steps:
            # 预热阶段：线性增长
            scale = (step + 1) / warmup_steps
        else:
            # 余弦退火阶段
            e = step - warmup_steps
            es = total_steps - warmup_steps
            scale = 0.5 * (1 + np.cos(np.pi * e / es))
        
        assign_multi_lr(optimizer, scale)
        return scale
    return _lr_adjuster




# # # 记得修改scheduler = cosine_lr(optimizer, (args.lr,) args.warmup, total_steps)
# def assign_learning_rate(optimizer, new_lr):
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = new_lr

# def _warmup_lr(base_lr, warmup_length, step):
#     return base_lr * (step + 1) / warmup_length

# def cosine_lr(optimizer, base_lr, warmup_length, steps):
#     def _lr_adjuster(step):
#         if step < warmup_length:
#             lr = _warmup_lr(base_lr, warmup_length, step)
#         else:
#             e = step - warmup_length
#             es = steps - warmup_length
#             lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
#         assign_learning_rate(optimizer, lr)
#         return lr
#     return _lr_adjuster

