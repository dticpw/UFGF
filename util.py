import os
import sys
import datetime
import torch
import random
import numpy as np
import logging
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from datetime import datetime

def cosine_sim(x, y):
    return x @ (y.T)

def set_seed_logger(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False

    return args.seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


#%% 3.准备可视化数据 -----------------------------------------
def prepare_tsne_data(features_dict):
    # 合并特征并创建标签
    features = []
    labels = []
    
    # 图像特征（假设每个模态特征形状为[batch_size, feature_dim]）
    if 'ImgFea_b_grad' in features_dict:
        features.append(features_dict['ImgFea_b_grad'])
        labels.extend([0] * len(features_dict['ImgFea_b_grad']))  # 0表示图像
    
    # 原始文本特征
    if 'TxtFea_b_grad' in features_dict:
        features.append(features_dict['TxtFea_b_grad'])
        labels.extend([1] * len(features_dict['TxtFea_b_grad']))  # 1表示原始文本
    
    # 新文本特征（如果需要）
    if 'attnT' in features_dict:
        features.append(features_dict['attnT'])
        labels.extend([2] * len(features_dict['attnT']))  # 2表示新文本
    
    if 'attnI' in features_dict:
        features.append(features_dict['attnI'])
        labels.extend([3] * len(features_dict['attnI']))  # 2表示新文本
    
    return np.concatenate(features, axis=0), np.array(labels)

### PCA+tSNE
def run_tsne(data):
    # Step 1: 先用 PCA 降到 50 维
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(data)

    # Step 2: 再用 t-SNE 降到 2 维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_result = tsne.fit_transform(data_pca)
    return tsne_result

def get_name(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    existing_files = os.listdir(folder) # 存储文件夹名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_file = os.path.join(folder, f'featurelist_{current_time}.png')
    return img_file

## tsne可视化 
def plot_tsne(tsne_data, labels, title, folder):
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'green', 'gray']  # 0:图像，1:原始文本，2:新文本
    markers = ['o', 'x', 'v', '*']
    for i in range(4):
        mask = labels == i
        plt.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                    c=colors[i], marker=markers[i],
                    label=['Image', 'Original Text', 'attnT', 'attnI'][i], alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.show()

    img_file = get_name(folder)
    plt.savefig(img_file)
    plt.close()



# ## UMAP 可视化
# def plot_umap(data, labels, title, folder, n_neighbors=15, min_dist=0.1):
#     """
#     data: numpy array of shape (N, D)
#     labels: numpy array of shape (N,), with integer class labels (0, 1, 2)
#     """
#     reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
#     umap_data = reducer.fit_transform(data)  # (N, 2)

#     plt.figure(figsize=(10, 8))
#     colors = ['blue', 'orange', 'green']
#     for i in range(3):
#         mask = labels == i
#         plt.scatter(umap_data[mask, 0], umap_data[mask, 1],
#                     c=colors[i], label=['Image', 'Original Text', 'attnT'][i], alpha=0.6)

#     plt.legend()
#     plt.title(title)
#     plt.xlabel('UMAP 1')
#     plt.ylabel('UMAP 2')

#     img_file = get_name(folder)
#     plt.savefig(img_file)
#     print(f"UMAP plot saved to {img_file}")

def plot_sim(simmat, folder):
    plt.figure(figsize=(10, 8))
    plt.imshow(simmat, cmap='viridis')  # You can choose different colormaps as per your preference
    plt.colorbar()  # Add color bar for reference
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_file = os.path.join(folder, f'weight_{current_time}.png')
    plt.savefig(img_file)

def feature_plot(last_batch_features, attnWeight = None):
    folder = "/I2T/NPC-master/logs/f30k0_4/plots_0620ex"
    # 1.数据划分
    data, labels = prepare_tsne_data({
        'ImgFea_b_grad': last_batch_features['ImgFea_b_grad'],
        'TxtFea_b_grad': last_batch_features['TxtFea_b_grad'],
        'attnT': last_batch_features['attnT']
    })
    # simmat = cosine_sim(last_batch_features['ImgFea_b_grad'],last_batch_features['TxtFea_b_grad'])
    # plot_sim(simmat, folder)
    # plot_sim(attnWeight, folder)
    

    # 2.对两种情况分别降维
    tsne2d = run_tsne(data)

    # 3.绘制图
    plot_tsne(tsne2d, labels, 'tsne2d', folder)

    # plot_umap(data, labels, title='UMAP visualization', folder=folder)
    print("plotted!=====")




def visualize_and_save(tensor, img_folder, tensor_file=None):

    if not os.path.exists(img_folder):
        os.makedirs(img_folder, exist_ok=True)

    # existing_files = os.listdir(img_folder) # 存储文件夹名
    # # 遍历existing_files列表中以"tensor_"开头的文件名，切掉'_'前的和'.'后的，转成int
    # existing_indices = [int(filename.split('_')[1].split('.')[0]) for filename in existing_files if filename.startswith('tensor_')]
    # if existing_indices:
    #     index = max(existing_indices) + 1
    # else:
    #     index = 0
    date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 如果张量在GPU上，将其移到CPU
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # 将张量转换为ndarray类型，确保可以与matplotlib兼容
        tensor = tensor.detach().numpy()  # 转换为numpy.ndarray

    # labels = ['t', 'tc', 'p', 'pc']
    # positions = [0,10,20,30]
    # tick_pos = [(positions[i]+4) for i in range(len(labels))]
    # plt.imshow(tensor, cmap='viridis')  # You can choose different colormaps as per your preference
    # plt.colorbar()  # Add color bar for reference
    # # plt.xticks(range(0, tensor.shape[1], 20))  # 以4为单位设置x轴刻度
    # # plt.yticks(range(0, tensor.shape[0], 20))  # 以4为单位设置y轴刻度
    # plt.xticks(tick_pos, labels)
    # plt.yticks(tick_pos, labels)
    # img_file = os.path.join(img_folder, f'tensor_{date}.png')
    # plt.savefig(img_file, dpi=300)
    # plt.close()


    plt.imshow(tensor, cmap='viridis')  # You can choose different colormaps as per your preference
    plt.colorbar()  # Add color bar for reference
    plt.xticks(range(0, tensor.shape[1], 10))  # 以4为单位设置x轴刻度
    plt.yticks(range(0, tensor.shape[0], 10))  # 以4为单位设置y轴刻度
    img_file = os.path.join(img_folder, f'tensor_{date}.png')
    plt.savefig(img_file, dpi=300)
    plt.close()







