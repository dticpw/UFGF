import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def ce(sim_matrix):
    logpt = F.log_softmax(sim_matrix, dim=-1)
    logpt = torch.diag(logpt)
    return -logpt

class TotalLoss(nn.Module):
    def __init__(self,):
        super(TotalLoss, self).__init__()
    def forward(self, i2t_sim_matrix, t2i_sim_matrix, subset='batch'):
        if subset == 'mbank':
            loss = (ce(i2t_sim_matrix) + ce(t2i_sim_matrix)) / 2
            return loss
        elif subset == 'batch':
            i2t_loss = ce(i2t_sim_matrix).mean() # (bs,bs)
            t2i_loss = ce(t2i_sim_matrix).mean()
            return i2t_loss, t2i_loss
    # def forward(self, i2t_sim_matrix, t2i_sim_matrix, subset=None, weights=None, causal_strengths=None):
    #     """
    #     Enhanced forward method with support for causal weighting
        
    #     Args:
    #         i2t_sim_matrix: Image-to-text similarity matrix
    #         t2i_sim_matrix: Text-to-image similarity matrix
    #         subset: 'mbank', 'batch', or 'causal'
    #         weights: Optional sample quality weights for causal weighting
    #         causal_strengths: Optional causal strengths for causal weighting
    #     """
    #     if subset == 'mbank':
    #         loss = (ce(i2t_sim_matrix) + ce(t2i_sim_matrix)) / 2
    #         return loss
            
    #     elif subset == 'causal' and weights is not None and causal_strengths is not None:
    #         # Get standard cross-entropy loss for each sample
    #         i2t_loss = ce(i2t_sim_matrix)  # (bs,)
    #         t2i_loss = ce(t2i_sim_matrix)  # (bs,)
            
    #         # Apply sample quality weighting
    #         combined_loss = (i2t_loss + t2i_loss) / 2
    #         weighted_loss = weights * combined_loss
            
    #         # Apply causal strength adjustment
    #         # Higher causal strength = higher weight (1.0 to 2.0 scaling)
    #         causal_factor = 1.0 + causal_strengths  
    #         final_loss = weighted_loss * causal_factor
            
    #         # Return mean loss
    #         return final_loss.mean()
        
    #     else: # subset == 'batch':
    #         # Default standard calculation
    #         i2t_loss = ce(i2t_sim_matrix).mean() 
    #         t2i_loss = ce(t2i_sim_matrix).mean()
    #         return i2t_loss, t2i_loss







        



