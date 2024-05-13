from stringprep import c22_specials
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

class Loss_Record(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self):
        super(Loss_Record, self).__init__()
        self.loss = torch.tensor(0.0)
        self.num = 0
    def update(self,loss,batch_size):
        self.loss+=loss.detach().cpu().item()
        self.num+=batch_size

    def reset(self):
        self.loss = torch.tensor(0.0)
        self.num = 0

    def mean_loss(self):
        if self.num==0:
            self.num+=0.01
        return self.loss/self.num
    


def split_dset_structure_model_source_target(file_name):
    file_name=file_name.split('.mat')[0]
    target=file_name[-1]
    source=file_name[-3]
    file_name=file_name[:-4]
    dset=file_name.split('_')[0]
    r=len(dset)
    model_name=file_name[r+1:]

    model_structure=model_name.split('_')[0]
    return dset,model_structure,model_name,source,target


def get_source_model_path(pth_save_dir,dset,model_name,source):
   
    model_name='_'+model_name
    path=pth_save_dir+model_name+'/'+dset+'/'+source+'/'
    return path