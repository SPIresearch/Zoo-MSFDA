import argparse
from cmath import exp
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss_function
from torch.utils.data import DataLoader
from data_list import ImageList,Splited_List
import random, pdb, math, copy
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import scipy.stats as stats
from torch.optim.lr_scheduler import StepLR
from mmd import mmd_rbf
from a_distance import calculate_a_distance
from scipy.io import loadmat

from transfer_metrics import LogME,NCE,LEEP,NCE_ours_addpz
logme = LogME(regression=False)


def obtain_label_cpu(all_output, all_label, all_fea, args):
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
   
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    #acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    #log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    
    #print(log_str+'\n')

    return pred_label.astype('int')




def snd_principle(all_output, all_label, all_fea,pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)


    
    pred=nn.Softmax(dim=1)(all_output).cuda()

    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
   
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = pred.float().cpu().numpy()#all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    dd=torch.from_numpy(dd)
    dd=nn.Softmax(dim=1)(dd/0.05)
    
    loss=torch.sum(entropy_loss(dd))/(dd.shape[0])
    loss=loss
    return loss,pred_acc





class LogisticRegressionModel(nn.Module):
    def __init__(self,feature_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(feature_dim, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x=self.linear1(x)
        x=self.linear2(x)
        return torch.sigmoid(x)

def get_reweighted_weights(data,data_no_shuffle,model,epoches=3):
    criterion = nn.BCELoss() 
    model=model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    model.train()
    
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(epoches):
        scheduler.step()
        for input,label in data:
            
            input, label = input.cuda(), label.cuda()
            outputs = model(input)
            label=label.float().unsqueeze(1)
            #pdb.set_trace()
            #print(label.shape,outputs.shape,input.shape)
            
            loss = criterion(outputs, label)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        start_test=True
        for input,label in data_no_shuffle:

            input, label = input.cuda(), label.cuda()
            label=label.float().unsqueeze(1)
            outputs = model(input)
            loss = criterion(outputs, label)
            
            # Backward pass and optimization
           
            if start_test:
                all_output = outputs.float().cuda()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cuda()), 0)

    return all_output              
              


def ce_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    #pse_label=torch.from_numpy(pse_label).long().cuda()
    all_output=all_output.cuda()
    with torch.no_grad():
        loss=nn.CrossEntropyLoss()(all_output,pse_label)
       
    loss=loss.item()

    return loss,pred_acc

def ce_nce_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    #pse_label=torch.from_numpy(pse_label).long()
    pse_label1=get_one_hot(all_output,pse_label)#np.array(pse_label)
    predict1=get_one_hot(all_output,predict)#np.array(predict)
   
    with torch.no_grad():
        loss1=nn.CrossEntropyLoss(reduction='none')(all_output,pse_label)
        loss2=nn.CrossEntropyLoss(reduction='none')(predict1,pse_label1)#+0.001

        loss=torch.sum(loss2)/(predict1.shape[0])
    loss=loss.item()

    return loss,pred_acc

def get_one_hot(inputs,targets):
    log_probs =torch.log(inputs) #self.logsoftmax(inputs)
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).long(), 1)
    return targets

def KL_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    #pse_label=torch.from_numpy(pse_label)
    pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    with torch.no_grad():
        loss=nn.KLDivLoss()(all_output.log(),pse_label)
       
    loss=loss.item()

    return loss,pred_acc

def LogME_true_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
 
    logme = LogME(regression=False)
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    #all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    all_fea=np.array(all_fea)
    all_label=np.array(all_label)
    loss=logme.fit(all_fea,all_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc


def LogME_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    logme = LogME(regression=False)
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    #all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    #pse_label=torch.from_numpy(pse_label)
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_fea=np.array(all_fea)
    pse_label=np.array(pse_label)
    loss=logme.fit(all_fea,pse_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc


def Leep_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    #pse_label=torch.from_numpy(pse_label)
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_output=np.array(all_output)
    pse_label=np.array(pse_label)
    
    loss=LEEP(all_output,pse_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc



def NCE_pse_principle(all_output, all_label, all_fea, pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    #pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label)
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_output=np.array(all_output)
    pse_label=np.array(pse_label)
    predict=np.array(predict)
    loss=NCE(predict,pse_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc

def entropy_loss(p):

    epsilon = 1e-5
    entropy = -p * torch.log(p + epsilon)

    loss = torch.sum(entropy, dim=1)
    return loss

def entropy_principle(all_output, all_label, all_fea,pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred=nn.Softmax(dim=1)(all_output).cuda()

    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    loss=torch.sum(entropy_loss(pred))/(pred.shape[0])
    loss=-loss
    return loss,pred_acc


def MI_principle(all_output, all_label, all_fea,pse_label, args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred=nn.Softmax(dim=1)(all_output).cuda()
    softmax_out = pred
    entropy_loss = torch.mean(loss_function.Entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    entropy_loss -= gentropy_loss
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    #loss=torch.sum(entropy_loss(pred))/(pred.shape[0])
    entropy_loss=-entropy_loss
    return entropy_loss,pred_acc



def MDE_principle(all_output, all_label, all_fea,pse_label, args,T=1):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    #pdb.set_trace()
    energy = -T * (torch.logsumexp(all_output / T, dim=1))

    avg_energies = torch.log_softmax(energy, dim=0).mean()
    avg_energies = torch.log(-avg_energies).item()
  
 
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    #loss=torch.sum(entropy_loss(pred))/(pred.shape[0])
    #entropy_loss=-entropy_loss
    return avg_energies,pred_acc



def ce_principle(all_output, all_label, all_fea, pse_label,args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    pse_label=predict
    all_output = all_output.cuda()
    pse_label = pse_label.cuda()
    with torch.no_grad():
        loss=nn.CrossEntropyLoss()(all_output,pse_label)
     
    loss=loss.item()

    return loss,pred_acc

def Leep_true_principle(all_output, all_label, all_fea, pse_label,args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
   
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_output=np.array(all_output)
    all_label=np.array(all_label.long())
    
    loss=LEEP(all_output,all_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 
def MI(pred):
   
    softmax_out = pred
    entropy_loss =torch.mean(Entropy(softmax_out))
 
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    #entropy_loss -= gentropy_loss
    return entropy_loss,gentropy_loss


def MI2(pred):
   
    softmax_out = pred
    entropy_loss =torch.mean(Entropy(softmax_out))
    _,pred=torch.max(softmax_out,1)
    softmax_out=torch.eye(65)[pred]
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    #entropy_loss -= gentropy_loss
    return entropy_loss,gentropy_loss



def SUTE_principle(all_output, all_label, all_fea, pse_label,args):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred=nn.Softmax(dim=1)(all_output)
   
    entropy_loss,diversity_loss =  MI2(pred)

    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    

    #pdb.set_trace()
    #pse_label1=get_one_hot(all_output,pse_label)#np.array(pse_label)
    all_label=np.array(all_label)
    pse_label=np.array(pse_label)
    predict=np.array(predict)

   
    nce=NCE(predict,pse_label)#+0.1*entropy_pse.item()#(torch.mean(Entropy(pse_label1))).item()
    nentropy_loss=-entropy_loss
    #loss=loss2-0.1*entropy_loss#loss1#loss2-0.2*loss1#;loss2#-loss1
    return nce,nentropy_loss,diversity_loss,pred_acc
def calculate_entropy(n, c):
    # 计算每个值的概率
    probability = 1 / c
    
    # 计算对数项
    log_term = -torch.log(torch.tensor(probability))
    
    # 计算熵
    entropy = n * probability * log_term
    return entropy.item() 
def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到。")





def load_mat(file):
    result=loadmat(file)
    feature = result['ft']
    label = result['label'][0]
    output = result['output']
    pse=result['pse'][0]
   
    return feature,output,label,pse



def mmd_principle(all_output, all_label, all_fea, pse_label,args,feature_source):
    #pdb.set_trace()
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
        feature_source=torch.from_numpy(feature_source)
    except:
        pass

    all_fea=torch.nn.AdaptiveAvgPool1d(1000)(all_fea)
    feature_source=torch.nn.AdaptiveAvgPool1d(1000)(feature_source)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    min_samples = min(all_fea.shape[0],feature_source.shape[0])//10+1
    random_indices = torch.randperm(min_samples)

    # 从两个张量中分别随机抽取min_samples个样本
    all_fea = all_fea[random_indices]
    feature_source = feature_source[random_indices]

    mmd=mmd_rbf(all_fea,feature_source)
    mmd=-mmd
    return mmd,pred_acc

def A_distance_principle(all_output, all_label, all_fea, pse_label,args,feature_source):
    try:
        all_output=torch.from_numpy(all_output)
        all_label=torch.from_numpy(all_label)
        all_fea=torch.from_numpy(all_fea)
        pse_label=torch.from_numpy(pse_label).long()
        feature_source=torch.from_numpy(feature_source).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    min_samples = min(all_fea.shape[0],feature_source.shape[0])
    random_indices = torch.randperm(min_samples)

    # 从两个张量中分别随机抽取min_samples个样本
    all_fea = all_fea[random_indices]
    feature_source = feature_source[random_indices]
    a_dis=calculate_a_distance(all_fea,feature_source)
    a_dis=-a_dis
    return a_dis,pred_acc


def transfer_calcualte_for_a_model(method,output,label,feature,pse,args,feature_source=0):
    
    if method == "ANE":
        transfer_ability,pred_acc=entropy_principle(output,label,feature,pse,args)
    if method == "SND":
        transfer_ability,pred_acc=snd_principle(output,label,feature,pse,args)
    elif method == "NMI":
        transfer_ability,pred_acc=MI_principle(output,label,feature,pse,args)
    elif method == "LogME_pse":
        transfer_ability,pred_acc=LogME_pse_principle(output,label,feature,pse,args)
    elif method == "LEEP_pse":
        transfer_ability,pred_acc=Leep_pse_principle(output,label,feature,pse,args)
    elif method == "LogME_true": # using the target label
        transfer_ability,pred_acc=LogME_true_principle(output,label,feature,pse,args)
    elif method == "LEEP_true":# using the target label
        transfer_ability,pred_acc=Leep_true_principle(output,label,feature,pse,args)
    elif method == "MMD": # using the source data
        transfer_ability,pred_acc=mmd_principle(output,label,feature,pse,args,feature_source)

    elif method == "MDE":
        transfer_ability,pred_acc=MDE_principle(output,label,feature,pse,args,T=1)

    elif method == "A_distance": # using the source data
        transfer_ability,pred_acc=A_distance_principle(output,label,feature,pse,args,feature_source)
   
    elif method=='SUTE':
        nce,nentropy_loss,diversity_loss,pred_acc=SUTE_principle(output,label,feature,pse,args)

        #print(nce,nentropy_loss,diversity_loss)
        tau_h=2/3#int(args.class_num/3*2)
        tau_l=1/2#int(args.class_num/2)
        if diversity_loss>calculate_entropy(args.class_num,args.class_num)*tau_h:#calculate_entropy(tau_h,tau_h):
            diversity_loss=calculate_entropy(args.class_num,args.class_num)*tau_h

        transfer_ability=10*nce+0.1*nentropy_loss+1*diversity_loss
        if diversity_loss<calculate_entropy(args.class_num,args.class_num)*tau_l:
            diversity_loss=-float('inf')
            transfer_ability=-float('inf')
        if nce>-1e-10:
            transfer_ability=-float('inf')
    try:
        pred_acc=pred_acc.item()
        transfer_ability=transfer_ability.item()
    except:
        pass

    return transfer_ability,pred_acc





def draw_histo_ce(model_config_file,feature_path,transferability_output_filename,method,args):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set=[]
    pred_acc_set=[]
    pred_acc_set_all=[]
    all_output=[]

    


    if file_contents:
        import matplotlib.pyplot as plt

        
        with open(transferability_output_filename, 'w') as output_file:
            for line in file_contents:
                source=line.split(".mat")[0][-3]
                source_line=line[:-5]+source+'.mat'
                #pdb.set_trace()
                feature,output,label,pse=load_mat(feature_path+line)
                outputs=torch.from_numpy(output)
                labels=torch.from_numpy(label).long()
                losses = F.cross_entropy(outputs, labels, reduction='none')  # reduction='none'表示不进行汇总，而是保留每个样本的损失值
                plt.figure()
                # 画出分布图
                plt.hist(losses.numpy(), bins=20)  # 将Tensor转换为NumPy数组，并画出直方图
                plt.xlabel('Cross-Entropy Loss')
                plt.ylabel('Frequency')
                #plt.title('Distribution of Cross-Entropy Loss')
                # 保存图片
                plt.savefig(f'./fig/{line.split(".")[0]}.png')



    print("Transferability file is saved in {}.".format(transferability_output_filename))

    print("Mean Accuracy",sum(pred_acc_set_all)/len(pred_acc_set_all))
   
    a=pd.Series(loss_set)
    b=pd.Series(pred_acc_set)
    print("Spearman",stats.spearmanr(a,b))

    return all_output



def transfer_calcualte_for_individual_models(model_config_file,feature_path,transferability_output_filename,method,args):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set=[]
    pred_acc_set=[]
    pred_acc_set_all=[]
    all_output=[]

    


    if file_contents:


        
        with open(transferability_output_filename, 'w') as output_file:
            for line in file_contents:
                source=line.split(".mat")[0][-3]
                source_line=line[:-5]+source+'.mat'
                #pdb.set_trace()
                feature,output,label,pse=load_mat(feature_path+line)
                feature_source,_,_,_=load_mat(feature_path+line)
                transfer_ability,pred_acc=transfer_calcualte_for_a_model(method,output,label,feature,pse,args,feature_source)
                #print(line,transfer_ability,pred_acc)
                pred_acc_set_all.append(pred_acc)

                if transfer_ability<-100:
                    #transfer_ability=-float('inf')
                    continue
                all_output.append(output)
                output_file.write(line+" "+str(transfer_ability)+" "+str(pred_acc)+ '\n')
                loss_set.append(transfer_ability)
                pred_acc_set.append(pred_acc)

    print("Transferability file is saved in {}.".format(transferability_output_filename))

    print("Mean Accuracy",sum(pred_acc_set_all)/len(pred_acc_set_all))
   
    a=pd.Series(loss_set)
    b=pd.Series(pred_acc_set)
    print("Spearman",stats.spearmanr(a,b))

    return all_output


def transfer_calcualte_for_individual_models2(model_config_file,feature_path,transferability_output_filename,method,args):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set=[]
    pred_acc_set=[]
    pred_acc_set_all=[]
    all_output=[]

    


    if file_contents:


        
        with open(transferability_output_filename, 'w') as output_file:
            for line in file_contents:
                source=line.split(".mat")[0][-3]
                source_line=line[:-5]+source+'.mat'
                #pdb.set_trace()
                feature,output,label,pse=load_mat(feature_path+line)
                feature_source,_,_,_=load_mat(feature_path+line)
                transfer_ability,pred_acc=transfer_calcualte_for_a_model(method,output,label,feature,pse,args,feature_source)
                print(line,transfer_ability,pred_acc)
                pred_acc_set_all.append(pred_acc)

                if transfer_ability<-100:
                    #transfer_ability=-float('inf')
                    continue
                all_output.append(output)
                output_file.write(line+" "+str(transfer_ability)+" "+str(pred_acc)+ '\n')
                loss_set.append(transfer_ability)
                pred_acc_set.append(pred_acc)

    print("Transferability file is saved in {}.".format(transferability_output_filename))

    print("Mean Accuracy",sum(pred_acc_set_all)/len(pred_acc_set_all))
   
    a=pd.Series(loss_set)
    b=pd.Series(pred_acc_set)
    print("Spearman",stats.spearmanr(a,b))

    return all_output,pred_acc_set
def print_acc_of_each_model(model_config_file,feature_path,transferability_output_filename,args):
    print(model_config_file)
    print(transferability_output_filename)
    method="entropy"
    file_contents = read_text_file(model_config_file)
    loss_set=[]
    pred_acc_set=[]
    pred_acc_set_all=[]
    all_output=[]

    if file_contents:
        with open(transferability_output_filename, 'w') as output_file:
            for line in file_contents:
                source=line.split(".mat")[0][-3]
                source_line=line[:-5]+source+'.mat'
                #pdb.set_trace()
                feature,output,label,pse=load_mat(feature_path+line)
                feature_source,_,_,_=load_mat(feature_path+source_line)
                transfer_ability,pred_acc=transfer_calcualte_for_a_model(method,output,label,feature,pse,args,feature_source)
                print(line,'Acc:',pred_acc)
                pred_acc_set_all.append(pred_acc)

                all_output.append(output)
                output_file.write(line+" Acc:"+str(pred_acc)+ '\n')
             
    return 1




def read_transferability_text_file(file_path):
    model_names = []
    transferability_metrics = []
    accuracies = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用strip()方法去除行末的换行符，并使用split()方法分割每一行
            columns = line.strip().split()
            
            # 将每一行的数据转化为适当的数据类型，例如第三列转化为浮点数
            model_name = columns[0]
            transferability_metric = float(columns[1])
            accuracy = float(columns[2])
            
            # 将数据添加到各自的列表中
            model_names.append(model_name)
            transferability_metrics.append(transferability_metric)
            accuracies.append(accuracy)
    return model_names,transferability_metrics,accuracies


def sort_lists_by_transferability(model_names, transferability_metrics,all_output):
    # 使用 zip 将三个列表打包成元组列表
    zipped_data = zip(transferability_metrics, model_names,all_output)
    
    # 按照 transferability_metrics 的值进行排序，由大到小
    sorted_data = sorted(zipped_data, key=lambda x: x[0], reverse=True)
    
    # 解压排序后的数据
    sorted_transferability_metrics, sorted_model_names,sorted_output = zip(*sorted_data)
    
    return sorted_model_names, sorted_transferability_metrics,sorted_output



def sort_lists_by_div(model_names, div_metrics,div_tran):
    # 使用 zip 将三个列表打包成元组列表
    zipped_data = zip(div_metrics, model_names,div_tran)
    
    # 按照 transferability_metrics 的值进行排序，由大到小
    sorted_data = sorted(zipped_data, key=lambda x: x[0], reverse=True)
    
    # 解压排序后的数据
    try:
        sorted_div_metrics, sorted_model_names,div_tran = zip(*sorted_data)
    except:
        sorted_div_metrics, sorted_model_names,div_tran = [],[],[]
    return sorted_model_names, sorted_div_metrics,div_tran




from hsic import hsic_gam
def cal_div(feature_path,model_set,model):
    div=0
    
    feature,_,_,_=load_mat(feature_path+model)
    for new_model in model_set:
        #a = np.arange(0, feature.shape[0])  # 如果 a 是一个列表，你可以使用 np.arange(len(a))
        # 随机抽取一千个数据的索引
        z=min(1000,feature.shape[0])
        random_indices = np.random.choice(feature.shape[0], z, replace=False)
        new_feature,new_output,label,_=load_mat(feature_path+new_model)
        div+=hsic_gam(new_feature[random_indices],feature[random_indices])[0]
    return div


def transfer_measure_for_ensemble(feature_path,model_set,method,args):
    
    for new_model in model_set:
        
        new_feature,new_output,label,_=load_mat(feature_path+new_model)
        z=(new_feature.shape[0])//1
        lists=[i*1 for i in range(z)]
        new_feature=new_feature[lists]
        new_output=new_output[lists]
        label=label[lists]

        new_feature=torch.from_numpy(new_feature)#*dis_tran
        new_output=torch.from_numpy(new_output)#*dis_tran
        label=torch.from_numpy(label)
        
        
        try:
            all_feature=torch.cat([all_feature,new_feature],1)
            all_output=all_output+new_output
        except:
            all_output=new_output
            all_feature=new_feature
    _, predict = torch.max(all_output, 1)
   
    
    pse=obtain_label_cpu(all_output,label,all_feature,args)
    pse=torch.from_numpy(pse)
    #pred_acc=torch.sum(pse==label)/float(predict.size()[0])
    
    transfer_ability,acc=transfer_calcualte_for_a_model(method,all_output,label,all_feature,pse,args)
   
    return transfer_ability,acc
        
    

def transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args):
    attempt_tran=np.array(attempt_tran)
    attempt_tran=torch.from_numpy(attempt_tran)
    attempt_tran=torch.nn.Softmax(0)(attempt_tran)
    for new_model in attempt_set:
        
        _,new_output,label,_=load_mat(feature_path+new_model)
       
        new_output=torch.from_numpy(new_output)#*dis_tran
        label=torch.from_numpy(label)
        
        
        try:
           
            all_output=all_output+new_output
        except:
            all_output=new_output
            
    _, predict = torch.max(all_output, 1)
   

  
    pred_acc=torch.sum(predict==label)/float(predict.size()[0])
    #print("tran_pred accuracy:",pred_acc)
   
    return predict,label,pred_acc
        





def model_selection_function2(K,feature_path,transferability_file_path,all_output,method,args): #transferability file path
    model_names,transferability_metrics,accuracies= read_transferability_text_file(transferability_file_path)



    sorted_model_names, sorted_transferability_metrics,sorted_output = sort_lists_by_transferability(model_names, transferability_metrics,all_output)

    save_set=sorted_model_names[:K]
    Tran_set=sorted_transferability_metrics[:K]
    Tran_output=[sorted_output[0],]
    print('selected models',save_set)

 
    attempt_tran=Tran_set
    attempt_set=copy.deepcopy(save_set)
    attempt_transfer,acc=transfer_measure_for_ensemble(feature_path,attempt_set,method,args)
    print('Aggregation performance:',acc)

    pse_tran,label,acc=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)
       
    print('Aggregation pse performance:',acc)
            




def model_selection_function(feature_path,transferability_file_path,all_output,method,args): #transferability file path
    model_names,transferability_metrics,accuracies= read_transferability_text_file(transferability_file_path)



    sorted_model_names, sorted_transferability_metrics,sorted_output = sort_lists_by_transferability(model_names, transferability_metrics,all_output)


    Initial=sorted_model_names[0]
    save_set=[Initial,]
    Tran_set=[sorted_transferability_metrics[0],]
    Tran_output=[sorted_output[0],]
    #print('Initial model',Initial)

    div_set=[]
    div_output=[]
    div_tran=[]
    save_transfer=sorted_transferability_metrics[0]
    for i,new_model in enumerate(sorted_model_names[1:]):
        # ## test
       
        new_model_trans=sorted_transferability_metrics[i+1]
        attempt_tran=Tran_set
        attempt_set=copy.deepcopy(save_set)
        attempt_set.append(new_model)
        

        attempt_transfer,acc=transfer_measure_for_ensemble(feature_path,attempt_set,method,args)
        pse_tran,label,acc=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)
        if attempt_transfer>save_transfer:
            #print('add model',new_model,'performance:',acc)
            save_set=attempt_set
            save_transfer=attempt_transfer
            
            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i+1])
        else:
            div_tran.append(sorted_transferability_metrics[i+1])
            div_set.append(new_model)
            div_output.append(sorted_output[i+1])
    transfer_model_set=copy.deepcopy(save_set)
    #print('transfer set:',transfer_model_set)

    #pse_tran=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)

    div_metrics=[]
    for i, model in enumerate(div_set):

        div_metrics.append(cal_div(feature_path,transfer_model_set,model))
    div_set, sorted_div,sorted_div_tran = sort_lists_by_div(model_names, div_metrics,div_tran)

    return transfer_model_set,div_set,attempt_tran,div_tran,sorted_div_tran,Tran_output,div_output,sorted_model_names,pse_tran,label





def model_selection_function_for_mde(feature_path,transferability_file_path,all_output,method,args): #transferability file path
    model_names,transferability_metrics,accuracies= read_transferability_text_file(transferability_file_path)

    sorted_model_names, sorted_transferability_metrics,sorted_output = sort_lists_by_transferability(model_names, transferability_metrics,all_output)


    Initial=sorted_model_names[0]
    save_set=[Initial,]
    Tran_set=[sorted_transferability_metrics[0],]
    Tran_output=[sorted_output[0],]
    print('Initial model',Initial)

    div_set=[]
    div_output=[]
    div_tran=[]
    save_transfer=sorted_transferability_metrics[0]
    for i,new_model in enumerate(sorted_model_names[1:]):
        if i>=10:
            continue
        new_model_trans=sorted_transferability_metrics[i+1]
        attempt_tran=Tran_set
        attempt_set=copy.deepcopy(save_set)
        attempt_set.append(new_model)
        
        print('add model',new_model)
        save_set=attempt_set
      
            
        attempt_tran.append(new_model_trans)
        Tran_output.append(sorted_output[i+1])
    transfer_model_set=copy.deepcopy(save_set)
    print('transfer set:',transfer_model_set)
    attempt_transfer,acc=transfer_measure_for_ensemble(feature_path,attempt_set,method,args)
    pse_tran,label,acc=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)
    #pse_tran=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)

    div_set,sorted_div_tran = model_names,div_tran
    
    return transfer_model_set,div_set,attempt_tran,div_tran,sorted_div_tran,Tran_output,div_output,sorted_model_names,pse_tran,label


def model_selection_function_for_premethod(feature_path,transferability_file_path,all_output,method,args): #transferability file path
    model_names,transferability_metrics,accuracies= read_transferability_text_file(transferability_file_path)

    sorted_model_names, sorted_transferability_metrics,sorted_output = sort_lists_by_transferability(model_names, transferability_metrics,all_output)


    Initial=sorted_model_names[0]
    save_set=[Initial,]
    Tran_set=[sorted_transferability_metrics[0],]
    Tran_output=[sorted_output[0],]
    print('Initial model',Initial)

    div_set=[]
    div_output=[]
    div_tran=[]
    save_transfer=sorted_transferability_metrics[0]
    for i,new_model in enumerate(sorted_model_names[1:]):
        new_model_trans=sorted_transferability_metrics[i+1]
        attempt_tran=Tran_set
        attempt_set=copy.deepcopy(save_set)
        attempt_set.append(new_model)
        

        attempt_transfer,acc=transfer_measure_for_ensemble(feature_path,attempt_set,method,args)
        pse_tran,label,acc=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)
        if attempt_transfer>save_transfer:
            print('add model',new_model,'performance:',acc)
            save_set=attempt_set
            save_transfer=attempt_transfer
            
            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i+1])
        else:
            div_tran.append(sorted_transferability_metrics[i+1])
            div_set.append(new_model)
            div_output.append(sorted_output[i+1])
    transfer_model_set=copy.deepcopy(save_set)
    print('transfer set:',transfer_model_set)

    #pse_tran=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)

    div_set,sorted_div_tran = model_names,div_tran

    return transfer_model_set,div_set,attempt_tran,div_tran,sorted_div_tran,Tran_output,div_output,sorted_model_names,pse_tran,label





def model_aggregating_performance(feature_path,transferability_file_path,all_output,method,args): #transferability file path
    model_names,transferability_metrics,accuracies= read_transferability_text_file(transferability_file_path)

    sorted_model_names, sorted_transferability_metrics,sorted_output = sort_lists_by_transferability(model_names, transferability_metrics,all_output)


    Initial=sorted_model_names[0]
    save_set=[Initial,]
    Tran_set=[sorted_transferability_metrics[0],]
    Tran_output=[sorted_output[0],]
    print('Initial model',Initial)

    div_set=[]
    div_output=[]
    div_tran=[]
    save_transfer=sorted_transferability_metrics[0]
    for i,new_model in enumerate(sorted_model_names[1:]):
        # ## test
        # if i>8:
        #     print('break')
        #     break
   
        new_model_trans=sorted_transferability_metrics[i+1]
        attempt_tran=Tran_set
        attempt_set=copy.deepcopy(save_set)
        attempt_set.append(new_model)
        

        attempt_transfer,acc=transfer_measure_for_ensemble(feature_path,attempt_set,method,args)
        pse_tran,label,acc=transfer_pse_for_ensemble(feature_path,attempt_tran,attempt_set,args)
        if attempt_transfer>save_transfer:
            print('add model',new_model,'performance:',acc)
            save_set=attempt_set
            save_transfer=attempt_transfer
            
            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i+1])
        else:
            div_tran.append(sorted_transferability_metrics[i+1])
            div_set.append(new_model)
            div_output.append(sorted_output[i+1])
    transfer_model_set=copy.deepcopy(save_set)
    print('transfer set:',transfer_model_set)
