import argparse
from cmath import exp
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch

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

from scipy.io import loadmat

from transfer_metrics import LogME,NCE,LEEP,NCE_ours_addpz
logme = LogME(regression=False)


def obtain_label(all_output, all_label, all_fea, args):


    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
   
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

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    
    print(log_str+'\n')

    return pred_label.astype('int')


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
              

def my_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    domain_network=LogisticRegressionModel(feature_dim=all_fea.shape[1])

    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label)
    pred=nn.Softmax(dim=1)(all_output)
    prob, predict = torch.max(pred, 1)
  
    d1=torch.nonzero(prob>=tau).squeeze().cuda().numpy().tolist()
    d2=torch.nonzero(prob<tau).squeeze().cuda().numpy().tolist()
    split_dataset=Splited_List(all_fea,all_output,pse_label,d1)
    split_dataloader_noshuffle=DataLoader(split_dataset,args.batch_size,shuffle=False,drop_last=False)
    split_dataloader=DataLoader(split_dataset,args.batch_size,shuffle=True,drop_last=False)
    weights=get_reweighted_weights(split_dataloader,split_dataloader_noshuffle,domain_network,10)

    num_d1=len(d1)
    num_d2=all_output.shape[0]-num_d1
    all_fea1=all_fea[d1]
    all_output1=all_output[d1]
    weights1=weights[d1]
    pse_label1=pse_label[d1]

    all_output2=all_output[d2]
   
    pse_label2=pse_label[d2]
    with torch.no_grad():
        loss_d1=nn.CrossEntropyLoss()(all_output1,pse_label1)
        loss_d2=nn.CrossEntropyLoss(reduction='none')(all_output1,pse_label1)

        loss_d2_ori=nn.CrossEntropyLoss()(all_output2,pse_label2)
        loss_d2 =(loss_d2 *num_d1/num_d2* weights1 / (1-weights1)).mean() +loss_d2_ori

        loss=(loss_d1*num_d1+loss_d2*num_d2)/(num_d1+num_d2)
    loss=loss.item()

    return loss,pred_acc


def ce_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label).long().cuda()
    all_output=all_output.cuda()
    with torch.no_grad():
        loss=nn.CrossEntropyLoss()(all_output,pse_label)
       
    loss=loss.item()

    return loss,pred_acc

def ce_nce_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label).long()
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
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cuda().long(), 1)
    return targets

def KL_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label)
    pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    with torch.no_grad():
        loss=nn.KLDivLoss()(all_output.log(),pse_label)
       
    loss=loss.item()

    return loss,pred_acc

def LogME_true_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
 
    logme = LogME(regression=False)
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    all_fea=np.array(all_fea)
    all_label=np.array(all_label)
    loss=logme.fit(all_fea,all_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc


def LogME_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
 
    logme = LogME(regression=False)
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label)
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_fea=np.array(all_fea)
    pse_label=np.array(pse_label)
    loss=logme.fit(all_fea,pse_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc


def Leep_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
 
   
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    pse_label=obtain_label(all_output, all_label, all_fea, args)
    pse_label=torch.from_numpy(pse_label)
    #pse_label=get_one_hot(all_output,pse_label)
    all_output=nn.Softmax(dim=1)(all_output)
    #with torch.no_grad():
    all_output=np.array(all_output)
    pse_label=np.array(pse_label)
    
    loss=LEEP(all_output,pse_label)
    #print(loss)
    #loss=loss.item()

    return loss,pred_acc



def NCE_pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
 
   
    # f has shape of [N, D], y has shape [N]
  
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])

    
    pse_label=obtain_label(all_output, all_label, all_fea, args)
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

def entropy_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred=nn.Softmax(dim=1)(all_output).cuda()

    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    loss=torch.sum(entropy_loss(pred))/(pred.shape[0])

    return loss,pred_acc


def MI_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred=nn.Softmax(dim=1)(all_output).cuda()
    softmax_out = pred
    entropy_loss = torch.mean(loss_function.Entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
    entropy_loss -= gentropy_loss
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])
    #loss=torch.sum(entropy_loss(pred))/(pred.shape[0])

    return entropy_loss,pred_acc

def pse_principle(all_output, all_label, all_fea, args,tau=0.5):
    all_output=torch.from_numpy(all_output)
    all_label=torch.from_numpy(all_label)
    #pdb.set_trace()
    all_fea=torch.from_numpy(all_fea)
    _, predict = torch.max(all_output, 1)
    pred_acc=torch.sum(predict==all_label)/float(all_output.size()[0])


    pse_label=obtain_label(all_output, all_label, all_fea, args)


    return pse_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAiDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office31', choices=['office31', 'office-home', 'domainnet'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='vit_l_16', help="resnet50,vit_b_16,clip_ViT-B/32")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='ckps/vit_new/')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--fix', type=bool, default=False)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--in_path', type=str, default="/home/rzhe/MMDA/txt/domainnet_target_1.txt")
    parser.add_argument('--out_path', type=str, default="./test/domainnet_target_1.txt")
    parser.add_argument('--method', type=str, default="entropy")

    args = parser.parse_args()

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
        return feature,output,label

    #text_file_path = "/home/rzhe/MMDA/txt/domainnet_target_1.txt" 

    feature_path= "/data/liruizhe/office31_result/feature/"
    output_filename= "/data/liruizhe/office31_result/feature_pse/"

    if not os.path.exists(output_filename):
        os.makedirs(output_filename)
    #output_filename = f"./test/domainnet_target_1.txt"
    method = args.method
    for text_file_path in os.listdir(feature_path):
        if os.path.exists(output_filename+text_file_path):
            continue
        feature,output,label=load_mat(feature_path+text_file_path)

        pse=pse_principle(output,label,feature,args)
        import scipy
        scipy.io.savemat(output_filename+text_file_path,{'ft':feature,'output':output,'label':label,"pse":pse})