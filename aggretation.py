import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_function import *


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(
        2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
        - (high - low)
        + low
    )


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(torch.tensor(1.0) * init_weights)

    def forward(self, x):
        x = self.w * torch.ones((x[0].shape[0]), 1).cuda()
        x = torch.sigmoid(x)
        return x


def IM_loss(output):
    softmax_out = nn.Softmax(dim=1)(output)
    Entropy = torch.mean(Entropy(softmax_out))

    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    Entropy -= gentropy_loss
    return Entropy


class OursEnsemble(nn.Module):
    def __init__(
        self,
        netF_list,
        netC_list,
        classes,
        source,
        optimized_models,
        tran_tran,
        aug_set,
        no_update_set,
        model_features,
        model_outs=[],
    ):
        """
        :param classes: Number of classification categories
        :param source: Number of source
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(OursEnsemble, self).__init__()
        self.source = source
        self.classes = classes
        self.optimized_models = optimized_models
        self.source = source
        self.aug_set = aug_set
        self.classes = classes
        self.no_update_set = no_update_set
        self.model_features = model_features

        self.model_outs = model_outs

        self.netF = nn.ModuleList()
        self.netC = nn.ModuleList()
        self.loss = 0
        self.tran_tran = torch.tensor(
            tran_tran, dtype=torch.float32
        )  # nn.Parameter(3*torch.tensor(tran_tran, dtype=torch.float32))
        for i in range(self.source):
            self.netF.append(netF_list[i])
            self.netC.append(netC_list[i])

    def return_each_feature_output(self):
        return self.features_all, self.outputs_all

    def SIM(self):
        self.loss = 0
        self.tran = nn.Softmax(0)(self.tran_tran)
        for i in range(self.optimized_models):
            self.loss += IM_loss(self.outputs_all[i]) * self.tran[i]
        # self.loss+=IM_loss(self.outputs_all_w)
        return self.loss

    def forward(self, x, idx):
        self.tran = nn.Softmax(0)(self.tran_tran)
        self.outputs_all = torch.zeros(self.source, x[0].shape[0], self.classes).cuda()
        self.outputs_all_w = torch.zeros(x[0].shape[0], self.classes).cuda()
        self.features_all = {}
        for i in range(self.optimized_models):

            if i not in self.no_update_set:
                if self.aug_set[i] == 0:
                    data = x[0]
                else:
                    data = x[1]
                self.features_all[i] = self.netF[i](data)
            else:
                self.features_all[i] = self.model_features[i][idx].cuda()

            outputs = self.netC[i](self.features_all[i])

            self.outputs_all[i] = outputs
            self.outputs_all_w += outputs * self.tran[i]
            if i == 0:
                all_feature = self.features_all[i]
            else:
                all_feature = torch.cat([all_feature, self.features_all[i]], 1)
        # self.outputs_all_w/=self.source
        return all_feature, self.outputs_all_w

    def forward_mix(self, x, idx, mix_idx, l):
        self.tran = nn.Softmax(0)(self.tran_tran)
        self.outputs_all = torch.zeros(self.source, x[0].shape[0], self.classes).cuda()
        self.outputs_all_w = torch.zeros(x[0].shape[0], self.classes).cuda()
        self.features_all = {}
        for i in range(self.optimized_models):

            if i not in self.no_update_set:
                if self.aug_set[i] == 0:
                    data = x[0]
                else:
                    data = x[1]
                input_a, input_b = data, data[mix_idx]

                mixed_input = l * input_a + (1 - l) * input_b
                self.features_all[i] = self.netF[i](mixed_input)
            else:
                fea = self.model_features[i][idx].cuda()
                input_a, input_b = fea, fea[mix_idx]
                self.features_all[i] = (
                    l * input_a + (1 - l) * input_b
                )  # self.model_features[i][idx].cuda()

            outputs = self.netC[i](self.features_all[i])
            self.outputs_all[i] = outputs
            self.outputs_all_w += outputs * self.tran[i]
            if i == 0:
                all_feature = self.features_all[i]
            else:
                all_feature = torch.cat([all_feature, self.features_all[i]], 1)
        # self.outputs_all_w/=self.source
        return all_feature, self.outputs_all_w

    def forward_tran(self, x, idx):
        self.tran = nn.Softmax(0)(self.tran_tran)
        self.outputs_all = torch.zeros(self.source, x[0].shape[0], self.classes).cuda()
        self.outputs_all_w = torch.zeros(x[0].shape[0], self.classes).cuda()
        self.features_all = {}
        for i in range(self.optimized_models - 1):

            if i not in self.no_update_set:
                if self.aug_set[i] == 0:
                    data = x[0]
                else:
                    data = x[1]
                self.features_all[i] = self.netF[i](data)
            else:
                self.features_all[i] = self.model_features[i][idx].cuda()

            outputs = self.netC[i](self.features_all[i])
            self.outputs_all[i] = outputs
            self.outputs_all_w += outputs * self.tran[i]
            if i == 0:
                all_feature = self.features_all[i]
            else:
                all_feature = torch.cat([all_feature, self.features_all[i]], 1)
        # self.outputs_all_w/=self.source
        return all_feature, self.outputs_all_w
