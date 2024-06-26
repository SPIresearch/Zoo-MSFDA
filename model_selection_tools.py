import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR

import loss_function
from utils import LEEP, LogME, SC_cal, calculate_a_distance, hsic_gam, mmd_rbf

logme = LogME(regression=False)


def obtain_label_cpu(all_output, all_label, all_fea, args):
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)

    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    if args.distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    return pred_label.astype("int")


def snd_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)

    pred = nn.Softmax(dim=1)(all_output).cuda()

    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)

    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    if args.distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = pred.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    dd = torch.from_numpy(dd)
    dd = nn.Softmax(dim=1)(dd / 0.05)

    loss = torch.sum(Entropy(dd)) / (dd.shape[0])
    loss = loss
    return loss, pred_acc


class LogisticRegressionModel(nn.Module):
    def __init__(self, feature_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(feature_dim, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return torch.sigmoid(x)


def get_reweighted_weights(data, data_no_shuffle, model, epoches=3):
    criterion = nn.BCELoss()
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    model.train()

    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(epoches):
        scheduler.step()
        for input, label in data:

            input, label = input.cuda(), label.cuda()
            outputs = model(input)
            label = label.float().unsqueeze(1)

            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        start_test = True
        for input, label in data_no_shuffle:

            input, label = input.cuda(), label.cuda()
            label = label.float().unsqueeze(1)
            outputs = model(input)
            loss = criterion(outputs, label)

            if start_test:
                all_output = outputs.float().cuda()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cuda()), 0)

    return all_output


def ce_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_output = all_output.cuda()
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(all_output, struc_pse_label)

    loss = loss.item()

    return loss, pred_acc


def ce_nce_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    struc_pse_label1 = get_one_hot(all_output, struc_pse_label)
    predict1 = get_one_hot(all_output, predict)

    with torch.no_grad():
        loss1 = nn.CrossEntropyLoss(reduction="none")(all_output, struc_pse_label)
        loss2 = nn.CrossEntropyLoss(reduction="none")(predict1, struc_pse_label1)

        loss = torch.sum(loss2) / (predict1.shape[0])
    loss = loss.item()

    return loss, pred_acc


def get_one_hot(inputs, targets):
    log_probs = torch.log(inputs)
    targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).long(), 1)
    return targets


def KL_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    struc_pse_label = get_one_hot(all_output, struc_pse_label)
    all_output = nn.Softmax(dim=1)(all_output)
    with torch.no_grad():
        loss = nn.KLDivLoss()(all_output.log(), struc_pse_label)

    loss = loss.item()

    return loss, pred_acc


def LogME_true_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass

    logme = LogME(regression=False)

    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])
    all_fea = np.array(all_fea)
    all_label = np.array(all_label)
    loss = logme.fit(all_fea, all_label)

    return loss, pred_acc


def LogME_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    logme = LogME(regression=False)

    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)

    all_fea = np.array(all_fea)
    struc_pse_label = np.array(struc_pse_label)
    loss = logme.fit(all_fea, struc_pse_label)

    return loss, pred_acc


def Leep_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)

    all_output = np.array(all_output)
    struc_pse_label = np.array(struc_pse_label)

    loss = LEEP(all_output, struc_pse_label)

    return loss, pred_acc


def NCE_pse_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    struc_pse_label = torch.from_numpy(struc_pse_label)

    all_output = nn.Softmax(dim=1)(all_output)

    all_output = np.array(all_output)
    struc_pse_label = np.array(struc_pse_label)
    predict = np.array(predict)
    loss = SC(predict, struc_pse_label)

    return loss, pred_acc


def Entropy(p):

    epsilon = 1e-5
    entropy = -p * torch.log(p + epsilon)

    loss = torch.sum(entropy, dim=1)
    return loss


def entropy_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred = nn.Softmax(dim=1)(all_output).cuda()

    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])
    loss = torch.sum(Entropy(pred)) / (pred.shape[0])
    loss = -loss
    return loss, pred_acc


def MI_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred = nn.Softmax(dim=1)(all_output).cuda()
    softmax_out = pred
    Entropy = torch.mean(loss_function.Entropy(softmax_out))
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
    Entropy -= gentropy_loss
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    Entropy = -Entropy
    return Entropy, pred_acc


def MDE_principle(all_output, all_label, all_fea, struc_pse_label, args, T=1):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)

    energy = -T * (torch.logsumexp(all_output / T, dim=1))

    avg_energies = torch.log_softmax(energy, dim=0).mean()
    avg_energies = torch.log(-avg_energies).item()

    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    return avg_energies, pred_acc


def ce_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    struc_pse_label = predict
    all_output = all_output.cuda()
    struc_pse_label = struc_pse_label.cuda()
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(all_output, struc_pse_label)

    loss = loss.item()

    return loss, pred_acc


def Leep_true_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)

    all_output = np.array(all_output)
    all_label = np.array(all_label.long())

    loss = LEEP(all_output, all_label)

    return loss, pred_acc


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def MI(pred):

    softmax_out = pred
    Entropy = torch.mean(Entropy(softmax_out))

    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    return Entropy, gentropy_loss


def MI2(pred):

    softmax_out = pred
    Entropy = torch.mean(Entropy(softmax_out))
    _, pred = torch.max(softmax_out, 1)
    softmax_out = torch.eye(65)[pred]
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    return Entropy, gentropy_loss


def SUTE_principle(all_output, all_label, all_fea, struc_pse_label, args):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred = nn.Softmax(dim=1)(all_output)

    Entropy, GD = MI2(pred)

    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])

    all_label = np.array(all_label)
    struc_pse_label = np.array(struc_pse_label)
    predict = np.array(predict)

    SC = SC_cal(predict, struc_pse_label)
    IC = -Entropy

    return SC, IC, GD, pred_acc


def calculate_entropy(n, c):

    probability = 1 / c

    log_term = -torch.log(torch.tensor(probability))

    entropy = n * probability * log_term
    return entropy.item()


def read_text_file(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"文件 '{file_path}' 未找到。")


def load_mat(file):
    result = loadmat(file)
    feature = result["ft"]
    label = result["label"][0]
    output = result["output"]
    pse = result["pse"][0]

    return feature, output, label, pse


def mmd_principle(
    all_output, all_label, all_fea, struc_pse_label, args, feature_source
):

    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
        feature_source = torch.from_numpy(feature_source)
    except:
        pass

    all_fea = torch.nn.AdaptiveAvgPool1d(1000)(all_fea)
    feature_source = torch.nn.AdaptiveAvgPool1d(1000)(feature_source)
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])
    min_samples = min(all_fea.shape[0], feature_source.shape[0]) // 10 + 1
    random_indices = torch.randperm(min_samples)

    all_fea = all_fea[random_indices]
    feature_source = feature_source[random_indices]

    mmd = mmd_rbf(all_fea, feature_source)
    mmd = -mmd
    return mmd, pred_acc


def A_distance_principle(
    all_output, all_label, all_fea, struc_pse_label, args, feature_source
):
    try:
        all_output = torch.from_numpy(all_output)
        all_label = torch.from_numpy(all_label)
        all_fea = torch.from_numpy(all_fea)
        struc_pse_label = torch.from_numpy(struc_pse_label).long()
        feature_source = torch.from_numpy(feature_source).long()
    except:
        pass
    _, predict = torch.max(all_output, 1)
    pred_acc = torch.sum(predict == all_label) / float(all_output.size()[0])
    min_samples = min(all_fea.shape[0], feature_source.shape[0])
    random_indices = torch.randperm(min_samples)

    all_fea = all_fea[random_indices]
    feature_source = feature_source[random_indices]
    a_dis = calculate_a_distance(all_fea, feature_source)
    a_dis = -a_dis
    return a_dis, pred_acc


def transfer_calcualte_for_a_model(
    method, output, label, feature, pse, args, feature_source=0
):

    if method == "ANE":
        transfer_ability, pred_acc = entropy_principle(
            output, label, feature, pse, args
        )
    if method == "SND":
        transfer_ability, pred_acc = snd_principle(output, label, feature, pse, args)
    elif method == "NMI":
        transfer_ability, pred_acc = MI_principle(output, label, feature, pse, args)
    elif method == "LogME_pse":
        transfer_ability, pred_acc = LogME_pse_principle(
            output, label, feature, pse, args
        )
    elif method == "LEEP_pse":
        transfer_ability, pred_acc = Leep_pse_principle(
            output, label, feature, pse, args
        )
    elif method == "LogME_true":
        transfer_ability, pred_acc = LogME_true_principle(
            output, label, feature, pse, args
        )
    elif method == "LEEP_true":
        transfer_ability, pred_acc = Leep_true_principle(
            output, label, feature, pse, args
        )
    elif method == "MMD":
        transfer_ability, pred_acc = mmd_principle(
            output, label, feature, pse, args, feature_source
        )

    elif method == "MDE":
        transfer_ability, pred_acc = MDE_principle(
            output, label, feature, pse, args, T=1
        )

    elif method == "A_distance":
        transfer_ability, pred_acc = A_distance_principle(
            output, label, feature, pse, args, feature_source
        )

    elif method == "SUTE":
        SC, IC, GD, pred_acc = SUTE_principle(output, label, feature, pse, args)

        tau_h = 2 / 3
        tau_l = 1 / 2
        if GD > calculate_entropy(args.class_num, args.class_num) * tau_h:
            GD = calculate_entropy(args.class_num, args.class_num) * tau_h

        transfer_ability = 10 * SC + 0.1 * IC + 1 * GD
        if GD < calculate_entropy(args.class_num, args.class_num) * tau_l:
            GD = -float("inf")
            transfer_ability = -float("inf")
        if SC > -1e-10:
            transfer_ability = -float("inf")
    try:
        pred_acc = pred_acc.item()
        transfer_ability = transfer_ability.item()
    except:
        pass

    return transfer_ability, pred_acc


def draw_histo_ce(
    model_config_file, feature_path, transferability_output_filename, method, args
):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set = []
    pred_acc_set = []
    pred_acc_set_all = []
    all_output = []

    if file_contents:
        import matplotlib.pyplot as plt

        with open(transferability_output_filename, "w") as output_file:
            for line in file_contents:
                source = line.split(".mat")[0][-3]
                source_line = line[:-5] + source + ".mat"

                feature, output, label, pse = load_mat(feature_path + line)
                outputs = torch.from_numpy(output)
                labels = torch.from_numpy(label).long()
                losses = F.cross_entropy(outputs, labels, reduction="none")
                plt.figure()

                plt.hist(losses.numpy(), bins=20)
                plt.xlabel("Cross-Entropy Loss")
                plt.ylabel("Frequency")

                plt.savefig(f'./fig/{line.split(".")[0]}.png')

    print(
        "Transferability file is saved in {}.".format(transferability_output_filename)
    )

    print("Mean Accuracy", sum(pred_acc_set_all) / len(pred_acc_set_all))

    a = pd.Series(loss_set)
    b = pd.Series(pred_acc_set)
    print("Spearman", stats.spearmanr(a, b))

    return all_output


def transfer_calcualte_for_individual_models(
    model_config_file, feature_path, transferability_output_filename, method, args
):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set = []
    pred_acc_set = []
    pred_acc_set_all = []
    all_output = []

    if file_contents:

        with open(transferability_output_filename, "w") as output_file:
            for line in file_contents:
                source = line.split(".mat")[0][-3]
                source_line = line[:-5] + source + ".mat"

                feature, output, label, pse = load_mat(feature_path + line)
                feature_source, _, _, _ = load_mat(feature_path + line)
                transfer_ability, pred_acc = transfer_calcualte_for_a_model(
                    method, output, label, feature, pse, args, feature_source
                )

                pred_acc_set_all.append(pred_acc)

                if transfer_ability < -100:

                    continue
                all_output.append(output)
                output_file.write(
                    line + " " + str(transfer_ability) + " " + str(pred_acc) + "\n"
                )
                loss_set.append(transfer_ability)
                pred_acc_set.append(pred_acc)

    print(
        "Transferability file is saved in {}.".format(transferability_output_filename)
    )

    print("Mean Accuracy", sum(pred_acc_set_all) / len(pred_acc_set_all))

    a = pd.Series(loss_set)
    b = pd.Series(pred_acc_set)
    print("Spearman", stats.spearmanr(a, b))

    return all_output


def transfer_calcualte_for_individual_models2(
    model_config_file, feature_path, transferability_output_filename, method, args
):
    print(model_config_file)
    print(transferability_output_filename)

    file_contents = read_text_file(model_config_file)
    loss_set = []
    pred_acc_set = []
    pred_acc_set_all = []
    all_output = []

    if file_contents:

        with open(transferability_output_filename, "w") as output_file:
            for line in file_contents:
                source = line.split(".mat")[0][-3]
                source_line = line[:-5] + source + ".mat"

                feature, output, label, pse = load_mat(feature_path + line)
                feature_source, _, _, _ = load_mat(feature_path + line)
                transfer_ability, pred_acc = transfer_calcualte_for_a_model(
                    method, output, label, feature, pse, args, feature_source
                )
                print(line, transfer_ability, pred_acc)
                pred_acc_set_all.append(pred_acc)

                if transfer_ability < -100:

                    continue
                all_output.append(output)
                output_file.write(
                    line + " " + str(transfer_ability) + " " + str(pred_acc) + "\n"
                )
                loss_set.append(transfer_ability)
                pred_acc_set.append(pred_acc)

    print(
        "Transferability file is saved in {}.".format(transferability_output_filename)
    )

    print("Mean Accuracy", sum(pred_acc_set_all) / len(pred_acc_set_all))

    a = pd.Series(loss_set)
    b = pd.Series(pred_acc_set)
    print("Spearman", stats.spearmanr(a, b))

    return all_output, pred_acc_set


def print_acc_of_each_model(
    model_config_file, feature_path, transferability_output_filename, args
):
    print(model_config_file)
    print(transferability_output_filename)
    method = "entropy"
    file_contents = read_text_file(model_config_file)
    loss_set = []
    pred_acc_set = []
    pred_acc_set_all = []
    all_output = []

    if file_contents:
        with open(transferability_output_filename, "w") as output_file:
            for line in file_contents:
                source = line.split(".mat")[0][-3]
                source_line = line[:-5] + source + ".mat"

                feature, output, label, pse = load_mat(feature_path + line)
                feature_source, _, _, _ = load_mat(feature_path + source_line)
                transfer_ability, pred_acc = transfer_calcualte_for_a_model(
                    method, output, label, feature, pse, args, feature_source
                )
                print(line, "Acc:", pred_acc)
                pred_acc_set_all.append(pred_acc)

                all_output.append(output)
                output_file.write(line + " Acc:" + str(pred_acc) + "\n")

    return 1


def read_transferability_text_file(file_path):
    model_names = []
    transferability_metrics = []
    accuracies = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:

            columns = line.strip().split()

            model_name = columns[0]
            transferability_metric = float(columns[1])
            accuracy = float(columns[2])

            model_names.append(model_name)
            transferability_metrics.append(transferability_metric)
            accuracies.append(accuracy)
    return model_names, transferability_metrics, accuracies


def sort_lists_by_transferability(model_names, transferability_metrics, all_output):

    zipped_data = zip(transferability_metrics, model_names, all_output)

    sorted_data = sorted(zipped_data, key=lambda x: x[0], reverse=True)

    sorted_transferability_metrics, sorted_model_names, sorted_output = zip(
        *sorted_data
    )

    return sorted_model_names, sorted_transferability_metrics, sorted_output


def sort_lists_by_div(model_names, div_metrics, div_tran):

    zipped_data = zip(div_metrics, model_names, div_tran)

    sorted_data = sorted(zipped_data, key=lambda x: x[0], reverse=True)

    try:
        sorted_div_metrics, sorted_model_names, div_tran = zip(*sorted_data)
    except:
        sorted_div_metrics, sorted_model_names, div_tran = [], [], []
    return sorted_model_names, sorted_div_metrics, div_tran


def cal_div(feature_path, model_set, model):
    div = 0

    feature, _, _, _ = load_mat(feature_path + model)
    for new_model in model_set:

        z = min(1000, feature.shape[0])
        random_indices = np.random.choice(feature.shape[0], z, replace=False)
        new_feature, new_output, label, _ = load_mat(feature_path + new_model)
        div += hsic_gam(new_feature[random_indices], feature[random_indices])[0]
    return div


def transfer_measure_for_ensemble(feature_path, model_set, method, args):

    for new_model in model_set:

        new_feature, new_output, label, _ = load_mat(feature_path + new_model)
        z = (new_feature.shape[0]) // 1
        lists = [i * 1 for i in range(z)]
        new_feature = new_feature[lists]
        new_output = new_output[lists]
        label = label[lists]

        new_feature = torch.from_numpy(new_feature)
        new_output = torch.from_numpy(new_output)
        label = torch.from_numpy(label)

        try:
            all_feature = torch.cat([all_feature, new_feature], 1)
            all_output = all_output + new_output
        except:
            all_output = new_output
            all_feature = new_feature
    _, predict = torch.max(all_output, 1)

    pse = obtain_label_cpu(all_output, label, all_feature, args)
    pse = torch.from_numpy(pse)

    transfer_ability, acc = transfer_calcualte_for_a_model(
        method, all_output, label, all_feature, pse, args
    )

    return transfer_ability, acc


def transfer_pse_for_ensemble(feature_path, attempt_tran, attempt_set, args):
    attempt_tran = np.array(attempt_tran)
    attempt_tran = torch.from_numpy(attempt_tran)
    attempt_tran = torch.nn.Softmax(0)(attempt_tran)
    for new_model in attempt_set:

        _, new_output, label, _ = load_mat(feature_path + new_model)

        new_output = torch.from_numpy(new_output)
        label = torch.from_numpy(label)

        try:

            all_output = all_output + new_output
        except:
            all_output = new_output

    _, predict = torch.max(all_output, 1)

    pred_acc = torch.sum(predict == label) / float(predict.size()[0])

    return predict, label, pred_acc


def model_selection_function2(
    K, feature_path, transferability_file_path, all_output, method, args
):
    model_names, transferability_metrics, accuracies = read_transferability_text_file(
        transferability_file_path
    )

    sorted_model_names, sorted_transferability_metrics, sorted_output = (
        sort_lists_by_transferability(model_names, transferability_metrics, all_output)
    )

    save_set = sorted_model_names[:K]
    Tran_set = sorted_transferability_metrics[:K]
    Tran_output = [
        sorted_output[0],
    ]
    print("selected models", save_set)

    attempt_tran = Tran_set
    attempt_set = copy.deepcopy(save_set)
    attempt_transfer, acc = transfer_measure_for_ensemble(
        feature_path, attempt_set, method, args
    )
    print("Aggregation performance:", acc)

    pse_tran, label, acc = transfer_pse_for_ensemble(
        feature_path, attempt_tran, attempt_set, args
    )

    print("Aggregation pse performance:", acc)


def model_selection_function(
    feature_path, transferability_file_path, all_output, method, args
):
    model_names, transferability_metrics, accuracies = read_transferability_text_file(
        transferability_file_path
    )

    sorted_model_names, sorted_transferability_metrics, sorted_output = (
        sort_lists_by_transferability(model_names, transferability_metrics, all_output)
    )

    Initial = sorted_model_names[0]
    save_set = [
        Initial,
    ]
    Tran_set = [
        sorted_transferability_metrics[0],
    ]
    Tran_output = [
        sorted_output[0],
    ]

    div_set = []
    div_output = []
    div_tran = []
    save_transfer = sorted_transferability_metrics[0]
    for i, new_model in enumerate(sorted_model_names[1:]):

        new_model_trans = sorted_transferability_metrics[i + 1]
        attempt_tran = Tran_set
        attempt_set = copy.deepcopy(save_set)
        attempt_set.append(new_model)

        attempt_transfer, acc = transfer_measure_for_ensemble(
            feature_path, attempt_set, method, args
        )
        pse_tran, label, acc = transfer_pse_for_ensemble(
            feature_path, attempt_tran, attempt_set, args
        )
        if attempt_transfer > save_transfer:

            save_set = attempt_set
            save_transfer = attempt_transfer

            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i + 1])
        else:
            div_tran.append(sorted_transferability_metrics[i + 1])
            div_set.append(new_model)
            div_output.append(sorted_output[i + 1])
    transfer_model_set = copy.deepcopy(save_set)

    div_metrics = []
    for i, model in enumerate(div_set):

        div_metrics.append(cal_div(feature_path, transfer_model_set, model))
    div_set, sorted_div, sorted_div_tran = sort_lists_by_div(
        model_names, div_metrics, div_tran
    )

    return (
        transfer_model_set,
        div_set,
        attempt_tran,
        div_tran,
        sorted_div_tran,
        Tran_output,
        div_output,
        sorted_model_names,
        pse_tran,
        label,
    )


def model_selection_function_for_mde(
    feature_path, transferability_file_path, all_output, method, args
):
    model_names, transferability_metrics, accuracies = read_transferability_text_file(
        transferability_file_path
    )

    sorted_model_names, sorted_transferability_metrics, sorted_output = (
        sort_lists_by_transferability(model_names, transferability_metrics, all_output)
    )

    Initial = sorted_model_names[0]
    save_set = [
        Initial,
    ]
    Tran_set = [
        sorted_transferability_metrics[0],
    ]
    Tran_output = [
        sorted_output[0],
    ]
    print("Initial model", Initial)

    div_set = []
    div_output = []
    div_tran = []
    save_transfer = sorted_transferability_metrics[0]
    for i, new_model in enumerate(sorted_model_names[1:]):
        if i >= 10:
            continue
        new_model_trans = sorted_transferability_metrics[i + 1]
        attempt_tran = Tran_set
        attempt_set = copy.deepcopy(save_set)
        attempt_set.append(new_model)

        print("add model", new_model)
        save_set = attempt_set

        attempt_tran.append(new_model_trans)
        Tran_output.append(sorted_output[i + 1])
    transfer_model_set = copy.deepcopy(save_set)
    print("transfer set:", transfer_model_set)
    attempt_transfer, acc = transfer_measure_for_ensemble(
        feature_path, attempt_set, method, args
    )
    pse_tran, label, acc = transfer_pse_for_ensemble(
        feature_path, attempt_tran, attempt_set, args
    )

    div_set, sorted_div_tran = model_names, div_tran

    return (
        transfer_model_set,
        div_set,
        attempt_tran,
        div_tran,
        sorted_div_tran,
        Tran_output,
        div_output,
        sorted_model_names,
        pse_tran,
        label,
    )


def model_selection_function_for_premethod(
    feature_path, transferability_file_path, all_output, method, args
):
    model_names, transferability_metrics, accuracies = read_transferability_text_file(
        transferability_file_path
    )

    sorted_model_names, sorted_transferability_metrics, sorted_output = (
        sort_lists_by_transferability(model_names, transferability_metrics, all_output)
    )

    Initial = sorted_model_names[0]
    save_set = [
        Initial,
    ]
    Tran_set = [
        sorted_transferability_metrics[0],
    ]
    Tran_output = [
        sorted_output[0],
    ]
    print("Initial model", Initial)

    div_set = []
    div_output = []
    div_tran = []
    save_transfer = sorted_transferability_metrics[0]
    for i, new_model in enumerate(sorted_model_names[1:]):
        new_model_trans = sorted_transferability_metrics[i + 1]
        attempt_tran = Tran_set
        attempt_set = copy.deepcopy(save_set)
        attempt_set.append(new_model)

        attempt_transfer, acc = transfer_measure_for_ensemble(
            feature_path, attempt_set, method, args
        )
        pse_tran, label, acc = transfer_pse_for_ensemble(
            feature_path, attempt_tran, attempt_set, args
        )
        if attempt_transfer > save_transfer:
            print("add model", new_model, "performance:", acc)
            save_set = attempt_set
            save_transfer = attempt_transfer

            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i + 1])
        else:
            div_tran.append(sorted_transferability_metrics[i + 1])
            div_set.append(new_model)
            div_output.append(sorted_output[i + 1])
    transfer_model_set = copy.deepcopy(save_set)
    print("transfer set:", transfer_model_set)

    div_set, sorted_div_tran = model_names, div_tran

    return (
        transfer_model_set,
        div_set,
        attempt_tran,
        div_tran,
        sorted_div_tran,
        Tran_output,
        div_output,
        sorted_model_names,
        pse_tran,
        label,
    )


def model_aggregating_performance(
    feature_path, transferability_file_path, all_output, method, args
):
    model_names, transferability_metrics, accuracies = read_transferability_text_file(
        transferability_file_path
    )

    sorted_model_names, sorted_transferability_metrics, sorted_output = (
        sort_lists_by_transferability(model_names, transferability_metrics, all_output)
    )

    Initial = sorted_model_names[0]
    save_set = [
        Initial,
    ]
    Tran_set = [
        sorted_transferability_metrics[0],
    ]
    Tran_output = [
        sorted_output[0],
    ]
    print("Initial model", Initial)

    div_set = []
    div_output = []
    div_tran = []
    save_transfer = sorted_transferability_metrics[0]
    for i, new_model in enumerate(sorted_model_names[1:]):

        new_model_trans = sorted_transferability_metrics[i + 1]
        attempt_tran = Tran_set
        attempt_set = copy.deepcopy(save_set)
        attempt_set.append(new_model)

        attempt_transfer, acc = transfer_measure_for_ensemble(
            feature_path, attempt_set, method, args
        )
        pse_tran, label, acc = transfer_pse_for_ensemble(
            feature_path, attempt_tran, attempt_set, args
        )
        if attempt_transfer > save_transfer:
            print("add model", new_model, "performance:", acc)
            save_set = attempt_set
            save_transfer = attempt_transfer

            attempt_tran.append(new_model_trans)
            Tran_output.append(sorted_output[i + 1])
        else:
            div_tran.append(sorted_transferability_metrics[i + 1])
            div_set.append(new_model)
            div_output.append(sorted_output[i + 1])
    transfer_model_set = copy.deepcopy(save_set)
    print("transfer set:", transfer_model_set)
