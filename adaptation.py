import argparse
import os
import os.path as osp
import warnings
from cmath import exp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import loss_function
import network
from aggretation import OursEnsemble
from data_list import ImageList_idx_fast
from loss_function import CrossEntropy1
from model_selection_tools import *
from utils import *

warnings.filterwarnings("ignore")


def MI(output):
    softmax_out = nn.Softmax(dim=1)(output)
    Entropy = torch.mean(loss_function.Entropy(softmax_out))

    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

    Entropy -= gentropy_loss
    return Entropy


def calculate_loss(output, div_pse, div_mask):

    selected_loss = nn.CrossEntropyLoss(reduction="none")(output, div_pse)
    selected_loss = torch.mean(selected_loss * div_mask)
    loss = selected_loss
    return loss


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def data_load(args):

    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_idx_fast(
        txt_tar, transform1=image_train(256, 224), transform2=image_train(518, 518)
    )
    dset_loaders["target"] = DataLoader(
        dsets["target"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
        pin_memory=True,
    )
    dsets["test"] = ImageList_idx_fast(
        txt_test, transform1=image_test(256, 224), transform2=image_test(518, 518)
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=3 * train_bs,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
        pin_memory=True,
    )

    return dset_loaders


def cal_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            tar_idx = data[2]
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            _, outputs = model(inputs, tar_idx)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = (
        torch.mean(loss_function.Entropy(nn.Softmax(dim=1)(all_output)))
        .cpu()
        .data.item()
    )
    pred_label = np.array(predict)

    return accuracy * 100, mean_ent, pred_label.astype("int")


def construct_net(model_name):
    resize_size = 256
    crop_size = 224

    if model_name == "vit_l_16":
        netF = network.VitBase(res_name=model_name).cuda()
    elif model_name == "vit_b_16":
        netF = network.VitBase(res_name=model_name).cuda()
    elif model_name == "vit_b_32":
        netF = network.VitBase(res_name=model_name).cuda()
    elif model_name == "vit_h_14":
        netF = network.VitBase(res_name=model_name).cuda()
        resize_size = 518
        crop_size = 518
    elif model_name == "vit_l_32":
        netF = network.VitBase(res_name=model_name).cuda()
    elif model_name == "swin_t":

        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_l":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_b":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_s":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_v2_t":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_v2_s":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "swin_v2_b":
        netF = network.SwinBase(res_name=model_name).cuda()
    elif model_name == "vgg":
        netF = network.VGGBase(vgg_name=model_name).cuda()
    elif model_name[:8] == "resnet50":
        netF = network.ResBase(res_name="resnet50").cuda()
    elif model_name == "resnet50_v1":
        netF = network.ResBase(res_name="resnet50_v1").cuda()
    elif model_name == "resnet50_nopre":
        netF = network.ResBase(res_name="resnet50_nopre").cuda()
    elif model_name == "resnet101":
        netF = network.ResBase(res_name="resnet101").cuda()
    elif model_name == "densenet161":
        netF = network.DENSEBase(res_name="densenet161").cuda()
    elif model_name == "densenet201":
        netF = network.DENSEBase(res_name="densenet201").cuda()
    elif model_name == "mobilenet_v3_small":
        netF = network.Mobv3Base(res_name="mobilenet_v3_small").cuda()
    elif model_name == "mobilenet_v3_large":
        netF = network.Mobv3Base(res_name="mobilenet_v3_large").cuda()
    elif model_name == "efficientnet_v2_s":

        netF = network.EffBase(res_name="efficientnet_v2_s").cuda()
    elif model_name == "efficientnet_v2_m":
        netF = network.EffBase(res_name="efficientnet_v2_m").cuda()
    elif model_name == "efficientnet_v2_l":
        netF = network.EffBase(res_name="efficientnet_v2_l").cuda()

    netC = network.feat_classifier(
        type=args.layer,
        feature_dim=netF.in_features,
        class_num=args.class_num,
        bottleneck_dim=args.bottleneck,
    ).cuda()
    if resize_size == 256:
        aug_set = 0
    else:
        aug_set = 1
    return netF, netC, aug_set


def model_initial(
    net, update_F_list, update_C_list, source, optimized_models, path_list
):
    param_group = []
    exclude_names = ["netF", "netC"]

    for k, v in net.named_parameters():
        v.requires_grad = False
    for i in range(optimized_models):
        modelpath = path_list[i] + "/source_F.pt"

        net.netF[i].load_state_dict(torch.load(modelpath))
        net.netF[i].eval()
        for k, v in net.netF[i].named_parameters():
            if i in update_F_list:
                v.requires_grad = True
                param_group += [{"params": v, "lr": args.lr * 0.1}]
            else:
                v.requires_grad = False

        modelpath = path_list[i] + "/source_C.pt"

        net.netC[i].load_state_dict(torch.load(modelpath))
        net.netC[i].eval()
        for k, v in net.netC[i].named_parameters():
            if i in update_C_list:
                v.requires_grad = True
                param_group += [{"params": v, "lr": args.lr}]
            else:
                v.requires_grad = False

    for name, param in net.named_parameters():
        if all(exclude_name not in name for exclude_name in exclude_names):

            v.requires_grad = True
            param_group += [{"params": param, "lr": args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return net, optimizer


def model_initial_tc(
    net, update_F_list, update_C_list, source, optimized_models, path_list
):
    param_group = []
    exclude_names = ["netF", "netC"]

    for k, v in net.named_parameters():
        v.requires_grad = False

    for i in range(optimized_models):
        modelpath = path_list[i] + "/source_F.pt"

        net.netF[i].load_state_dict(torch.load(modelpath))
        net.netF[i].eval()

        modelpath = path_list[i] + "/source_C.pt"

        net.netC[i].load_state_dict(torch.load(modelpath))
        net.netC[i].eval()
        for k, v in net.netC[i].named_parameters():
            if i in update_C_list:
                v.requires_grad = True
                param_group += [{"params": v, "lr": args.lr}]
            else:
                v.requires_grad = False

    for name, param in net.named_parameters():
        if all(exclude_name not in name for exclude_name in exclude_names):

            v.requires_grad = True
            param_group += [{"params": param, "lr": args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return net, optimizer


def train_target(
    args, model, optimizer, tran_tran, selected_pred_class, mask, pse_tran
):
    dset_loaders = data_load(args)
    selected_pred_class = selected_pred_class.squeeze()
    model_tran = nn.Softmax(0)(3 * tran_tran)

    print("begin transfer_weights:", model_tran)

    model = model.cuda()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    model.train()

    for i in range(model.source):
        model.netF[i].eval()
        model.netC[i].train()
        if i >= model.optimized_models:
            model.netC[i].eval()
            model.netF[i].eval()

    while iter_num < max_iter:

        if iter_num == 0:
            model.eval()

            mem_label = pse_tran
            try:
                mem_label = torch.from_numpy(mem_label)
            except:
                pass
            model.train()
            print("transfer_weights:", model.tran_tran.data)

            for i in range(model.source):
                model.netF[i].eval()
                model.netC[i].train()
                if i >= model.optimized_models:
                    model.netC[i].eval()
                    model.netF[i].eval()

        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test[0].size(0) == 1:
            continue

        inputs_test[0] = inputs_test[0].cuda()
        inputs_test[1] = inputs_test[1].cuda()

        pse = mem_label[tar_idx]
        pse = pse.cuda()

        div_pse = selected_pred_class[tar_idx].cuda()
        div_mask = mask[tar_idx].cuda()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test, outputs_test = model(inputs_test, tar_idx)
        softmax_out = nn.Softmax(dim=1)(outputs_test)
        self_loss = model.SIM()
        loss1 = calculate_loss(outputs_test, div_pse, div_mask)
        loss = loss1 + self_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        try:
            inputs_test, _, tar_idx = next(iter_test1)
        except:
            iter_test1 = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test1)

        if inputs_test[0].size(0) == 1:
            continue

        inputs_test[0] = inputs_test[0].cuda()
        inputs_test[1] = inputs_test[1].cuda()

        pse = mem_label[tar_idx]
        pse = pse.cuda()

        div_pse = selected_pred_class[tar_idx].cuda()
        div_mask = mask[tar_idx].cuda()

        all_targets = torch.nn.functional.one_hot(pse, num_classes=args.class_num)

        args.alpha = 0.75
        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)

        mix_idx = torch.randperm(inputs_test[0].size(0))

        target_a, target_b = all_targets, all_targets[mix_idx]

        mixed_target = l * target_a + (1 - l) * target_b

        features_test, outputs_test = model.forward_mix(
            inputs_test, tar_idx, mix_idx, l
        )

        classifier_loss = CrossEntropy1()(outputs_test, mixed_target)
        loss = classifier_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_num += 1

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()

            acc_s_te, _, mem_label = cal_acc(dset_loaders["test"], model, False)
            mem_label = torch.from_numpy(mem_label)
            log_str = "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                args.name, iter_num, max_iter, acc_s_te
            )

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            torch.save(model.state_dict(), osp.join(args.output_dir, "model" + ".pt"))
            model.train()

            for i in range(model.source):
                model.netF[i].eval()
                model.netC[i].train()
                if i >= model.optimized_models:
                    model.netC[i].eval()
                    model.netF[i].eval()

    torch.save(model.state_dict(), osp.join(args.output_dir, "model" + ".pt"))

    return model


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, model, args):
    start_test = True
    all_fea_list = {}
    all_out_list = {}
    pred_list = {}
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            tar_idx = data[2]
            inputs[0] = inputs[0].cuda()
            inputs[1] = inputs[1].cuda()
            feas, outputs = model.forward_tran(inputs, tar_idx)

            if start_test:
                all_fea = feas.float().cpu()

                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    print("Accuracy = {:.2f}% ".format(accuracy * 100))

    pred_label = np.array(predict)

    return pred_label.astype("int")


def aggregation_model(
    netF,
    netC,
    aug,
    optimized_models,
    model_path,
    tran_tran,
    model_features,
    model_outputs,
    args,
):
    update_F_list = []
    update_C_list = []

    feature_encoder_wo_update_set = []
    for i in range(optimized_models):
        feature_encoder_wo_update_set.append(i)

    model = OursEnsemble(
        netF,
        netC,
        args.class_num,
        len(netF),
        optimized_models,
        tran_tran,
        aug,
        feature_encoder_wo_update_set,
        model_features,
        model_outputs,
    )

    for i in range(optimized_models):

        update_C_list.append(i)

    model, optimizer = model_initial_tc(
        model, update_F_list, update_C_list, len(netF), optimized_models, model_path
    )

    return model, optimizer


def ent(output):
    softmax_out = nn.Softmax(dim=1)(output)
    Entropy = torch.mean(loss_function.Entropy(softmax_out))

    return Entropy


def calculate_MI_cls_loss(output, pse):

    Entropy = ent(output)
    classifier_loss = nn.CrossEntropyLoss()(output, pse)
    loss = Entropy
    return loss


def pse_training(outputs, all_label, args):

    pred_class = []
    pred_pro = []
    for out in outputs:
        out = torch.from_numpy(out)
        out = nn.Softmax(1)(out)
        prob, predict = torch.max(out, 1)
        pred_pro.append(prob)
        pred_class.append(predict)

    pred_pro = torch.stack(pred_pro, dim=1)
    pred_class = torch.stack(pred_class, dim=1)
    selected_model_indices = torch.argmax(pred_pro, dim=1)
    selected_pred_class = torch.gather(
        pred_class, 1, selected_model_indices.view(-1, 1)
    )
    selected_pred_prob = torch.gather(pred_pro, 1, selected_model_indices.view(-1, 1))
    mask = (selected_pred_prob > 0.95).float()
    selected_pred_class_masked = torch.masked_select(selected_pred_class, mask.bool())

    all_label_masked = torch.masked_select(all_label.unsqueeze(1), mask.bool())

    correct_predictions = (selected_pred_class_masked == all_label_masked).sum().item()
    total_samples = mask.sum().item()

    accuracy = correct_predictions / total_samples

    print("Accuracy for samples where mask == 1:", accuracy)
    return selected_pred_class, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAiDA")
    parser.add_argument("--t", type=int, default=3, help="target")
    parser.add_argument("--max_epoch", type=int, default=15, help="max iterations")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dset",
        type=str,
        default="office-home",
        choices=["office31", "office-home", "domainnet"],
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--source_pth_save_dir", type=str, default="./source/")
    parser.add_argument("--output_dir", type=str, default="./target/")
    parser.add_argument("--fix", type=bool, default=False)
    parser.add_argument(
        "--distance", type=str, default="cosine", choices=["euclidean", "cosine"]
    )
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--source_feature_save_dir", type=str, default="./features/")
    parser.add_argument("--model_config_path", type=str, default="./configs/")
    parser.add_argument(
        "--transferability_save_path", type=str, default="./model_sort/"
    )
    parser.add_argument("--config_file", type=str, default="office-home_main")
    parser.add_argument(
        "--trans_method",
        type=str,
        default="SUTE",
        choices=[
            "SUTE",
            "NMI",
            "ANE",
            "SND",
            "MMD",
            "A-distance",
            "MDE",
            "LogME_pse",
            "LogME_true",
            "LEEP_pse",
            "LEEP_true",
        ],
    )
    parser.add_argument("--diversity", type=bool, default=True)
    parser.add_argument("--data_folder", type=str, default="./data/")

    args = parser.parse_args()

    if args.dset == "office-home":
        names = ["Art", "Clipart", "Product", "Real_World"]
        args.class_num = 65
    if args.dset == "office31":
        names = ["amazon", "dslr", "webcam"]
        args.class_num = 31
    if args.dset == "office-caltech":
        names = ["amazon", "caltech", "dslr", "webcam"]
        args.class_num = 10
    if args.dset == "domainnet":
        names = ["real", "infograph", "painting", "sketch", "clipart", "quickdraw"]
        args.class_num = 345

    feature_path = args.source_feature_save_dir

    method = args.trans_method

    folder = args.data_folder
    for tt in range(len(names)):
        args.t = tt
        args.t_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"
        model_config_file = (
            args.model_config_path + args.config_file + "/" + str(args.t) + ".txt"
        )

        if not os.path.exists(args.transferability_save_path + args.config_file):
            os.makedirs(args.transferability_save_path + args.config_file)
        transferability_output_filename = (
            args.transferability_save_path
            + args.config_file
            + "/"
            + str(args.t)
            + ".txt"
        )

        all_output = transfer_calcualte_for_individual_models(
            model_config_file,
            feature_path,
            transferability_output_filename,
            method,
            args=args,
        )
        (
            transfer_set,
            div_set,
            tran_tran,
            div_tran,
            sorted_div_tran,
            tran_output,
            div_output,
            sorted_model_names,
            pse_tran,
            label,
        ) = model_selection_function(
            feature_path, transferability_output_filename, all_output, method, args
        )

        if args.diversity:
            transfer_set.append(div_set[0])
            tran_tran.append(sorted_div_tran[0])

        tran_tran = np.array(tran_tran)
        tran_tran = torch.from_numpy(tran_tran)
        other_set = [
            selected_model
            for selected_model in sorted_model_names
            if selected_model not in transfer_set
        ]

        netF = nn.ModuleList()
        netC = nn.ModuleList()
        aug_set = []
        model_path = []
        model_name_set = []
        new_netF = {}
        new_netC = {}

        model_features = {}
        model_outputs = {}
        for i, selected_model in enumerate(transfer_set):

            dset, model_structure, model_name, source, target = (
                split_dset_structure_model_source_target(selected_model)
            )

            fea, output, _, _ = load_mat(feature_path + selected_model)
            model_features[i] = torch.from_numpy(fea)
            model_outputs[i] = torch.from_numpy(output)
            source = names[int(source)][0].upper()
            target = names[int(target)][0].upper()

            new_netF[i], new_netC[i], aug = construct_net(model_name)

            netF.append(new_netF[i])
            netC.append(new_netC[i])
            model_name_set.append(model_name)
            aug_set.append(aug)

            model_path.append(
                get_source_model_path(
                    args.source_pth_save_dir, dset, model_name, source
                )
            )

        print("selected inlier models", model_name_set)
        print("outliner models", other_set)
        optimized_models = len(transfer_set)

        model, optimizer = aggregation_model(
            netF,
            netC,
            aug_set,
            optimized_models,
            model_path,
            tran_tran,
            model_features,
            model_outputs,
            args,
        )

        args.output_dir = osp.join(
            args.output_dir, args.dset, args.config_file, names[args.t][0].upper()
        )

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.name = "->" + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system("mkdir -p " + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, "log" + ".txt"), "w")
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()

        epoches = 1
        try:
            selected_pred_class, mask = pse_training(div_output[1:], label, args)
        except:
            selected_pred_class, mask = pse_training(tran_output, label, args)
        pred_acc = torch.sum(pse_tran == label) / float(pse_tran.shape[0])
        print("SAA TF Accuracy = ", pred_acc)

        model = train_target(
            args, model, optimizer, tran_tran, selected_pred_class, mask, pse_tran
        )
