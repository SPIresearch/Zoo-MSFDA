import argparse
import copy
import math
import os
import os.path as osp
import pdb
import random

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torchvision import transforms

import loss_function
import network
from data_list import ImageList
from loss_function import CrossEntropyLabelSmooth


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


def image_train(resize_size=384, crop_size=384):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=384, crop_size=384):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def model_forward(args, netF, netC, X, y):
    if args.fix:
        with torch.no_grad():
            fea = netF(X)
    else:
        fea = netF(X)
    output = netC(fea)

    losses = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(
        output, y
    )

    return losses


def get_classes(image_list):
    if len(image_list[0].split()) > 2:
        images = [
            (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
            for val in image_list
        ]
    else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    image_list = images
    class_dict1 = {}

    for index in range(len(image_list)):
        path, target = image_list[index]
        classes = path.split("/")[-2]
        if target not in class_dict1.keys():
            class_dict1[target] = classes

    classes_list = [class_dict1[i] for i in range(len(list(class_dict1.keys())))]
    return classes_list


def data_load(args):

    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)

        tr_txt, te_txt = torch.utils.data.random_split(
            txt_src, [tr_size, dsize - tr_size]
        )
    else:
        dsize = len(txt_src)
        tr_size = int(0.9 * dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    class_list = get_classes(txt_test)

    dsets["source_tr"] = ImageList(
        tr_txt,
        transform=image_train(args.resize_size, args.crop_size),
        class_list=class_list,
    )
    dset_loaders["source_tr"] = DataLoader(
        dsets["source_tr"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["source_te"] = ImageList(
        te_txt,
        transform=image_test(args.resize_size, args.crop_size),
        class_list=class_list,
    )
    dset_loaders["source_te"] = DataLoader(
        dsets["source_te"],
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dsets["test"] = ImageList(
        txt_test,
        transform=image_test(args.resize_size, args.crop_size),
        class_list=class_list,
    )
    dset_loaders["test"] = DataLoader(
        dsets["test"],
        batch_size=train_bs,
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )

    return dset_loaders


def cal_acc(loader, netF, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netF(inputs)
            outputs = netC(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = torch.mean(loss_function.Entropy(all_output)).cpu().data.item()

    return accuracy * 100, mean_ent


def calculate_structure_semantics(all_output, all_label, all_fea, args):

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)

    _, predict = torch.max(all_output, 1)

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], "cosine")
    pred_struc_label = dd.argmin(axis=1)
    pred_struc_label = labelset[pred_struc_label]

    for round in range(1):
        aff = np.eye(K)[pred_struc_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], "cosine")
        pred_struc_label = dd.argmin(axis=1)
        pred_struc_label = labelset[pred_struc_label]

    return pred_struc_label.astype("int")


def obtain_pse_label_struc(all_output, all_label, all_fea, args, tau=0.5):

    _, predict = torch.max(all_output, 1)

    pse_label_struc = calculate_structure_semantics(
        all_output, all_label, all_fea, args
    )

    return pse_label_struc


def cal_acc_test(loader, netF, netC, feature_save_dir="./features", flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features = netF(inputs)
            outputs = netC(features)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    if not os.path.exists(feature_save_dir):
        os.makedirs(feature_save_dir)
    scipy.io.savemat(
        f"{feature_save_dir}/{args.dset}_{args.net}_{str(args.s)}_{str(args.t)}.mat",
        {
            "ft": all_features.numpy(),
            "output": all_output.numpy(),
            "label": all_label.numpy(),
        },
    )
    pse_struc = obtain_pse_label_struc(all_output, all_label, all_features, args)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    return accuracy * 100, accuracy


def train_source(args, netF, netC):
    dset_loaders = data_load(args)

    param_group = []
    learning_rate = args.lr

    for k, v in netF.named_parameters():
        if args.fix:
            v.requires_grad = False
        else:
            param_group += [{"params": v, "lr": learning_rate * 0.1}]

    for k, v in netC.named_parameters():

        param_group += [{"params": v, "lr": learning_rate}]

    if args.optimizer == "sgd":
        optimizer = optim.SGD(param_group)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(param_group)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(param_group)
    elif args.optimizer == "asgd":
        optimizer = optim.ASGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 5
    iter_num = 0
    if args.fix:
        netF.eval()
    else:
        netF.train()

    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source, text_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source, text_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        classifier_loss = model_forward(args, netF, netC, inputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()

            acc_s_te, _ = cal_acc(dset_loaders["source_te"], netF, netC, False)

            log_str = "Task: {}, Iter:{}/{}; Accuracy = {:.2f}%".format(
                args.name_src, iter_num, max_iter, acc_s_te
            )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            if not args.fix:
                netF.train()
            netC.train()
            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()

                best_netC = netC.state_dict()

            torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))

            torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netC


def simple_transform(x, beta):
    x = 1 / torch.pow(torch.log(1 / x + 1), beta)
    return x


def test_target(args, netF, netC):
    dset_loaders = data_load(args)

    args.modelpath = args.output_dir_src + "/source_F.pt"
    netF.load_state_dict(torch.load(args.modelpath))

    args.modelpath = args.output_dir_src + "/source_C.pt"
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()

    netC.eval()

    acc, _ = cal_acc_test(
        dset_loaders["test"], netF, netC, args.feature_save_dir, False
    )

    log_str = "\nTraining: {}, Task: {}, Accuracy = {:.2f}%".format(
        args.trte, args.name, acc
    )

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAiDA")
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="3", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=50, help="max iterations")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--dset",
        type=str,
        default="office-home",
        choices=["office31", "office-home", "domainnet"],
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--net", type=str, default="vit_l_16")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--pth_save_dir", type=str, default="./source/")
    parser.add_argument("--trte", type=str, default="val", choices=["full", "val"])
    parser.add_argument("--fix", type=bool, default=False)
    parser.add_argument("--feature_save_dir", type=str, default="./features")
    parser.add_argument("--data_folder", type=str, default="./data/")
    parser.add_argument("--optimizer", type=str, default="sgd")
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

    def initialize(args):
        resize_size = 256
        crop_size = 224

        if args.net == "vit_l_16":
            netF = network.VitBase(res_name=args.net).cuda()
        elif args.net == "vit_b_16":
            netF = network.VitBase(res_name=args.net).cuda()
        elif args.net == "vit_b_32":
            netF = network.VitBase(res_name=args.net).cuda()
        elif args.net == "vit_h_14":
            netF = network.VitBase(res_name=args.net).cuda()
            resize_size = 518
            crop_size = 518
        elif args.net == "vit_l_32":
            netF = network.VitBase(res_name=args.net).cuda()
        elif args.net == "swin_t":

            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_l":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_b":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_s":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_v2_t":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_v2_s":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "swin_v2_b":
            netF = network.SwinBase(res_name=args.net).cuda()
        elif args.net == "resnet50":
            netF = network.ResBase(res_name="resnet50").cuda()

        elif args.net == "resnet50_v1":
            netF = network.ResBase(res_name="resnet50_v1").cuda()

        elif args.net == "resnet50_nopre":
            netF = network.ResBase(res_name="resnet50_nopre").cuda()
        elif args.net == "resnet101":
            netF = network.ResBase(res_name="resnet101").cuda()
        elif args.net == "efficientnet_v2_s":

            netF = network.EffBase(res_name="efficientnet_v2_s").cuda()
        elif args.net == "efficientnet_v2_m":
            netF = network.EffBase(res_name="efficientnet_v2_m").cuda()
        elif args.net == "efficientnet_v2_l":
            netF = network.EffBase(res_name="efficientnet_v2_l").cuda()

        netC = network.feat_classifier(
            type=args.layer,
            feature_dim=netF.in_features,
            class_num=args.class_num,
            bottleneck_dim=args.bottleneck,
        ).cuda()
        return netF, netC, resize_size, crop_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    oripath = args.pth_save_dir

    for k in range(len(names)):
        args.s = k
        netF, netC, args.resize_size, args.crop_size = initialize(args)
        folder = args.data_folder
        args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
        args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

        args.pth_save_dir = oripath + "_" + args.net

        args.output_dir_src = osp.join(
            args.pth_save_dir, args.dset, names[args.s][0].upper()
        )
        args.name_src = names[args.s][0].upper()
        if not osp.exists(args.output_dir_src):
            os.system("mkdir -p " + args.output_dir_src)
        if not osp.exists(args.output_dir_src):
            os.mkdir(args.output_dir_src)

        args.out_file = open(osp.join(args.output_dir_src, "log.txt"), "w")
        args.out_file.write(print_args(args) + "\n")
        args.out_file.flush()

        train_source(args, netF, netC)

        args.out_file = open(
            osp.join(args.output_dir_src, "log_test_transform.txt"), "w"
        )
        for i in range(len(names)):
            if i == args.s:
                continue
            args.t = i
            args.name = names[args.s][0].upper() + names[args.t][0].upper()
            args.s_dset_path = folder + args.dset + "/" + names[args.s] + "_list.txt"
            args.test_dset_path = folder + args.dset + "/" + names[args.t] + "_list.txt"

            test_target(args, netF, netC)
