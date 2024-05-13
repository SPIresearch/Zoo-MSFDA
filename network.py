from pyexpat import model
import numpy as np
import torch,pdb
import torch.nn as nn
from torchvision import models
import clip
import torch.nn.utils.weight_norm as weightNorm
import timm
from timm.models import create_model
import torch.nn.functional as F
import copy
from transformers import AutoImageProcessor, SwinModel,AutoFeatureExtractor, SwinForImageClassification,CLIPModel

from loss_function import ce_loss,KL,entropy_loss,total_entropy_loss,CrossEntropy1
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        res_dict = {"resnet18":models.resnet18(pretrained=True), "resnet34":models.resnet34(pretrained=True), "resnet50":models.resnet50(pretrained=True),
"resnet101":models.resnet101(pretrained=True), "resnet152":models.resnet152(pretrained=True),"resnet50_v1":models.resnet50(weights='IMAGENET1K_V1'), "resnet50_nopre":models.resnet50(pretrained=False)}

        model_resnet =res_dict[res_name]
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = nn.ReLU()
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool =nn.AdaptiveAvgPool2d(1) #model_resnet.avgpool
        self.in_features =2048 #model_resnet.fc.in_features
        #self.in_features =256 #model_resnet.fc.in_features
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f = self.layer4(x)
        x = self.avgpool(f)
        x = x.view(x.size(0), -1)
        return x#,f


class EffBase(nn.Module):
    def __init__(self, res_name):
        super(EffBase, self).__init__()
        eff_dict = {"efficientnet_v2_s":models.efficientnet_v2_s(weights='IMAGENET1K_V1'),"efficientnet_v2_m":models.efficientnet_v2_m(weights='IMAGENET1K_V1'),"efficientnet_v2_l":models.efficientnet_v2_l(weights='IMAGENET1K_V1')}
        self.eff =copy.deepcopy(eff_dict[res_name])
        self.in_features = self.eff.classifier[1].in_features
        self.eff.classifier = nn.Sequential()
        

    def forward(self, x):
        x = self.eff(x)
        x = x.view(x.size(0), -1)
        return x

class DENSEBase(nn.Module):
    def __init__(self, res_name):
        super(DENSEBase, self).__init__()
        dsense_dict = {"densenet161":models.densenet161(weights='DEFAULT'),"densenet201":models.densenet201(weights='DEFAULT')}
        self.den =copy.deepcopy(dsense_dict[res_name])
        #pdb.set_trace()
        self.in_features = self.den.classifier.in_features
        self.den.classifier = nn.Sequential()
        

    def forward(self, x):
        x = self.den(x)
        x = x.view(x.size(0), -1)
        return x

class Mobv3Base(nn.Module):
    def __init__(self, res_name):
        super(Mobv3Base, self).__init__()
        dsense_dict = {"mobilenet_v3_small":models.mobilenet_v3_small(weights='DEFAULT'),"mobilenet_v3_large":models.mobilenet_v3_large(weights='DEFAULT')}
        self.den =copy.deepcopy(dsense_dict[res_name])
        #pdb.set_trace()
        self.in_features = self.den.classifier[0].in_features
        self.den.classifier = nn.Sequential()
        

    def forward(self, x):
        x = self.den(x)
        x = x.view(x.size(0), -1)
        return x



class VitBase(nn.Module):
    def __init__(self, res_name):
        super(VitBase, self).__init__()
        vit_dict = {"vit_l_16":models.vit_l_16,"vit_b_16":models.vit_b_16,"vit_b_32":models.vit_b_32,"vit_l_32":models.vit_l_32,"vit_h_14":models.vit_h_14
            
            # 'vit_s_32_21k':create_model('vit_small_patch32_224.augreg_in21k', pretrained=True),
            # 'vit_s_32_21k':create_model('vit_small_patch32_224.augreg_in21k', pretrained=True)
                                        }
        if res_name=='vit_h_14':
            self.vit=copy.deepcopy(models.vit_h_14(weights="IMAGENET1K_SWAG_E2E_V1"))
        else:
            self.vit =vit_dict[res_name](pretrained=True)#timm.create_model('seresnet50', pretrained=True) #se_resnet50()#res_dict[res_name](pretrained=False)
        
        self.in_features =self.vit.heads[0].in_features
        self.vit.heads=nn.Sequential()
         #model_resnet.fc.in_features
    def forward(self, x):
        
        x = self.vit(x)
        
        x = x.view(x.size(0), -1)
        return x


class SwinBase(nn.Module):
    def __init__(self, res_name):
        super(SwinBase, self).__init__()
        swin_dict = {"swin_t":models.swin_t(weights='IMAGENET1K_V1'),"swin_s":models.swin_s(weights='IMAGENET1K_V1'),"swin_b":models.swin_b(weights='IMAGENET1K_V1'),"swin_v2_t":models.swin_v2_t(weights='IMAGENET1K_V1'),"swin_v2_s":models.swin_v2_s(weights='IMAGENET1K_V1'),"swin_v2_b":models.swin_v2_b(weights='IMAGENET1K_V1')}
        self.res_name=res_name
        if res_name=='swin_l':
            #self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-large-patch4-window7-224")
            self.swin = create_model('swin_large_patch4_window7_224', pretrained=True)#SwinForImageClassification.from_pretrained("microsoft/swin-large-patch4-window7-224")     
            self.pool=self.swin.head.global_pool
        else:
            self.swin =copy.deepcopy(swin_dict[res_name])
        self.in_features =self.swin.head.in_features
        #pdb.set_trace()
        self.swin.head = nn.Sequential()
        #self.pool= nn.SelectAdaptivePool2d (pool_type=avg, flatten=Identity())
    def forward(self, x):
        
        x = self.swin(x)
        if self.res_name=='swin_l':
            x=self.pool(x)
        x = x.view(x.size(0), -1)
        
        return x



class ClipBase(nn.Module):
    def __init__(self,res_name):
        super(ClipBase, self).__init__()
        
        self.clip,_ = clip.load(res_name)

        self.in_features =512#768#512 #model_resnet.fc.in_features
        
    def forward(self, x,t):
        image_features = self.clip.encode_image(x)
        #text_features = model.encode_text(text)
        logits_per_image, logits_per_text = self.clip(x, t)
        
        return image_features,logits_per_image





class feat_classifier(nn.Module):
    def __init__(self, feature_dim,class_num,  bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
       
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        x = self.fc(x)
 
        return x

