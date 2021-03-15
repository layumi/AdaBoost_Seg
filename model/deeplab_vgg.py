import numpy as np
import torch
from torch import nn
from torchvision import models


def NormLayer(norm_dim, norm_style = 'gn'):
    if norm_style == 'bn':
        norm_layer = nn.BatchNorm2d(norm_dim)
    elif norm_style == 'in':
        norm_layer = nn.InstanceNorm2d(norm_dim, affine = True)
    elif norm_style == 'ln':
        norm_layer = nn.LayerNorm(norm_dim,  elementwise_affine=True)
    elif norm_style == 'gn':
        norm_layer = nn.GroupNorm(num_groups=32, num_channels=norm_dim, affine = True)
    return norm_layer

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)
    
class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes, norm_style = 'gn'):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(dims_in, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                NormLayer(256, norm_style),
                nn.ReLU(inplace=True) ]))


        self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        NormLayer(512, norm_style) ])
            
        self.head = nn.Sequential(*[nn.Dropout2d(0.2),
            nn.Conv2d(512, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])
        
        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        out = self.head(out)
        return out


class DeeplabVGG(nn.Module):
    def __init__(self, num_classes, vgg16_caffe_path=None, pretrained=False):
        super(DeeplabVGG, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(vgg16_caffe_path))

        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        #remove pool4/pool5
        features = nn.Sequential(*(features[i] for i in list(range(23))+list(range(24,30))))

        for i in [23,25,27]:
            features[i].dilation = (2,2)
            features[i].padding = (2,2)

        fc6 = nn.Conv2d(512, 1024, kernel_size=3, padding=4, dilation=4)
        fc7 = nn.Conv2d(1024, 1024, kernel_size=3, padding=4, dilation=4)

        self.features1 = nn.Sequential(*[features[i] for i in range(len(features))])
        self.features2 = nn.Sequential(*[ fc6, nn.ReLU(inplace=True), fc7, nn.ReLU(inplace=True)])

        self.classifier1 = Classifier_Module(512, [6,12,18,24],[6,12,18,24],num_classes)
        self.classifier2 = Classifier_Module(1024, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        x = self.features1(x)
        x1 = self.classifier1(x)
        x = self.features2(x)
        x2 = self.classifier2(x)
        return x1, x2

    def optim_parameters(self, args):
        return self.parameters()
