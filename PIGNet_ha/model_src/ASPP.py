import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGEConv


__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def model_size(model):
    total_size = 0
    for param in model.parameters():
        # °¢ ÆÄ¶ó¹ÌÅÍÀÇ ¿ø¼Ò °³¼ö °è»ê
        num_elements = torch.prod(torch.tensor(param.size())).item()
        # ¿ø¼Ò Å¸ÀÔ º°·Î ¹ÙÀÌÆ® Å©±â °è»ê (¿¹: float32 -> 4 bytes)
        num_bytes = num_elements * param.element_size()
        total_size += num_bytes
    return total_size
class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(2048, 512,  kernel_size=3, stride=1,padding=6,dilation=6)
        self.conv2 = nn.Conv2d(512, embedding_size,  kernel_size=3, stride=1,padding = 3,dilation=3)
        # self.conv3 = nn.Conv2d(512, embedding_size, kernel_size=1, stride=1)

        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(512,momentum=0.0003)
        self.bn2 = nn.BatchNorm2d(embedding_size,momentum=0.0003)
        # self.bn3 = nn.BatchNorm2d(embedding_size,momentum=0.0003)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.gelu(x)

        return x


class DecoderCNN(nn.Module):
    def __init__(self):
        super(DecoderCNN, self).__init__()
        self.conv3_transpose = nn.ConvTranspose2d(512, 1024, kernel_size=1, stride=1)
        self.conv2_transpose = nn.ConvTranspose2d(1024,2048, kernel_size=2, stride=3,dilation=2)
        self.gelu = nn.GELU()
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(2048)


    def forward(self, x):
        x = self.conv3_transpose(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.conv2_transpose(x)
        x = self.gelu(x)

        return x


class blockSAGEsq(nn.Module):
    def __init__(self,hidden, inner):
        super(blockSAGEsq,self).__init__()
        self.hidden = int(hidden)
        self.inner  = int(inner)
        self.sage1 = SAGEConv(self.hidden, self.hidden, aggregator = 'pool')
        self.sage2 = SAGEConv(self.hidden, self.hidden, aggregator = 'pool')

        # self.sage1 = SAGEConv(self.hidden, self.hidden, aggregator='gcn')
        # self.sage2 = SAGEConv(self.hidden, self.hidden, aggregator='gcn')


        self.linear = nn.Linear(self.hidden, self.inner)
    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.gelu(x)
        x = self.sage2(x, edge_index)
        x = F.gelu(x)
        x = self.linear(x)
        x = F.gelu(x)

        return x, edge_index

class SPP(nn.Module):

    def __init__(self, C, embedding_size, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(SPP, self).__init__()
        self._C = C
        self._embedding_size = embedding_size

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.gelu = nn.GELU()
        self.aspp1 = conv(C, self._embedding_size, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, self._embedding_size, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.aspp3 = conv(C, self._embedding_size, kernel_size=5, stride=1, padding=2,
                               bias=False)
        # self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
        #                        dilation=int(12*mult), padding=int(12*mult),
        #                        bias=False)
        # self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
        #                        dilation=int(18*mult), padding=int(18*mult),
        #                        bias=False)
        self.aspp5 = conv(C, self._embedding_size, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(self._embedding_size, momentum)
        self.aspp2_bn = norm(self._embedding_size, momentum)
        self.aspp3_bn = norm(self._embedding_size, momentum)
        # self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(self._embedding_size, momentum)

    def forward(self, x):

        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.gelu(x1)

        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.gelu(x2)
        #
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.gelu(x3)
        #
        # x4 = self.aspp4(x)
        # x4 = self.aspp4_bn(x4)
        # x4 = self.relu(x4)

        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.gelu(x5)

        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)

        # x = torch.cat((x1, x2, x5), 1)

        return [x1, x2, x3, x5]



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_groups=None, weight_std=False, beta=False, **kwargs):
        if 'embedding_size' in kwargs:
            self.embedding_size = kwargs['embedding_size']
        else:
            self.embedding_size = 21
        if 'n_layer' in kwargs:
            self.n_layer = kwargs['n_layer']
        else:
            self.n_layer = 12
        if 'n_skip_l' in kwargs:
            self.n_skip_l = kwargs['n_skip_l']
        else:
            self.n_skip_l = 1


        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,dilation=2)
        self.aspp = ASPP(2048,[12, 24, 36])


        # self.upsample = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=2, stride=1, padding=1,dilation=2)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) #block1
        x = self.layer2(x) #block2
        x = self.layer3(x) #block3
        x = self.layer4(x) #block4


        x = self.aspp(x)


        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)


        return x



def resnet50(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    """model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model"""
    if pretrained:
        print("Pretrained!!")
        model_dict = model.state_dict()
        if num_groups and weight_std:
            print("1")
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            #print("2")
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            print("3")
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    else:
        print("Not Pretrained!!")

    return model

def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_groups=num_groups, weight_std=weight_std, **kwargs)
    if pretrained:
        print("0")
        model_dict = model.state_dict()
        if num_groups and weight_std:
            print("1")
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            print("2")
            pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            print("3")
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)



class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 21
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)


        return self.project(res)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)