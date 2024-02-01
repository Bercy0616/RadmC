import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math

__all__ = ['DenseNet']



class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_classes (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate,
                 block_config,
                 bn_size=4,
                 drop_rate=0,
                 in_channels=1,
                 num_classes = 2,
                 fc_num = 128,
                 flat = name):

        super(DenseNet, self).__init__()
        self.fc_num = fc_num
        num_init_features=64 if in_channels==3 else 32
        if 'multi' in name:
            self.flat = 'multi'
        elif 'radiomics' in name: 
            self.flat = 'radiomics'
        elif 'methylation' in name: 
            self.flat = 'methylation'

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     in_channels,
                     num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
        
        features_nums_out = [num_init_features]
        for i in range(4):
            if i != 3:
                features_nums_out.append((features_nums_out[-1]+block_config[i]*growth_rate)//2)
            else:
                features_nums_out.append(features_nums_out[-1]+block_config[i]*growth_rate)
        self.features_nums_out = features_nums_out
        first_features = self.features_nums_out[-4]
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            #if i == 3:
            
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        #for p in self.parameters():
        #    p.requires_grad = False


        self.linear = nn.Sequential(nn.Linear(num_features, 2))  
        self.classify = nn.Sequential(nn.Linear(num_features, self.fc_num),
                                 nn.BatchNorm1d(self.fc_num),
                                 nn.GELU(),
                                 nn.Linear(self.fc_num, 2))
        
        self.toplayer=nn.Conv3d(self.features_nums_out[-1],first_features,1,stride=(1, 1, 1),bias=False)
        
        #横向连接，保证通道数相同
        self.latlayer1=nn.Conv3d(self.features_nums_out[-2],first_features,1,stride=(1, 1, 1),bias=False)
        self.latlayer2=nn.Conv3d(self.features_nums_out[-3],first_features,1,stride=(1, 1, 1),bias=False)
        self.latlayer3=nn.Conv3d(first_features,first_features,1,stride=(1, 1, 1),bias=False)
        #3x3卷积融合特征
        self.smooth1=nn.Conv3d(first_features,first_features,3,1,1)
        self.smooth2=nn.Conv3d(first_features,first_features,3,1,1)
        self.smooth3=nn.Conv3d(first_features,first_features,3,1,1)
        
        self.fc1 = nn.Sequential(nn.Linear(first_features, first_features),
                                 nn.BatchNorm1d(first_features),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(first_features, first_features),
                                 nn.BatchNorm1d(first_features),
                                 nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(first_features, first_features),
                                 nn.BatchNorm1d(first_features),
                                 nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(first_features, first_features),
                                 nn.BatchNorm1d(first_features),
                                 nn.ReLU())
        
        self.embd2 = nn.Sequential(nn.Linear(first_features, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU())
        self.embd = nn.Sequential(nn.Linear(first_features*4, 32),
                                 nn.BatchNorm1d(32),
                                 nn.ReLU())
        
        self.linear3 = nn.Sequential(nn.Linear(128*2, 64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Linear(64,2))
        
        self.linear2 = nn.Sequential(nn.Linear(128, 64),
                                 nn.BatchNorm1d(64),
                                 nn.ReLU(),
                                 nn.Linear(64,2))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x2,fea):
        inputs = self.features.conv0(x2)
        inputs = self.features.relu0(inputs)
        c1 = self.features.pool0(inputs)
        c2 = self.features.denseblock1(c1)
        c2 = self.features.transition1(c2)
        c3 = self.features.denseblock2(c2)
        c3 = self.features.transition2(c3)
        c4 = self.features.denseblock3(c3)
        c4 = self.features.transition3(c4)
        c5 = self.features.denseblock4(c4)
        c5 = self.features.norm5(c5)
        c5 = F.relu(c5, inplace=True)
        
        p5=self.toplayer(c5)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        p3=self._upsample_add(p4,self.latlayer2(c3))
        p2=self._upsample_add(p3,self.latlayer3(c2))
        p4=self.smooth1(p4)
        p3=self.smooth2(p3)
        p2=self.smooth3(p2)
        p5 = F.adaptive_avg_pool3d(p5, (1, 1, 1))
        p5 = p5.view(p5.size(0), -1)
        p4 = F.adaptive_avg_pool3d(p4, (1, 1, 1))
        p4 = p4.view(p4.size(0), -1)
        p3 = F.adaptive_avg_pool3d(p3, (1, 1, 1))
        p3 = p3.view(p3.size(0), -1)
        p2 = F.adaptive_avg_pool3d(p2, (1, 1, 1))
        p2 = p2.view(p2.size(0), -1)
        p5 = self.fc1(p5)
        p4 = self.fc2(p4)
        p3 = self.fc3(p3)
        p2 = self.fc4(p2)
        feature = p5+p4+p3+p2
        feature = self.embd2(feature)

        if self.flat == 'radiomics':
            out = self.linear2(feature)
            out = F.softmax(out,dim=1)
        else:
            feature = torch.cat([feature,fea],-1)
            out = self.linear3(feature)
            out = F.softmax(out,dim=1)
        return out
    
    def _upsample_add(self,x,y):
        _,_,H,W,D=y.shape
        return nn.functional.interpolate(x,size=(H,W,D),mode='trilinear')+y
    
    def cal_features(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(features.size(0), -1)
        return out

def densenetmini_3d(num_classes,fc_num,name,**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 6, 6, 6),
        num_classes = num_classes,
        fc_num = 128,
        flat = name,
        **kwargs)
    return model

def densenet121_3d(num_classes,fc_num,name,**kwargs):
    model = DenseNet(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_classes = num_classes,
        fc_num = 128,
        flat = name,
        **kwargs)
    return model


if __name__ == '__main__':
    resize = 64
    net2 = densenet121_3d(num_classes=2,fc_num = 128)
    x = torch.randn(2, 1, resize, resize, resize)
    x2 = torch.randn(2,32)
    y2= net2(x,x2)
    print(y2.shape)
