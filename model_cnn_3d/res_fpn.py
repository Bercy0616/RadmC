import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)
    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResFPN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 n_classes=2,
                 in_channels=1,
                 fc_num = 128):

        super(ResFPN, self).__init__()
        self.fc_num = fc_num
        self.num_classes = n_classes
        #first_features = 64 if in_channels == 3 else 32
        first_features = 64
        self.inplanes = first_features
        self.conv1 = nn.Conv3d(
            in_channels,
            first_features,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(first_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, first_features, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        
        self.toplayer=nn.Conv3d(512,first_features,1,stride=(1, 1, 1),bias=False)
        
        #横向连接，保证通道数相同
        self.latlayer1=nn.Conv3d(256,first_features,1,stride=(1, 1, 1),bias=False)
        self.latlayer2=nn.Conv3d(128,first_features,1,stride=(1, 1, 1),bias=False)
        self.latlayer3=nn.Conv3d(first_features,first_features,1,stride=(1, 1, 1),bias=False)
        #3x3卷积融合特征
        self.smooth1=nn.Conv3d(first_features,first_features,3,1,1)
        self.smooth2=nn.Conv3d(first_features,first_features,3,1,1)
        self.smooth3=nn.Conv3d(first_features,first_features,3,1,1)
        
        #for p in self.parameters():
        #    p.requires_grad = False

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
        
        self.classify_add = nn.Sequential(nn.Linear(first_features, self.fc_num),
                                 nn.BatchNorm1d(self.fc_num),
                                 nn.ReLU(),
                                 nn.Linear(self.fc_num, 2)) 
        
        self.classify_con = nn.Sequential(nn.Linear(first_features*4, self.fc_num),
                                 nn.BatchNorm1d(self.fc_num),
                                 nn.ReLU(),
                                 nn.Linear(self.fc_num, 2))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #自上而下的采样模块
    def _upsample_add(self,x,y):
        _,_,H,W,D=y.shape
        return nn.functional.interpolate(x,size=(H,W,D),mode='trilinear')+y
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        #print('c1：',c1.shape)
        c2=self.layer1(c1)
        #print('c2：',c2.shape)
        c3=self.layer2(c2)
        #print('c3：',c3.shape)
        c4=self.layer3(c3)
        #print('c4：',c4.shape)
        c5=self.layer4(c4)
        #print('c5：',c5.shape)
        p5=self.toplayer(c5)
        #print('p5：',p5.shape)
        p4=self._upsample_add(p5,self.latlayer1(c4))
        #print('p4：',p4.shape)
        p3=self._upsample_add(p4,self.latlayer2(c3))
        #print('p3：',p3.shape)
        p2=self._upsample_add(p3,self.latlayer3(c2))
        #print('p2：',p2.shape)
        #卷积的融合，平滑处理
        p4=self.smooth1(p4)
        #print('p4v2：',p4.shape)
        p3=self.smooth2(p3)
        #print('p3v2：',p3.shape)
        p2=self.smooth3(p2)
        #print('p2v2：',p2.shape)
        
        p5 = F.adaptive_avg_pool3d(p5, (1, 1, 1))
        p5 = p5.view(p5.size(0), -1)
        #print('p5_out：',p5.shape)
        p4 = F.adaptive_avg_pool3d(p4, (1, 1, 1))
        p4 = p4.view(p4.size(0), -1)
        #print('p4_out：',p4.shape)
        p3 = F.adaptive_avg_pool3d(p3, (1, 1, 1))
        p3 = p3.view(p3.size(0), -1)
        #print('p3_out：',p3.shape)
        p2 = F.adaptive_avg_pool3d(p2, (1, 1, 1))
        p2 = p2.view(p2.size(0), -1)
        #print('p2_out：',p2.shape)

        p5 = self.fc1(p5)
        p4 = self.fc2(p4)
        p3 = self.fc3(p3)
        p2 = self.fc4(p2)
        #print('p5_out2：',p5.shape)
        feature = p5+p4+p3+p2
        #print(feature.shape)
        out = self.classify_add(feature)
        #print('out：',out.shape)
        out = F.softmax(out,dim=1)
        #print('out：',out.shape)
        return out
        #return feature

if __name__ == '__main__':
    net = ResFPN(BasicBlock,[2,2,2,2])
    img_size = (64, 64, 64)
    x = torch.randn(3,1,img_size[0],img_size[1],img_size[2])
    out = net(x)
    print(out.shape)