import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math

class FCN_methylation(nn.Module):
    def __init__(self):
        super(FCN_methylation, self).__init__()
        self.inputs_num1 = 363-4
        self.inputs_num2 = 2907-4
        self.inputs_num3 = 3076-4
        self.inputs_num = self.inputs_num1+self.inputs_num2+self.inputs_num3

        self.mlp = nn.Sequential(nn.Linear(self.inputs_num, 256),
                                 nn.BatchNorm1d(256),
                                 nn.LeakyReLU(negative_slope=0.01),
                                 nn.Linear(256, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(negative_slope=0.01))
              
        self.classifier = nn.Sequential(nn.Linear(128, 2))

        #for p in self.parameters():
        #    p.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f1, f2, f3):
        feature = torch.concat([f1,f2,f3],axis = 1)
        feature = self.mlp(feature)
        out = self.classifier(feature)
        out = F.softmax(out,dim=1)
        return out,feature


if __name__ == '__main__':
    X1 = torch.ones([3,590])
    X2 = torch.ones([3,3132])
    X3 = torch.ones([3,3326])
    net = FCN_methylation()
    y = net(X1,X2,X3)
    print(y.shape)
