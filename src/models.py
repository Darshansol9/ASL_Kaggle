import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
import torch

class ResNet34(nn.Module):

    def __init__(self,pretrained=None):
        super(ResNet34,self).__init__()

        #print('-------------Inside Model -----------------',pretrained)
        if(pretrained):
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet34'](pretrained=None)

        self.lo = nn.Linear(512,36)

    def forward(self,x):
        
        bs,_,_,_ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(bs,-1)
        lo = self.lo(x)
        
        return lo