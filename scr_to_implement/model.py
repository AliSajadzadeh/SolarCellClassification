import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self,in_channel,out_channel,stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv_matching = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        self.bn_matching = nn.BatchNorm2d(out_channel)
        self.input_matching = None






    def forward(self,x):
        self.input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        skipped_connection = ResBlock.layer_matching(self,self.input)
        x += skipped_connection
        x =self.relu(x)
        return x





    def layer_matching(self,input):
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.input_matching = nn.Sequential(self.conv_matching,self.bn_matching)
            self.output = self.input_matching(input)
        else:
            self.output = input
        return self.output



class ResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv7 = nn.Conv2d(3,64,kernel_size=7,stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        # toget the better result add 2D averagepooling and then apply flatten
        self.globalaverage = nn.AdaptiveAvgPool2d
        self.FC = nn.Linear(512,2)
        self.prediction = nn.Sigmoid()
        self.ResBlock_object64_64 = ResBlock(64,64,1)
        self.ResBlock_object64_128 = ResBlock(64, 128, 2)
        self.ResBlock_object128_256 = ResBlock(128, 256, 2)
        self.ResBlock_object256_512 = ResBlock(256,512, 2)



    def forward(self, x):
        x = self.conv7(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.ResBlock_object64_64.forward(x)
        x = self.ResBlock_object64_128.forward(x)
        x = self.ResBlock_object128_256.forward(x)
        x = self.ResBlock_object256_512.forward(x)
        x = x.mean([2,3])
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.FC(x)
        x = self.prediction(x)
        return x









