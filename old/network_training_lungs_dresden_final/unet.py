import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
#from groupy.gconv.pytorch_gconv.splitgconvT2d import P4MConvTP4M,P4MConvTZ2

import torch.nn.utils.weight_norm as weightNorm


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=1,do_batch=1):
        super().__init__()
        
        self.do_batch=do_batch
    
        self.conv=nn.Conv2d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)

#        self.conv=weightNorm(self.conv,name = "weight")
        dov=0.1
        self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout2d(dov))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        
        if self.do_batch:
            outputs = self.bn(outputs)          
            outputs=F.relu(outputs)
            outputs = self.do(outputs)

        return outputs
    
    
    

class unetConvT2(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=2,pad=1,out_pad=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_size, out_size,filter_size,stride=stride, padding=pad, output_padding=out_pad)
        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs=F.relu(outputs)
        return outputs
    
    
    
    
    

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        self.up = unetConvT2(in_size, out_size )
        
#        self.up=nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, inputs1, inputs2):
        
       
        inputs2 = self.up(inputs2)


        return torch.cat([inputs1, inputs2], 1)




class Unet(nn.Module):
    def __init__(self, feature_scale=1,input_size=3):
        super().__init__()
        
        self.feature_scale=feature_scale
        
        filters = [64, 128, 256, 512]
        filters = [int(np.round(x / self.feature_scale)) for x in filters]
        
        
        self.conv1 = nn.Sequential(unetConv2(input_size, filters[0]),unetConv2(filters[0], filters[0]),unetConv2(filters[0], filters[0]))

        self.conv2 =  nn.Sequential(unetConv2(filters[0], filters[1] ),unetConv2(filters[1], filters[1] ),unetConv2(filters[1], filters[1] ))


        self.conv3 = nn.Sequential(unetConv2(filters[1], filters[2] ),unetConv2(filters[2], filters[2] ),unetConv2(filters[2], filters[2] ))


        self.center = nn.Sequential(unetConv2(filters[2], filters[3] ),unetConv2(filters[3], filters[3] ))




        self.up_concat3 = unetUp(filters[3], filters[3] )
        self.up_conv3=nn.Sequential(unetConv2(filters[2]+filters[3], filters[2] ),unetConv2(filters[2], filters[2] ))




        self.up_concat2 = unetUp(filters[2], filters[2] )
        self.up_conv2=nn.Sequential(unetConv2(filters[1]+filters[2], filters[1] ),unetConv2(filters[1], filters[1] ))


        self.up_concat1 = unetUp(filters[1], filters[1])
        self.up_conv1=nn.Sequential(unetConv2(filters[0]+filters[1], filters[0] ),unetConv2(filters[0], filters[0],do_batch=0 ))
        
        
        
        self.final = nn.Conv2d(filters[0], 1, 1)
        
        
    def forward(self, inputs):
        

        conv1 = self.conv1(inputs)
        x = F.max_pool2d(conv1,2,2)

        conv2 = self.conv2(x)
        x = F.max_pool2d(conv2,2,2)

        conv3 = self.conv3(x)
        x = F.max_pool2d(conv3,2,2)


        x = self.center(x)

    
        
        x = self.up_concat3(conv3, x)
        x = self.up_conv3(x)


        x = self.up_concat2(conv2, x)
        x=self.up_conv2(x)
        
        x = self.up_concat1(conv1, x)
        x=self.up_conv1(x)
        

        x = self.final(x)
        
#        sig=self.sm(final)
        return x