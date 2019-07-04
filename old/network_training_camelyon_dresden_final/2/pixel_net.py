import torch
import torch.nn as nn
import torch.nn.functional as F

def maxpool_muj(x,kernel,stride,dil,dense):
    if not dense:
        x=F.max_pool2d(x, kernel,padding=0, stride=stride, dilation=dil)
    else:
        x=F.max_pool2d(x, kernel, stride=1,padding=0, dilation=dil)
        dil=dil*stride
    return x,dil


class Conv2_muj(nn.Conv2d):
    def forward(self,x,dil):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, dil, self.groups)



class Conv2BnRelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=0):
        super().__init__()
    
        self.conv=Conv2_muj(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm2d(out_size,momentum=0.1)

#        self.conv=weightNorm(self.conv,name = "weight")
        dov=0.05
        self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout2d(dov))

    def forward(self, inputs,dil=1,dense=0):
        
        
        outputs = self.conv(inputs,dil)
            
        outputs = self.bn(outputs)          
        outputs=F.relu(outputs)
        outputs = self.do(outputs)

        return outputs
    
    
    
class PixelNet(nn.Module):
    def __init__(self,K=24,input_size=3):
        super().__init__()
        
        self.dense=0
        
        self.initc=Conv2BnRelu(input_size, K,filter_size=1)

        self.c1_1=Conv2BnRelu(K, K)
        self.c1_2=Conv2BnRelu(K, K)
        self.c1_3=Conv2BnRelu(K, K)

        self.t1=Conv2BnRelu(4*K, K,filter_size=1)
        
        self.c2_1=Conv2BnRelu(K, K)
        self.c2_2=Conv2BnRelu(K, K)
        self.c2_3=Conv2BnRelu(K, K)
        
        self.t2=Conv2BnRelu(4*K, K,filter_size=1)
       
        
        self.c3_1=Conv2BnRelu(K, K)
        self.c3_2=Conv2BnRelu(K, K)
        self.c3_3=Conv2BnRelu(K, K)

        self.t3=Conv2BnRelu(4*K, K,filter_size=1)

        self.c4=Conv2BnRelu(K, K,filter_size=5)

        self.finalc=nn.Conv2d(K, 1,1)
        
        
    def sum_remve_border(self,input1,input2,kolik=2):
        
        
        
        input1_small=input1[:, :,kolik:-kolik,kolik:-kolik]
        
        return input1_small+input2
#        return torch.cat([input1_small,input2],1)
        
    def sum_remve_border2(self,x,y1,y2,y3,dil):
        
        
        kolik=dil*1
        y2=y2[:, :,kolik:-kolik,kolik:-kolik]
        
        kolik=dil*2
        y1=y1[:, :,kolik:-kolik,kolik:-kolik]
        
        kolik=dil*3
        x=x[:, :,kolik:-kolik,kolik:-kolik]
        
#        return input1_small+input2
        return torch.cat([x,y1,y2,y3],1)
    
    
    
    
    def dense_on(self):
        self.dense=1
    
    def dense_off(self):
        self.dense=0
        
    def forward(self, inputs):
        
        dil=1
        
        x=self.initc(inputs,dil,self.dense)#96
        
        y1=self.c1_1(x,dil,self.dense)#94
        y2=self.c1_2(y1,dil,self.dense)#92
        y3=self.c1_3(y2,dil,self.dense)#90
        x=self.sum_remve_border2(x,y1,y2,y3,dil)
        x=self.t1(x)
        x,dil=maxpool_muj(x,3,2,dil,self.dense)#44/88
        
        
        y1=self.c2_1(x,dil,self.dense)#42/84
        y2=self.c2_2(y1,dil,self.dense)#40/80
        y3=self.c2_3(y2,dil,self.dense)#38/76
        x=self.sum_remve_border2(x,y1,y2,y3,dil)
        x=self.t2(x)
        x,dil=maxpool_muj(x,3,2,dil,self.dense)#18/72
        
        
        
        y1=self.c3_1(x,dil,self.dense)#16/64
        y2=self.c3_2(y1,dil,self.dense)#14/56
        y3=self.c3_3(y2,dil,self.dense)#12/48
        x=self.sum_remve_border2(x,y1,y2,y3,dil)
        x=self.t3(x)
        x,dil=maxpool_muj(x,3,2,dil,self.dense)#5/40
        
        
#        if self.dense:
#            x=x[:,:,]
        
        x=self.c4(x,dil,self.dense)#1/8
        
        x=self.finalc(x)
        
        
        return x
        
        
        
        
        