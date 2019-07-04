import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def maxpool_muj(x,kernel,stride,dil,dense,gconv=0):
    
    if gconv:
        s = x.size()
        x = x.view(s[0], s[1]*s[2], s[3], s[4])
                 
    if not dense:
        x=F.max_pool2d(x, kernel,padding=0, stride=stride, dilation=dil)
    else:
        x=F.max_pool2d(x, kernel, stride=1,padding=0, dilation=dil)
        dil=dil*stride
       
    if gconv:
        x =  x.view(s[0], s[1], s[2], x.size()[2], x.size()[3])    
        
    return x,dil


def final_pooling(x):
    x=torch.mean(x, dim=2)
    return x


class Conv2_muj(nn.Conv2d):
    def forward(self,x,dil,gconv,p4m):
        if gconv:
            ww=self.weight
            sh=list(ww.size())
            w=np.arange(np.prod(sh))
            w=w.reshape(sh)
            w_list=[]
            
            bb=self.bias
            shb=list(bb.size())
            b=np.arange(np.prod(shb))
            b=b.reshape(shb)
            b_list=[]
            
            for rot in range(4):
                for flip in range(2):
                    w_mod=w;
                    w_mod=np.rot90(w_mod,rot,axes=(2,3))
                        
                    if flip==1:
                        w_mod=np.flip(w_mod,axis=2)
                        
                    w_list.append(w_mod)
                    
                    b_list.append(b)
                    
            w=np.concatenate(w_list,axis=0)  
            sh_w=np.shape(w)
            w=w.reshape(-1)
            
            b=np.concatenate(b_list,axis=0)  
            sh_b2=np.shape(b)
            b=b.reshape(-1)
            
            bb=bb.view(-1)
            bb=bb[b]
            bb=bb.view(sh_b2)
            bb=bb.contiguous()
            
            ww=ww.view(-1)
            ww=ww[w]
            ww=ww.view(sh_w)
            ww=ww.contiguous()
            
            if p4m:
                s = x.size()
                x = x.view(s[0], s[1]*s[2], s[3], s[4])
                x=F.conv2d(x, ww, bb, self.stride,self.padding, dil, self.groups)
                x =  x.view(s[0], int(x.size()[1]/8), 8, x.size()[2], x.size()[3])
            else:
                s = x.size()
                x=F.conv2d(x, ww,bb, self.stride,self.padding, dil, self.groups)
                x =  x.view(s[0], int(x.size()[1]/8), 8, x.size()[2], x.size()[3])
            
        else:
            x=F.conv2d(x, ww, b, self.stride,self.padding, dil, self.groups)  
            
        return x



class Conv2BnRelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=0,gconv=0,p4m=0):
        super().__init__()
        self.gconv=gconv
        self.p4m=p4m
        
        self.bn=1
        dov=0.05
    
        if p4m:
            self.conv=Conv2_muj(in_size*8, out_size,filter_size,stride,pad)
        else:
            self.conv=Conv2_muj(in_size, out_size,filter_size,stride,pad)
        
        if self.gconv:
            self.bn=nn.BatchNorm3d(out_size,momentum=0.1)
        else:
            self.bn=nn.BatchNorm2d(out_size,momentum=0.1)

        if self.gconv:
            self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout3d(dov))
        else:
            self.do=nn.Sequential(nn.Dropout(dov),nn.Dropout2d(dov))

    def forward(self, x,dil=1,dense=0):
        
        
        x = self.conv(x,dil,self.gconv,self.p4m)

        
        if self.bn:
#            if self.gconv:
#                s = x.size()
#                x = x.view(s[0], s[1]*s[2], s[3], s[4])
            x = self.bn(x)
#            if self.gconv:
#                x =  x.view(s[0], s[1], s[2], x.size()[2], x.size()[3])
            
         
        x=F.relu(x)
        x = self.do(x)

        return x
    
    
    
class PixelNet(nn.Module):
    def __init__(self,K=24,input_size=3,gconv=0,p4m=0):
        super().__init__()
        
        self.dense=0
        self.gconv=gconv
        
        self.initc=Conv2BnRelu(input_size, K,filter_size=1,gconv=gconv,p4m=0)

        self.c1_1=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c1_2=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t1=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)
        
        self.c2_1=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c2_2=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.t2=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)
       
        
        self.c3_1=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c3_2=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        self.c3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t3=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)

        self.c4=Conv2BnRelu(K, K,filter_size=5,gconv=gconv,p4m=1)

        self.finalc=nn.Conv2d(K, 1,1)
        
        
    def sum_remve_border(self,input1,input2,kolik=2):
        
        
        
        input1_small=input1[:, :,kolik:-kolik,kolik:-kolik]
        
        return input1_small+input2
#        return torch.cat([input1_small,input2],1)
        
    def sum_remve_border2(self,x,y1,y2,y3,dil):
        
        
        kolik=dil*1
        y2=y2[...,kolik:-kolik,kolik:-kolik]
        
        kolik=dil*2
        y1=y1[...,kolik:-kolik,kolik:-kolik]
        
        kolik=dil*3
        x=x[...,kolik:-kolik,kolik:-kolik]
        
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
        x,dil=maxpool_muj(x,3,2,dil,self.dense, self.gconv)#44/88
        
        
        y1=self.c2_1(x,dil,self.dense)#42/84
        y2=self.c2_2(y1,dil,self.dense)#40/80
        y3=self.c2_3(y2,dil,self.dense)#38/76
        x=self.sum_remve_border2(x,y1,y2,y3,dil)
        x=self.t2(x)
        x,dil=maxpool_muj(x,3,2,dil,self.dense,self.gconv)#18/72
        
        
        
        y1=self.c3_1(x,dil,self.dense)#16/64
        y2=self.c3_2(y1,dil,self.dense)#14/56
        y3=self.c3_3(y2,dil,self.dense)#12/48
        x=self.sum_remve_border2(x,y1,y2,y3,dil)
        x=self.t3(x)
        x,dil=maxpool_muj(x,3,2,dil,self.dense,self.gconv)#5/40
        
        
        
        
        x=self.c4(x,dil,self.dense)#1/8
        
        if self.gconv:
            x=final_pooling(x)
        
        
        x=self.finalc(x)
        
        
        return x
        
        
        
        
        