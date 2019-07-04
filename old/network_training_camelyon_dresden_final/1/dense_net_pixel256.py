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
            x=F.conv2d(x, self.weight, self.bias, self.stride,self.padding, dil, self.groups)  
            
        return x



class Conv2BnRelu(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=0,gconv=0,p4m=0):
        super().__init__()
        self.gconv=gconv
        self.p4m=p4m
        
        self.bn=1
        dov=0.05
    
        if p4m and gconv:
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
            x = self.bn(x)
        x=F.relu(x)
#        x = self.do(x)

        return x
    
    
    
class PixelNet(nn.Module):
    def __init__(self,k0=16,k=12,input_size=3,gconv=0,p4m=0,return_features=0):
        super().__init__()
        
        self.return_features=return_features
        self.dense_down=4
        
        self.gconv=gconv
        
        K=k0
        
        self.initc=Conv2BnRelu(input_size, K,filter_size=1,gconv=gconv,p4m=0)

        
        
        self.c1_1_1=Conv2BnRelu(K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c1_1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c1_2_1=Conv2BnRelu(2*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c1_2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c1_3_1=Conv2BnRelu(3*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c1_3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t1=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)
        
        
        
        K_old=K
        K+=k
        
        
        self.c2_1_1=Conv2BnRelu(K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c2_1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c2_2_1=Conv2BnRelu(K+K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c2_2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c2_3_1=Conv2BnRelu(2*K+K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c2_3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t2=Conv2BnRelu(3*K+K_old, K,filter_size=1,gconv=gconv,p4m=1)
        
        
        
       
        K_old=K
        K+=k
        
        self.c_add=Conv2BnRelu(K_old, K,gconv=gconv,p4m=1)
        
        
        
        self.c3_1_1=Conv2BnRelu(K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c3_1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c3_2_1=Conv2BnRelu(2*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c3_2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c3_3_1=Conv2BnRelu(3*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c3_3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t3=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)


        K_old=K
        K+=k
        
        self.c_add2=Conv2BnRelu(K_old, K,gconv=gconv,p4m=1)

        self.c4_1_1=Conv2BnRelu(K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c4_1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c4_2_1=Conv2BnRelu(2*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c4_2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c4_3_1=Conv2BnRelu(3*K, K,filter_size=1,gconv=gconv,p4m=1)
        self.c4_3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t4=Conv2BnRelu(4*K, K,filter_size=1,gconv=gconv,p4m=1)
        
        K_old=K
        K+=k

        



        self.c5_1_1=Conv2BnRelu(K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c5_1_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c5_2_1=Conv2BnRelu(K+K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c5_2_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)
        
        self.c5_3_1=Conv2BnRelu(2*K+K_old, K,filter_size=1,gconv=gconv,p4m=1)
        self.c5_3_3=Conv2BnRelu(K, K,gconv=gconv,p4m=1)

        self.t5=Conv2BnRelu(3*K+K_old, K,filter_size=1,gconv=gconv,p4m=1)

        
            
        self.finalc1=nn.Conv2d(K, 32,1)
        self.finalc2=nn.Conv2d(32, 1,1)
        
        return
        
        
        
    def cat_remove_border(self,l,dil):
        l=l[::-1]
        
        xx=l[0]
        for i,x in enumerate(l[1:]):
            kolik=dil*(i+1)
            x=x[...,kolik:-kolik,kolik:-kolik]
            xx=torch.cat([xx,x],1)

        return xx
    
    
    
    
    def dense_on(self,dense_down=4):
        self.dense_down=dense_down
    
    def dense_off(self):
        self.dense_down=4
        
    def forward(self, x):
        
        dense=0
        dil=1
        
        
        
        x=self.initc(x,dil,dense)#256
        
        if self.dense_down==0:
            dense=1
        
        tmp =self.c1_1_1(x,dil,dense)
        y1=self.c1_1_3(tmp,dil,dense)#254
        tmp=self.c1_2_1(self.cat_remove_border((x,y1),dil),dil,dense)
        y2=self.c1_2_3(tmp,dil,dense)#252
        tmp=self.c1_3_1(self.cat_remove_border((x,y1,y2),dil),dil,dense)
        y3=self.c1_3_3(tmp,dil,dense)#250
        tmp=self.t1(self.cat_remove_border((x,y1,y2,y3),dil))
        x,dil=maxpool_muj(tmp,3,2,dil,dense, self.gconv)#124
        
        
        if self.dense_down==1:
            dense=1
        
        tmp =self.c2_1_1(x,dil,dense)
        y1=self.c2_1_3(tmp,dil,dense)
        tmp=self.c2_2_1(self.cat_remove_border((x,y1),dil),dil,dense)
        y2=self.c2_2_3(tmp,dil,dense)
        tmp=self.c2_3_1(self.cat_remove_border((x,y1,y2),dil),dil,dense)
        y3=self.c2_3_3(tmp,dil,dense)
        tmp=self.t2(self.cat_remove_border((x,y1,y2,y3),dil))
        x,dil=maxpool_muj(tmp,3,2,dil,dense, self.gconv)#58
        
        
        
        if self.dense_down==2:
            dense=1
        
        x=self.c_add(x,dil,dense)#56/224 add
        
        tmp =self.c3_1_1(x,dil,dense)
        y1=self.c3_1_3(tmp,dil,dense)
        tmp=self.c3_2_1(self.cat_remove_border((x,y1),dil),dil,dense)
        y2=self.c3_2_3(tmp,dil,dense)
        tmp=self.c3_3_1(self.cat_remove_border((x,y1,y2),dil),dil,dense)
        y3=self.c3_3_3(tmp,dil,dense)
        tmp=self.t3(self.cat_remove_border((x,y1,y2,y3),dil))
        x,dil=maxpool_muj(tmp,3,2,dil,dense, self.gconv)#24
        
        
        if self.dense_down==3:
            dense=1
        
        x=self.c_add2(x,dil,dense)
        
        tmp =self.c4_1_1(x,dil,dense)
        y1=self.c4_1_3(tmp,dil,dense)#20
        tmp=self.c4_2_1(self.cat_remove_border((x,y1),dil),dil,dense)
        y2=self.c4_2_3(tmp,dil,dense)#18
        tmp=self.c4_3_1(self.cat_remove_border((x,y1,y2),dil),dil,dense)
        y3=self.c4_3_3(tmp,dil,dense)#16
        tmp=self.t4(self.cat_remove_border((x,y1,y2,y3),dil))
        x,dil=maxpool_muj(tmp,3,2,dil,dense, self.gconv)#7
        
        
        
        tmp =self.c5_1_1(x,dil,dense)
        y1=self.c5_1_3(tmp,dil,dense)#122
        tmp=self.c5_2_1(self.cat_remove_border((x,y1),dil),dil,dense)
        y2=self.c5_2_3(tmp,dil,dense)#120
        tmp=self.c5_3_1(self.cat_remove_border((x,y1,y2),dil),dil,dense)
        y3=self.c5_3_3(tmp,dil,dense)
        x=self.t5(self.cat_remove_border((x,y1,y2,y3),dil))#3

        
        
        if self.gconv:
            x=final_pooling(x)
        
        
        f=self.finalc1(x)
        x=self.finalc2(f)
        
        if self.return_features:
            return x,f
        
        else:
            
            return x
        
        
        
        
        