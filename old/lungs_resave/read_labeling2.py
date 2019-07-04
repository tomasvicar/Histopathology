
import numpy as np

#path=r'C:\Users\Tom\Desktop\deep_drazdany\histolky\data\scribles\train\2017_11_30__0034-4.czi.labeling'
##label_dict=dict(??=0, solid=1, diffuse=1,tum=2,nontum=3)
#
#label_dict=	{'??': 0, 'solid':1,'diffuse':1,'tum':2,'nontum':3}

def stringlist_to_list(str0):
    
    if len(str0)>2:
        strs=str0[1:-1].split('],[')
        res=np.array([[1,2]])
        for str in strs:
            tmp=str.split(',')
            a=np.array([[int(tmp[0]),int(tmp[1])]])
            res= np.append(res,a,axis=0)
        res=res[1:,:]
    else:
       res=np.array([[]],dtype=np.int)
        
        
    return res

def close_bracket(message,start):
    stav=1
    k=0
    for ch in message[start+1:]:
        k+=1
        if ch=='[':
            stav=stav+1
        if ch==']':
            stav=stav-1
        
        if stav==0:
            return start+k+1
        
        
    assert('zavorka nekonci')
            
        



def read_lbl2(path,label_dict):
    f = open(path,'r')
    message = f.read().lower()
    
    tmp="\"max\":"
    m1=message.find(tmp)
    if m1==-1:
        raise NameError('velikost nenalezena')
    m2=message.find("]",m1+len(tmp))
    velikost=message[m1+len(tmp):m2+1]
    velikost=stringlist_to_list(velikost)
    tmp=list(velikost[0,:]+1);
    img=np.zeros(tmp,dtype=np.uint8)
    

    
    for key, value in label_dict.items():
        pos=message.find(key)
        if pos>-1:
            start=message.find("[",pos)
            end=close_bracket(message,start)
        
            data=message[start+1:end-1]
            data=stringlist_to_list(data)
            if np.shape(data)[1]>0:
                img[data[:,0],data[:,1]]=value
        else:
            assert('prazdne')
        
    
    return img.T
    
#lbl=read_lbl2(path,label_dict)    