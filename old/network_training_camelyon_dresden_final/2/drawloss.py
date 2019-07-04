import visdom
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


class DisplayLosses():
    def __init__(self,evaluate_it=10):
        self.it=-1
        self.it_test=-1
        self.evaluate_it=evaluate_it
        self.vis=visdom.Visdom()
        
        self.X_train=np.array([],dtype=np.float64)
        self.X_test=np.array([],dtype=np.float64)
        
        self.loss_window=self.vis.line(Y=np.array([0]),name='loss_train')
        time.sleep(0.01)
        self.acc_window=self.vis.line(Y=np.array([0]),name='acc_train')
        time.sleep(0.01)
        self.dice_window=self.vis.line(Y=np.array([0]),name='dice_train')
        time.sleep(0.01)
        self.auc_window=self.vis.line(Y=np.array([0]),name='auc_train')
        time.sleep(0.01)
        self.sen_window=self.vis.line(Y=np.array([0]),name='sen_train')
        time.sleep(0.01)
        self.spe_window=self.vis.line(Y=np.array([0]),name='spe_train')
        time.sleep(0.01)
        
        
        time.sleep(0.01)
        self.loss_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='loss_test',win=self.loss_window, update='append')
        time.sleep(0.01)
        self.acc_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='acc_test',win=self.acc_window, update='append')
        time.sleep(0.01)
        self.dice_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='dice_test',win=self.dice_window, update='append')
        time.sleep(0.01)
        self.auc_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='auc_test',win=self.auc_window, update='append')
        time.sleep(0.01)
        self.sen_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='sen_test',win=self.sen_window, update='append')
        time.sleep(0.01)
        self.spe_window=self.vis.line(X=np.array([0]),Y=np.array([0]),name='spe_test',win=self.spe_window, update='append')
        time.sleep(0.01)
        
        
        
        self.loss_tmp=np.array([],dtype=np.float64)
        self.gt_tmp=np.array([],dtype=np.float64)
        self.res_tmp=np.array([],dtype=np.float64)
        
        self.TP=np.array([],dtype=np.float64)
        self.TN=np.array([],dtype=np.float64)
        self.FP=np.array([],dtype=np.float64)
        self.FN=np.array([],dtype=np.float64)
        
        self.loss=np.array([],dtype=np.float64)
        self.acc=np.array([],dtype=np.float64)
        self.dice=np.array([],dtype=np.float64)
        self.auc=np.array([],dtype=np.float64)
        self.sen=np.array([],dtype=np.float64)
        self.spe=np.array([],dtype=np.float64)
        
        
        
        
        self.loss_tmp_test=np.array([],dtype=np.float64)
        self.gt_tmp_test=np.array([],dtype=np.float64)
        self.res_tmp_test=np.array([],dtype=np.float64)
        
        self.TP_test=np.array([],dtype=np.float64)
        self.TN_test=np.array([],dtype=np.float64)
        self.FP_test=np.array([],dtype=np.float64)
        self.FN_test=np.array([],dtype=np.float64)
        
        self.loss_test=np.array([],dtype=np.float64)
        self.acc_test=np.array([],dtype=np.float64)
        self.dice_test=np.array([],dtype=np.float64)
        self.auc_test=np.array([],dtype=np.float64)
        self.sen_test=np.array([],dtype=np.float64)
        self.spe_test=np.array([],dtype=np.float64)
        
        

    def update_train(self,loss,gt,res,lr):
        self.it+=1
        
        self.loss_tmp=np.append(self.loss_tmp,loss)
        self.gt_tmp=np.append(self.gt_tmp,gt)
        self.res_tmp=np.append(self.res_tmp,res)
       
        
        if self.it%self.evaluate_it==0 and self.it!=0:
            
            
            
            TP=np.array(np.sum(np.bitwise_and(self.gt_tmp>0,self.res_tmp>0.5)))
            TN=np.array(np.sum(np.bitwise_and(self.gt_tmp==0,self.res_tmp<=0.5)))
            FP=np.array(np.sum(np.bitwise_and(self.gt_tmp==0,self.res_tmp>0.5)))
            FN=np.array(np.sum(np.bitwise_and(self.gt_tmp>0,self.res_tmp<=0.5)))
            
            auc=roc_auc_score(self.gt_tmp,self.res_tmp)
            
            
            self.TP=np.append(self.TP,TP)
            self.TN=np.append(self.TN,TN)
            self.FP=np.append(self.FP,FP)
            self.FN=np.append(self.FN,FN)
            self.loss=np.append(self.loss,np.mean(self.loss_tmp))
            self.acc=np.append(self.acc,(TP+TN)/(TP+FP+FN+TN))
            self.dice=np.append(self.dice,2*TP/(2*TP+FP+FN))
            self.auc=np.append(self.auc,auc)
            self.sen=np.append(self.sen,TP/(TP+FN))
            self.spe=np.append(self.spe,TN/(TN+FP))
            
            self.X_train=np.append(self.X_train,self.it)
            
            
            self.loss_tmp=np.array([],dtype=np.float64)
            self.gt_tmp=np.array([],dtype=np.float64)
            self.res_tmp=np.array([],dtype=np.float64)
        
            self.loss_window=self.vis.line(Y=np.reshape(np.array(self.loss[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.loss_window, update='append',name='loss_train',opts=dict(title='LOSS'))
            time.sleep(0.01)
            self.acc_window=self.vis.line(Y=np.reshape(np.array(self.acc[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.acc_window, update='append',name='acc_train',opts=dict(title='ACC'))
            time.sleep(0.01)
            self.dice_window=self.vis.line(Y=np.reshape(np.array(self.dice[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.dice_window, update='append',name='dice_train',opts=dict(title='DICE'))
            time.sleep(0.01)
            self.auc_window=self.vis.line(Y=np.reshape(np.array(self.auc[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.auc_window, update='append',name='auc_train',opts=dict(title='AUC'))
            time.sleep(0.01)
            self.sen_window=self.vis.line(Y=np.reshape(np.array(self.sen[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.sen_window, update='append',name='sen_train',opts=dict(title='SEN'))
            time.sleep(0.01)
            self.spe_window=self.vis.line(Y=np.reshape(np.array(self.spe[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.spe_window, update='append',name='spe_train',opts=dict(title='SPE'))
            time.sleep(0.01)
            
            
            print(str(self.it) + '   auc:' + str(auc) + '  lr:' + str(lr))

        
    def update_test(self,loss,gt,res):
        self.it_test+=1
    
        self.loss_tmp_test=np.append(self.loss_tmp_test,loss)
        self.gt_tmp_test=np.append(self.gt_tmp_test,gt)
        self.res_tmp_test=np.append(self.res_tmp_test,res)
    
    def draw_test(self):

        
        
        TP=np.sum(np.bitwise_and(self.gt_tmp_test>0,self.res_tmp_test>0.5))
        TN=np.sum(np.bitwise_and(self.gt_tmp_test==0,self.res_tmp_test<=0.5))
        FP=np.sum(np.bitwise_and(self.gt_tmp_test==0,self.res_tmp_test>0.5))
        FN=np.sum(np.bitwise_and(self.gt_tmp_test>0,self.res_tmp_test<=0.5))
        
        auc=roc_auc_score(self.gt_tmp_test,self.res_tmp_test)
        
        
        self.TP_test=np.append(self.TP_test,TP)
        self.TN_test=np.append(self.TN_test,TN)
        self.FP_test=np.append(self.FP_test,FP)
        self.FN_test=np.append(self.FN_test,FN)
        self.loss_test=np.append(self.loss_test,np.mean(self.loss_tmp_test))
        self.acc_test=np.append(self.acc_test,(TP+TN)/(TP+FP+FN+TN))
        self.dice_test=np.append(self.dice_test,2*TP/(2*TP+FP+FN))
        self.auc_test=np.append(self.auc_test,auc)
        self.sen_test=np.append(self.sen_test,TP/(TP+FN))
        self.spe_test=np.append(self.spe_test,TN/(TN+FP))
        
        self.X_test=np.append(self.X_test,self.it)
        
        
        self.loss_tmp_test=np.array([],dtype=np.float64)
        self.gt_tmp_test=np.array([],dtype=np.float64)
        self.res_tmp_test=np.array([],dtype=np.float64)
    
        self.loss_window=self.vis.line(Y=np.reshape(np.array(self.loss_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.loss_window, update='append',name='loss_test',opts=dict(title='LOSS'))
        time.sleep(0.01)
        self.acc_window=self.vis.line(Y=np.reshape(np.array(self.acc_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.acc_window, update='append',name='acc_test',opts=dict(title='ACC'))
        time.sleep(0.01)
        self.dice_window=self.vis.line(Y=np.reshape(np.array(self.dice_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.dice_window, update='append',name='dice_test',opts=dict(title='DICE'))
        time.sleep(0.01)
        self.auc_window=self.vis.line(Y=np.reshape(np.array(self.auc_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.auc_window, update='append',name='auc_test',opts=dict(title='AUC'))
        time.sleep(0.01)
        self.sen_window=self.vis.line(Y=np.reshape(np.array(self.sen_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.sen_window, update='append',name='sen_test',opts=dict(title='SEN'))
        time.sleep(0.01)
        self.spe_window=self.vis.line(Y=np.reshape(np.array(self.spe_test[-1]),(1)),X=np.reshape(np.array(self.it),(1)),win=self.spe_window, update='append',name='spe_test',opts=dict(title='SPE'))
        time.sleep(0.01)
        
        
        
    def get_auc_test_last(self):
        return self.auc_test[-1]
    
    def get_data(self):
        return np.stack([self.X_train,self.loss,self.acc,self.dice,self.auc,self.sen,self.spe],axis=1),np.stack([self.X_test,self.loss_test,self.acc_test,self.dice_test,self.auc_test,self.sen_test,self.spe_test],axis=1)
    
    def save_plots(self,path):
        plt.plot(self.X_train,self.loss,'b')
        plt.plot(self.X_test,self.loss_test,'r')
        plt.title('loss')
        plt.savefig(path +'\\loss.png')
        time.sleep(0.01)
        plt.clf()
        
        plt.plot(self.X_train,self.acc,'b')
        plt.plot(self.X_test,self.acc_test,'r')
        plt.title('acc')
        plt.savefig(path +'\\acc.png')
        time.sleep(0.01)
        plt.clf()
        
        plt.plot(self.X_train,self.dice,'b')
        plt.plot(self.X_test,self.dice_test,'r')
        plt.title('dice')
        plt.savefig(path +'\\dice.png')
        time.sleep(0.01)
        plt.clf()
        
        plt.plot(self.X_train,self.auc,'b')
        plt.plot(self.X_test,self.auc_test,'r')
        plt.title('auc')
        plt.savefig(path +'\\auc.png')
        time.sleep(0.01)
        plt.clf()
        
        
        plt.plot(self.X_train,self.sen,'b')
        plt.plot(self.X_test,self.sen_test,'r')
        plt.title('sen')
        plt.savefig(path +'\\sen.png')
        time.sleep(0.01)
        plt.clf()
        
        plt.plot(self.X_train,self.spe,'b')
        plt.plot(self.X_test,self.spe_test,'r')
        plt.title('spe')
        plt.savefig(path +'\\spe.png')
        time.sleep(0.01)
        plt.clf()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        