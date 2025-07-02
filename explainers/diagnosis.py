import torch
import torch.nn as nn
import numpy as np
from explainers.base import ExplainerBase
# from explainers.MILPverifier import FD_recon, DNN_recon

class CP(ExplainerBase):
    def __init__(self,model):
        super(CP, self).__init__(model, 'Contribution Plots')

    def explain(self, df, plot=True):
        x = df.values
        e = self.res_model(x)
        SPE = e * e
        # SPE = SPE / np.sum(SPE,axis=1,keepdims=True)
        if plot:
            self.plot_shap(SPE,df)
        return SPE

class RBC(ExplainerBase):
    def __init__(self, model):
        super(RBC, self).__init__(model, 'Reconstruction-based Contribution')

    def explain(self, df, plot=True):
        x = df.values
        e = self.res_model(x)
        SPE = e * e
        for m in self.model.children():
            for name,param in m.named_parameters():
                if 'weight' in name:
                    C = param.data.numpy()
        delta = np.eye(x.shape[1]) - C
        RBC = SPE / np.diagonal(delta)
        # RBC = RBC / np.sum(RBC,axis=1,keepdims=True)
        if plot:
            self.plot_shap(RBC,df)
        return RBC

class LLBBC(ExplainerBase):
    def __init__(self, model):
        super(LLBBC, self).__init__(model, 'LLBBC')

        self.act = 'tanh'
        self.weights = [j.numpy() for i,j in model.state_dict().items() if 'weight' in i]
        self.shapes = [j.numpy().shape for i,j in model.state_dict().items() if 'weight' in i]
        self.n_hid = len(model.encoder)

    def dev_sigmoid(self,x):
        y = x*(1-x)    
        out = np.diag(y)
        return out

    def dev_tanh(self,x):
        y = 1-x**2 
        out = np.diag(y)
        return out

    def inf(self, x_in, h_ref=0):
        # print(mode)
        x_in = torch.from_numpy(x_in).float()
        h_in = x_in
        h_out = []
        for l in range(self.n_hid):
            
            sub_model =  nn.Sequential(*list(self.model.children())[0][l])
            sub_out = sub_model(h_in)
            h_in = sub_out
            h_out.append(sub_out.detach().numpy())
        
        K_de = self.weights[-1]
  
        for l in range(self.n_hid-1, -1, -1):
            if self.act == 'tanh':
                K_de_mid = np.dot(self.dev_tanh(h_out[l]), self.weights[l])
                K_de = np.dot(K_de, K_de_mid)            
            
            elif self.act == 'sig':
                K_de_mid = np.dot(self.dev_sigmoid(h_out[l]), self.weights[l])
                K_de = np.dot(K_de, K_de_mid)            
                   
            else:
                raise ValueError("activate function error")

        K = np.identity(x_in.shape[0])-K_de
        M = np.dot(K.T, K)
        llbbc = np.zeros(x_in.shape[0])
        for i in range(x_in.shape[0]):
            llbbc[i] = np.dot(M[i,:], x_in)**2/M[i,i]
            
        return K_de, llbbc
    
    def explain(self, df, plot=True):
        x = df.values
        llbbc_res = np.zeros(shape = x.shape)
        for i in range(x.shape[0]):
            K_de, llr = self.inf(x[i,:])
            
            # 这里是归一化 后续用热度图做可视化
            llr = llr/np.sum(llr)
            llbbc_res[i,:] = llr
        if plot:
            self.plot_shap(llbbc_res,df)
        return llbbc_res
