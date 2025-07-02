import shap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

class ExplainerBase():
    def __init__(self, model, method_name):
        self.method_name = method_name
        self.model = model
        self.model.eval()
        def res(x):
            if not torch.is_tensor(x):
               x = torch.from_numpy(x).float()
            return (x - self.model(x)).detach().numpy()
        self.res_model = res
        if 'pca' in str(type(model)) or 'AutoEncoder' in str(type(model)):
            self.model_type = 'FD'
        if 'FC' in str(type(model)):
            self.model_type = 'FC'

    def prediction(self,x,y=0):      
        if self.model_type in ['FD']:
            x = x.detach().clone()
            x.requires_grad = True
            x_hat = self.model(x)
            squared_diff = (x - x_hat) ** 2
            SPE = squared_diff.mean(dim=1)
            return SPE
        else:
            x = x.detach().clone()
            x.requires_grad = True
            output = self.model(x)
        return output.gather(1,y.view(-1,1)).squeeze(1)
    
    def plot_shap(self, attr, explicand):
        # plt.figure()
        shap.summary_plot(attr, explicand, max_display=8)
        # plt.xlabel(self.method_name)
        # plt.tight_layout()


