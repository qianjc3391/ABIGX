# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 19:46:07 2022

@author: Jacob
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from scipy.special import ndtr

class AutoEncoder_FD(nn.Module):
    def __init__(self):
        super(AutoEncoder_FD, self).__init__()

        self.ae = nn.Sequential(
            nn.Linear(33,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,20),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,33),
        )

    def forward(self, x):
        xhat = self.ae(x)
        return xhat

    def cal_q(self, data):
        x_hat = self.ae(torch.tensor(data).float())
        sqe=(torch.tensor(data)-x_hat)**2
        sqe=sqe.detach().numpy()
        q = np.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        mean_q = np.mean(q_normal)
        std_q = np.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.95,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     idx = (np.abs(array - value)).argmin()
        #     return idx
        # kde = stats.gaussian_kde(q_normal)
        # x = np.linspace(0, 50, 100)
        # cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
        #             for item in x)
        # ind = find_nearest(cdf,0.99)
        # q_limit = x[ind]
        return q_limit

class SingleAutoEncoder(nn.Module):
    def __init__(self, n_in, n_hid):
        super().__init__()
        
        self.n_in = n_in
        self.n_hid = n_hid
        
        self.encoder = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.Tanh()
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(n_hid, n_in)
            )
        
    def forward(self, x_in, return_h_out=False):
        h_out = self.encoder(x_in)   
        x_rec = self.decoder(h_out)
        if not return_h_out:
            return x_rec
        else:
            return h_out, x_rec

    def cal_q(self, data):
        x_hat = self.decoder(self.encoder(
                torch.tensor(data).float()))
        sqe=(torch.tensor(data)-x_hat)**2
        sqe=sqe.detach().numpy()
        q = np.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        mean_q = np.mean(q_normal)
        std_q = np.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.95,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     idx = (np.abs(array - value)).argmin()
        #     return idx
        # kde = stats.gaussian_kde(q_normal)
        # x = np.linspace(0, 50, 100)
        # cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
        #             for item in x)
        # ind = find_nearest(cdf,0.99)
        # q_limit = x[ind]
        return q_limit

class StackedAutoEncoder(nn.Module):
    def __init__(self, n_in, n_hid = [100,101]):
        super().__init__()
        
        self.n_in = n_in
        self.n_hid = n_hid
        self.encoder = nn.Sequential()
        
        
        self.encoder.add_module(str(0), nn.Sequential(nn.Linear(n_in, n_hid[0]),nn.Tanh()))
                      
        for l in range(len(n_hid)-1):
            self.encoder.add_module(str(l+1), nn.Sequential(nn.Linear(n_hid[l],n_hid[l+1]),nn.Tanh()))
 
        self.decoder = nn.Sequential(
            nn.Linear(n_hid[-1], n_in)
            )
        
    def forward(self, x_in, return_h_out=False):
        h_out = self.encoder(x_in)   
        x_rec = self.decoder(h_out)
        if not return_h_out:
            return x_rec
        else:
            return h_out, x_rec

    def cal_q(self, data):
        x_hat = self.decoder(self.encoder(
                torch.tensor(data).float()))
        sqe=(torch.tensor(data)-x_hat)**2
        sqe=sqe.detach().numpy()
        q = np.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        mean_q = np.mean(q_normal)
        std_q = np.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.95,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     idx = (np.abs(array - value)).argmin()
        #     return idx
        # kde = stats.gaussian_kde(q_normal)
        # x = np.linspace(0, 50, 100)
        # cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
        #             for item in x)
        # ind = find_nearest(cdf,0.99)
        # q_limit = x[ind]
        return q_limit

class pca(nn.Module):
    def __init__(self,x_dim):
        super(pca, self).__init__()

        self.C = nn.Sequential(
            nn.Linear(x_dim,x_dim)
        )

    def forward(self, x):
        xhat = self.C(x)
        return xhat

    def cal_q(self, data):
        x_hat = self.C(torch.tensor(data).float())
        sqe=(torch.tensor(data)-x_hat)**2
        sqe=sqe.detach().numpy()
        q = np.sum(sqe,axis = 1)
        return q

    def cal_limit(self, q_normal):
        mean_q = np.mean(q_normal)
        std_q = np.std(q_normal)**2
        freedom = (2*(mean_q**2))/std_q
        chi_lim = stats.chi2.ppf(0.95,freedom)
        q_limit = std_q/(2*mean_q)*chi_lim
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     idx = (np.abs(array - value)).argmin()
        #     return idx
        # kde = stats.gaussian_kde(q_normal)
        # x = np.linspace(0, 50, 100)
        # cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
        #             for item in x)
        # ind = find_nearest(cdf,0.99)
        # q_limit = x[ind]
        return q_limit

class NN_FC(nn.Module):
    def __init__(self):
        super(NN_FC, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(33, 64),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.15,inplace=False),
            )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(0.15,inplace=False),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(64, 16)
            )
        self.weight_init()

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        return self.output_layer(X)

    def get_layer_output(self, X, n_layer=-2):
        out1 = self.layer1(X)
        out2 = self.layer2(out1)
        out3 = self.output_layer(out2)
        outs = [out1, out2, out3]
        return outs[n_layer]


    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())


def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()