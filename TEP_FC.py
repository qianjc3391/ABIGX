import numpy as np
import torch
from data import dataset
import torch.nn as nn
from models import AutoEncoder_FD, NN_FC
import pandas as pd
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from explainers.gradients import  VanillaGradient, IntegratedGradients, GuidedBackprop, IDGI
from explainers.diagnosis import CP, RBC, LLBBC
from explainers.abigx import ABIGX
TE_GROUND_TRUTH = {1:['XMV(4)','XMEAS(4)'],
                   2:['XMV(4)','XMEAS(4)'],
                   3:['XMV(1)','XMEAS(2)'],
                   4:['XMV(10)','XMEAS(21)'],
                   5:['XMEAS(22)'],
                   6:['XMEAS(1)','XMV(3)'],
                   7:['XMV(4)','XMEAS(4)'],
                   8:['XMEAS(18)','XMV(8)'],
                   9:['XMV(1)','XMEAS(2)'],
                   10:['XMEAS(18)'],
                   11:['XMV(10)','XMEAS(9)','XMEAS(21)'],
                   12:['XMEAS(22)'],
                   14:['XMV(10)','XMEAS(9)','XMEAS(21)'],
                   15:['XMV(11)']}


# Model
cl = ['XMEAS(%d)'%i for i in range(1,23)] + ['XMV(%d)'%i for i in range(1,12)]
# %%
model = torch.load('checkpoints/TE/NN_FC.pkl',weights_only=False)
model.eval()
for module in model.modules():
    if isinstance(module, nn.ReLU):
        module.inplace = False
trainset = dataset.TE_identification('data',train=True,normalized=False,train_mode='all')
trainset_fd = dataset.TE_identification('data',train=True,normalized=False,train_mode='normal')
testset = dataset.TE_identification('data',train=False,normalized=False)
normality = trainset.data[trainset.targets==0,:].copy()
correct = torch.argmax(model(torch.FloatTensor(normality)),dim=1) == 0
normality = normality[correct]
normality_mean = torch.FloatTensor(np.mean(normality,axis=0,keepdims=True))
model(normality_mean)
print(np.mean(trainset.data[trainset.targets==0,:],axis=0,keepdims=True))
print(np.mean(testset.data[testset.targets==0,:],axis=0,keepdims=True))
print(np.mean(trainset_fd.data[trainset_fd.targets==0,:],axis=0,keepdims=True))

results = {}
black = np.zeros_like(testset.data[testset.targets==0,:][-1])
for Fault_ID in range(1,16):
    print(Fault_ID)
    if Fault_ID == 13:
        continue
    explicand = testset.data[testset.targets==Fault_ID,:]
    fault_explicand = pd.DataFrame(data=explicand,columns=cl)

    attrs = {}
    # ABIGX
    explicand_y = torch.LongTensor(testset.targets[testset.targets==Fault_ID])
    e = ABIGX(model,normality,'cp_pgd',{'nb_iter':3000,'eps_iter':0.05})
    attr, afr_baseline = e.explain(fault_explicand,explicand_y.data,plot=False)
    attrs[e.method_name]=attr

    results[Fault_ID] = attrs

#%%

for Fault_ID in range(1,16):
    if Fault_ID == 13:
        continue
    print(Fault_ID)

    explicand = testset.data[testset.targets==Fault_ID,:]

    fault_explicand = pd.DataFrame(data=explicand,columns=cl)
    for method in results[Fault_ID].keys():
        # method = 'ABIGX'
        res = results[Fault_ID][method]
        gt_var = TE_GROUND_TRUTH[Fault_ID]
        gt_idx = [cl.index(i) for i in gt_var]

        attr = np.mean(np.abs(res),axis=0)
        idx = np.argsort(attr)[::-1][:8]
        print([cl[i]for i in idx])

        shap.plots.violin(res, fault_explicand, max_display=4,
                        show=False, color_bar=False)
        # Get the current summary plot's y-tick labels and color ground truth variables red if present
        ax = plt.gca()
        y_ticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
        for i, label in enumerate(y_ticklabels):
            if label in gt_var:
                ax.get_yticklabels()[i].set_color('red')
        ax.set_xlabel(f'{method}', fontsize=15)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=15)
        ax.set_title(f'Fault {Fault_ID}', fontsize=16)
        plt.tight_layout()
        plt.show()
        # plt.close()