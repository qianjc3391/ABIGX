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
model = torch.load('checkpoints/TE/StackedAE_FD.pkl', weights_only=False)
model.eval()
trainset = dataset.TE_identification('data',train=True,normalized=True,train_mode='normal')
testset = dataset.TE_identification('data',train=False,normalized=True)

black = np.zeros_like(testset.data[testset.targets==0,:][-1])
q_norm = model.cal_q(trainset.data)
q_limit = model.cal_limit(q_norm)

results = {}
for Fault_ID in range(1,16):
    if Fault_ID == 13:
        continue
    explicand = testset.data[testset.targets==Fault_ID,:]
    fault_explicand = pd.DataFrame(data=explicand,columns=cl)
    attrs = {}
    # ABIGX
    bigx = ABIGX(model,trainset.data,'cp_pgd',{'nb_iter':3000,'eps_iter':0.01})
    attr, afr_baseline = bigx.explain(fault_explicand,plot=False)
    attrs['ABIGX']=attr
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