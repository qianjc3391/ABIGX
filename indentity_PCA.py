import numpy as np
import torch
from data import dataset
import pandas as pd
from explainers.diagnosis import CP, RBC
from explainers.abigx import ABIGX
# MILP reconstruction requires gurobipy and cvxpy installed
# from explainers.MILPverifier import FD_recon, DNN_recon
import time
trainset = dataset.TE_identification('data',train=True,normalized=True,train_mode='normal')
testset = dataset.TE_identification('data',train=False,normalized=True)

attr_cps = []
attr_cppgds = []
attr_cpmilps = []

Fault_ID = 1
explicand = testset.data[testset.targets==Fault_ID,:]
explicand_y = testset.targets[testset.targets==Fault_ID]
cl = ['XMEAS(%d)'%i for i in range(1,23)] + ['XMV(%d)'%i for i in range(1,12)]
fault_explicand =  pd.DataFrame(data=explicand,columns=cl)
# Model

model = torch.load('checkpoints/TE/PCA_FD.pkl', weights_only=False)
model.eval()

# explainer = FD_recon(model, 0, -20, 20, 40, norm=norm, mode='min_SPE')
start_time = time.time()
exp = CP(model)
attr_CP = exp.explain(fault_explicand)
attr_cps.append(attr_CP)
print('CP time:', time.time() - start_time)

start_time = time.time()
bigx = ABIGX(model,trainset.data,'cp_pgd',{'nb_iter':3000,'eps_iter':5e-3})
attr_cpx, afr_baseline = bigx.explain(fault_explicand)
attr_cppgds.append(attr_cpx)
print('CP-PGD time:', time.time() - start_time)

# bigx = ABIGX(model,trainset.data,'cp_milp',{'nb_iter':3000,'eps_iter':5e-3})
# attr_cpmilp, afr_baseline = bigx.explain(fault_explicand)
# attr_cpmilps.append(attr_cpmilp)


attr_cps = np.array(attr_cps)
attr_cppgds = np.array(attr_cppgds)
print("Diff:", np.mean(abs(attr_cps - attr_cppgds)))


# attr_cpmilps = np.array(attr_cpmilps)
# print("Diff:", np.mean(abs(attr_cps - attr_cpmilps)))

#%%
attr_cps = []
attr_cppgds = []
attr_cpmilps = []

Fault_ID = 1 # Simply change the Fault_ID to test other faults

explicand = testset.data[testset.targets==Fault_ID,:]
explicand_y = testset.targets[testset.targets==Fault_ID]
cl = ['XMEAS(%d)'%i for i in range(1,23)] + ['XMV(%d)'%i for i in range(1,12)]
fault_explicand =  pd.DataFrame(data=explicand,columns=cl)

# explainer = FD_recon(model, 0, -20, 20, 40, norm=norm, mode='min_SPE')
start_time = time.time()
exp = RBC(model)
attr = exp.explain(fault_explicand)
attr_cps.append(attr)
print('RBC time:', time.time() - start_time)

bigx = ABIGX(model,trainset.data,'rbc_pgd',{'nb_iter':20000,'eps_iter':1})
attr_cpx, afr_baseline = bigx.explain(fault_explicand)
attr_cppgds.append(attr_cpx)


# bigx = ABIGX(model,trainset.data,'rbc_milp',{'nb_iter':3000,'eps_iter':5e-3})
# attr_cpmilp, afr_baseline = bigx.explain(fault_explicand)
# attr_cpmilps.append(attr_cpmilp)


attr_cps = np.array(attr_cps)
attr_cppgds = np.array(attr_cppgds)
print('Diff:', np.mean(abs(attr_cps - attr_cppgds)))

# attr_cpmilps = np.array(attr_cpmilps)
# print(np.mean(abs(attr_cps - attr_cpmilps)))
