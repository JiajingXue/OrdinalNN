import time
from Functions import macro_auc, rps_score, accuracy, getTPFP
from Proposed import Train_model
from DataGeneratingProcess import DGP
import numpy as np
import torch

epoch_proposed = 1000

ridge_p = 0.03
lam_p = 0.015

rho, dim, setting, proportion = 0.0, 100, 'nonlinear1', 'eq'

T1 = time.time()

###### datasets

y_test,  x_test,  z_test,  e_test,  label_test = DGP(10001, 200, rho, dim, setting, proportion)
y_valid, x_valid, z_valid, e_valid, label_valid= DGP(10000, 200, rho, dim, setting, proportion)

y_test, x_test, z_test, e_test =torch.Tensor(y_test),torch.Tensor(x_test),torch.Tensor(z_test),torch.Tensor(e_test)
y_valid,x_valid,z_valid,e_valid=torch.Tensor(y_valid),torch.Tensor(x_valid),torch.Tensor(z_valid),torch.Tensor(e_valid)

n = 700
seed = 42

###### one example

y_train, x_train, z_train, e_train, label_train = DGP(seed, n, rho, dim, setting, proportion)
y_train, x_train, z_train, e_train = torch.Tensor(y_train), torch.Tensor(x_train), \
    torch.Tensor(z_train), torch.Tensor(e_train)

proposed = Train_model(x_train, z_train, e_train, y_train, x_valid, z_valid, e_valid, y_valid,
                       ridge=ridge_p, Lam=lam_p, Num_Epochs=epoch_proposed, xi=3)[0]

auc_p, rps_p, acc_p = macro_auc(proposed, x_test, z_test, e_test, label_test),\
    rps_score(proposed, x_test, z_test, e_test, label_test),\
    accuracy(proposed, x_test, z_test, e_test, label_test)

main_p_tp, main_p_fp = getTPFP(proposed)[0], getTPFP(proposed)[1]
inte_p_tp, inte_p_fp = getTPFP(proposed)[2], getTPFP(proposed)[3]

T2 = time.time()

print('cost:', T2-T1)
print('prediction', auc_p, rps_p, acc_p)
print('variable selection', main_p_tp, main_p_fp, inte_p_tp, inte_p_fp)
