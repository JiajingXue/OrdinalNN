import time
from Functions import macro_auc, rps_score, accuracy, getTPFP
from Proposed import Train_model
from DataGeneratingProcess import DGP
import numpy as np
import torch

## metrics ##
def macro_auc(trained_net, test_main, test_inter, test_envi, label_test):

    
    label_test = np.array(label_test).astype(int).squeeze()
    if label_test.ndim > 1:
        label_test = label_test.reshape(-1)

    with torch.no_grad():
        logits = trained_net.forward(test_main, test_inter, test_envi)
        
        if logits.dim() > 2:
            logits = logits.squeeze()
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    
    if probs.ndim > 2:
        probs = probs.squeeze()
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T

    n_classes = probs.shape[1]

    # one-hot
    label_onehot = np.zeros((len(label_test), n_classes))
    label_onehot[np.arange(len(label_test)), label_test] = 1

    # macro AUC
    return roc_auc_score(
        y_true=label_onehot,
        y_score=probs,
        average='macro',
        multi_class='ovr'
    )

def rps_score(net, test_main, test_inter, test_envi, label_test):
    """
    Ranked Probability Score (RPS)

    params:
        net: well trained NN
        test_main: main features in test set (torch.Tensor)
        test_inter: inter features in test set (torch.Tensor)
        test_envi: envi features in test set (torch.Tensor)
        label_test: label in test set (int., ordinal)

    return:
        rps: Ranked Probability Score
    """
    with torch.no_grad():
        # output probability [batch_size, num_classes]
        pred_probs = torch.softmax(net.forward(test_main, test_inter, test_envi), dim=1)

    # int.
    true_labels = np.array(label_test).astype(int)
    pred_probs = pred_probs.cpu().numpy()

    num_classes = pred_probs.shape[1]
    n_samples = len(true_labels)

    min_label = np.min(true_labels)
    if min_label > 0:
        true_labels -= min_label

    # cumulative prob.
    cumulative_true = np.zeros((n_samples, num_classes))
    cumulative_pred = np.zeros((n_samples, num_classes))

    # real distribution
    for i in range(n_samples):
        
        true_class = true_labels[i]

        # examine j >= true_class
        for j in range(num_classes):
            cumulative_true[i, j] = 1.0 if j >= true_class else 0.0

    # prediction prob.
    for j in range(num_classes):
        cumulative_pred[:, j] = np.sum(pred_probs[:, :j + 1], axis=1)

    # calculate RPS
    rps_per_sample = np.sum((cumulative_pred - cumulative_true) ** 2, axis = 1)

    # average
    return np.mean(rps_per_sample)

def accuracy(trained_net, test_main,test_inter,test_envi, label_test):
    '''
    we use trained_net and label to calculate accuracy:
    :param net: trained_net
    :param label: label
    :return: accuracy
    '''
    n_test = test_main.shape[0]
    pred_categories = torch.argmax(trained_net.forward(test_main,test_inter,test_envi), \
                                   1, keepdim=True).reshape(n_test)
    label = np.array(label_test)
    pred_categories = np.array(pred_categories)
    acc = metrics.accuracy_score(label,pred_categories)
    return acc

def getTPFP(net, prop1 = 0.13, prop2 = 0.10):#, threshold1 = 2., threshold2 = 1.2):
    a_main = torch.zeros_like(net.sparse1.weight.data)
    a_inter= torch.zeros_like(net.sparse2.weight.data)
    b_main = torch.zeros_like(net.sparse1.weight.data)
    b_inter= torch.zeros_like(net.sparse2.weight.data)

    # threshold1 = prop1 * net.sparse1.weight.data.abs().max()#(torch.mean(net.sparse1.weight.data) + prop1 * torch.std(net.sparse1.weight.data)).abs()
    # threshold2 = prop2 * net.sparse2.weight.data.abs().max()#(torch.mean(net.sparse2.weight.data) + prop2 * torch.std(net.sparse2.weight.data)).abs()

    temp_main = int(len(net.sparse1.weight.data) * prop1)
    threshold1 = torch.topk(net.sparse1.weight.data, temp_main)[0].min()

    temp_inter = int(len(net.sparse2.weight.data) * prop2)
    threshold2 = torch.topk(net.sparse2.weight.data, temp_inter)[0].min()
    a_main[0:15] = 1
    a_inter[0:15] = 1
    b_main[net.sparse1.weight.data.abs() > threshold1] = 1
    b_inter[net.sparse2.weight.data.abs() > threshold2]=1

    cnf_matrix_main = confusion_matrix(a_main.tolist(), b_main.tolist()) # (true, pred)
    cnf_matrix_inter= confusion_matrix(a_inter.tolist(),b_inter.tolist())

    tp_main = cnf_matrix_main[1,1]
    fp_main = cnf_matrix_main[0,1]
    tp_inter = cnf_matrix_inter[1,1]
    fp_inter = cnf_matrix_inter[0,1]
    tpr_main = tp_main/(tp_main + cnf_matrix_main[1,0])
    fpr_main = fp_main/(fp_main + cnf_matrix_main[0,0])
    tpr_inter= tp_inter/(tp_inter + cnf_matrix_inter[1,0])
    fpr_inter= fp_inter/(fp_inter + cnf_matrix_inter[0,0])

    # === examine hierarchy ===
    # temp = [num % 100 for num in b_inter.tolist()]
    # exists = any(item in b_main.tolist() for item in temp)
    # print(exists)
    return tpr_main, fpr_main, tpr_inter,fpr_inter #tp_main, fp_main, tp_inter,fp_inter

###### one example:

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
