from torch.nn import Softmax
from torch import nn
import numpy as np
import torch
import torch.optim as optim

from Functions import CE

### Data generating process ###
def generateContinuousData(rho, dim, n):  # dim: the dimension of X; n: the sample size

    '''
    We generate data following multivariate normal distribution
    :param rho: AR(rho)
    :param dim: dimensionality
    :param n: sample of size
    :return: data with shape (n, dim) and AR(rho)
    '''

    cov = np.zeros(shape=(dim, dim))
    mean = np.zeros(dim)

    for i in range(dim):
        for j in range(dim):
            cov[i, j] = rho ** (abs(i - j))
    return np.random.multivariate_normal(mean=mean, cov=cov, size=n)


def DGP(seed, n, rho, dim, setting, proportion):
    np.random.seed(seed)
    error = np.random.logistic(0.0, 1.0, n)
    error_prime = np.random.normal(0.0,1.0,n)
  
    main = generateContinuousData(rho, dim, n)
    envi = generateContinuousData(0.5, 5, n)
    envi[:,0:2] = np.where(envi[:,0:2] > 0, 1, -1)
    inter= np.zeros(shape=(n, dim * 5))
    k = 0
    for i in range(5):
        for j in range(dim):
            inter[:,k] = envi[:,i] * main[:,j]
            k = k+1

    if setting == 'nonlinear1_weak':
        coef = np.random.uniform(0.4,0.7,35)
    else:
        coef = np.random.uniform(0.6, 0.9, 35)
    h = (np.sum(main[:,0:15] * coef[0:15], axis = 1)\
         + np.sum(inter[:,0:15] * coef[15:30], axis = 1)\
         + np.sum(envi * coef[30:35], axis = 1))
    breve = np.zeros(n)

    if setting == 'linear':
        u = h  + error
    elif setting == 'nonlinear1':
        u = h + np.sin(h) + error
    elif setting == 'nonlinear2':
        u = -2 * (h + 10) ** 2 + error_prime
    elif setting == 'nonlinear1_weak':
        u = h + np.sin(h) + error

    if proportion == 'eq':
        breve[u < np.percentile(u, 33)] = 0
        for i in range(n):
            if (u[i] >= np.percentile(u, 33)) and  (u[i] < np.percentile(u, 67)):
                breve[i] = 1
        breve[u > np.percentile(u, 67)] = 2

    elif proportion == 'neq':

        breve[u < np.percentile(u, 25)] = 0
        for i in range(n):
            if (u[i] >= np.percentile(u, 25)) and  (u[i] < np.percentile(u, 75)):
                breve[i] = 1
        breve[u > np.percentile(u, 75)] = 2

    # === one-hot encoding ===
    y_breve = np.zeros(shape=(n, 3))
    for i in range(n):
        y_breve[i, (breve.astype(int))[i]] = 1

    return y_breve, main, inter, envi, breve.reshape(n, 1)

  
### functions for OrdinalNN ###

def CE(y_pred, y_label):
    return -(y_label * torch.log(y_pred + 1e-5)).sum()/(y_label.shape[0] * y_label.shape[1])

def sparse(dim):
    '''
    sparse layer coefficient
    :param dim: dimensionality of input
    :param sig: standard deviation of initial values
    :return: initial sparse coefficients
    '''

    np.random.seed(1)
    return torch.Tensor(np.random.normal(0, 0.56, size=dim))

def weight(www, num_genes, num_inters, num_environs, num_hiddens1, num_hiddens2, num_outputs):
    np.random.seed(www)
    w1r = num_genes + num_inters + num_environs
    w1c = num_hiddens1
    w1 = np.random.uniform(-0.15, 0.15, size=(w1c, w1r))
    w1 = torch.Tensor(w1)

    w2r = num_hiddens1
    w2c = num_hiddens2
    w2 = np.random.uniform(-0.24, 0.24, size=(w2c, w2r))
    w2 = torch.Tensor(w2)

    w3r = num_hiddens2
    w3c = num_outputs
    w3 = np.random.uniform(-0.31, 0.31, size=(w3c, w3r))
    w3 = torch.Tensor(w3)
    return w1, w2, w3

def bias(iii, num_hiddens1, num_hiddens2, num_outputs):
    np.random.seed(iii)
    b1 = np.random.uniform(-0.01, 0.01, size=num_hiddens1)
    b2 = np.random.uniform(-0.01, 0.01, size=num_hiddens2)
    b3 = np.random.uniform(-0.01, 0.01, size=num_outputs)

    b1 = torch.Tensor(b1)
    b2 = torch.Tensor(b2)
    b3 = torch.Tensor(b3)
    return b1, b2, b3

class weight_sparse(nn.Module):
    def __init__(self, input):
        super(weight_sparse, self).__init__()
        self.input = input
        self.weight = nn.Parameter(sparse(self.input), requires_grad = True)

    def forward(self, feature):
        return feature * self.weight


class Cumulative(nn.Module): # example for three-category classification
    def __init__(self):
        super(Cumulative, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([-1.5]), requires_grad=True)
        self.gamma = nn.Parameter(torch.Tensor([1.09]), requires_grad=True)

    def forward(self, x):
        e1 = (torch.exp(self.alpha + x) / (1 + torch.exp(self.alpha + x))).reshape(x.shape[0])
        e2 = (torch.exp(self.alpha + torch.exp(self.gamma) + x) / (1 + torch.exp(self.alpha + torch.exp(self.gamma) + x))).reshape(x.shape[0])

        pi = torch.zeros(size=(x.shape[0], 3))
        pi[:, 0] = e1 - 0.
        pi[:, 1] = e2 - e1
        pi[:, 2] = 1. - e2

        return pi


class Cumu_NN(nn.Module):
    def __init__(self, num_genes, num_inters, num_environs, num_hiddens1, num_hiddens2, num_outputs):
        super(Cumu_NN, self).__init__()
        self.prelu = nn.PReLU()

        self.sparse1 = weight_sparse(num_genes)
        self.sparse2 = weight_sparse(num_inters)
        self.softmax = Softmax()
        self.Cu = Cumulative()

        self.sc1 = nn.Linear(num_genes + num_inters + num_environs, num_hiddens1)
        self.sc2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.sc3 = nn.Linear(num_hiddens2, num_outputs)

        self.sc1.weight.data = weight(6, num_genes, num_inters, num_environs, num_hiddens1, num_hiddens2, num_outputs)[0]
        self.sc2.weight.data = weight(6, num_genes, num_inters, num_environs, num_hiddens1, num_hiddens2, num_outputs)[1]
        self.sc3.weight.data = weight(6, num_genes, num_inters, num_environs, num_hiddens1, num_hiddens2, num_outputs)[2]

        self.sc1.bias.data = bias(6, num_hiddens1, num_hiddens2, num_outputs)[0]
        self.sc2.bias.data = bias(6, num_hiddens1, num_hiddens2, num_outputs)[1]
        self.sc3.bias.data = bias(6, num_hiddens1, num_hiddens2, num_outputs)[2]

    def forward(self, x1, x2, x3):
        x1 = self.sparse1(x1)
        x2 = self.sparse2(x2)

        x = torch.cat((x1, x2), 1)
        x = torch.cat((x, x3), 1)

        x = self.prelu(self.sc1(x))
        x = self.prelu(self.sc2(x))
        p = self.Cu(self.sc3(x))

        return p

def Train_model(train_main, train_inter, train_envi, train_y,\
             valid_main, valid_inter, valid_envi, valid_y,\
             ridge, Lam, Num_Epochs, xi): 
    loss_train_list = []
    loss_test_list = []
    main_dim = train_main.shape[1]
    inter_dim= train_inter.shape[1]
    envi_dim = train_envi.shape[1]
    d = np.sqrt(envi_dim + 1)
    net = Cumu_NN(main_dim, inter_dim, envi_dim, 128, 128, 1)
    opt = optim.SGD([
        {'params': net.sc1.parameters(), 'weight_decay': ridge, 'lr': 0.01, 'momentum': 0.99, 'dampening': 0.75,
         'nesterov': False},
        {'params': net.sc2.parameters(), 'weight_decay': ridge, 'lr': 0.03, 'momentum': 0.99, 'dampening': 0.75,
         'nesterov': False},
        {'params': net.sc3.parameters(), 'weight_decay': ridge, 'lr': 0.01, 'momentum': 0.99, 'dampening': 0.75,
         'nesterov': False},
        {'params': net.sparse1.parameters(), 'weight_decay': 0, 'lr': 0.05, 'momentum': 0.99, 'dampening': 0.75,
         'nesterov': False},
        {'params': net.sparse2.parameters(), 'weight_decay': 0, 'lr': 0.05, 'momentum': 0.99, 'dampening': 0.5,
         'nesterov': False},
        {'params': net.Cu.parameters(), 'weight_decay': 0.00001, 'lr': 0.05, 'momentum': 0.99, 'dampening': 0.75,
         'nesterov': False},
    ])

    for epoch in range(Num_Epochs + 1):
        net.train()
        # === mcp ===
        regularization_loss = 0

        b = torch.zeros([main_dim])
        for i in range(main_dim):
            temp = 0
            for j in range(envi_dim):
                temp += torch.abs(net.sparse2.weight.data[i + j * main_dim])
            b[i] = torch.abs(net.sparse1.weight.data[i]) + temp

        a = d * Lam - b / torch.tensor(xi)
        zero = torch.zeros([main_dim])
        Pb = torch.where(a < 0, zero, a)
        w1 = Pb / (2 * b ** 2)

        for param in net.sparse1.parameters():
            regularization_loss += torch.sum(param ** 2 * torch.abs(param.data) * w1)

        bInter = torch.tensor(w1.tolist() * envi_dim)
        for param in net.sparse2.parameters():
            a = Lam - torch.abs(param.data) / torch.tensor(xi)
            zero = torch.zeros(inter_dim)
            Pbeta = torch.where(a < 0, zero, a)
            regularization_loss += torch.sum(
                param ** 2 * (torch.abs(param.data) * bInter + Pbeta / (2 * torch.abs(param.data))))

        pred = net(train_main, train_inter, train_envi)
        opt.zero_grad()  ###reset gradients to zeros
        loss = CE(pred, train_y) + regularization_loss
        loss_train_list.append(loss)

        predt = net(valid_main, valid_inter, valid_envi)
        losst = CE(predt, valid_y) + regularization_loss
        loss_test_list.append(losst)

        loss.backward()  ###calculate gradients
        opt.step()  ###update weights and biases

    loss_train_list = torch.Tensor(loss_train_list)
    loss_test_list = torch.Tensor(loss_test_list)

    return net, loss_train_list, loss_test_list
