import numpy as np
import math
import torch
import torch.nn as nn
import warnings
import random

torch.autograd.set_detect_anomaly(True)

EPS = 1e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print("Device:", device)

def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean


class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        if method:
            warnings.warn("arg method is deprecated")
        self.T = T

    def forward(self, x, z, z_marg=None): # x = encoder(data), z = attacker(x)
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg) # (batch,)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()
                # if random.random() < 1e-2:
                # print('loss: {:.5f}\t max-abs of grad(T.FC): {:.5f}\t avg-abs: {:.5f}'.format(loss.item(), self.T.FC.weight.grad.abs().max(),self.T.FC.weight.grad.abs().mean() ))
                #     # import pdb;pdb.set_trace()
                #     print("max:",self.T.FC.weight.grad.max())
                #     print("abs avg:", self.T.FC.weight.grad.abs().mean())
                mu_mi -= loss.item()
            # if iter % (iters // 3) == 0:
            #     pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi

def batch_2(x, y, y_2, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    if isinstance(y_2, np.ndarray):
        y_2 = torch.from_numpy(y_2).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]
        y_2 = y_2[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]
        y_2_b = y_2[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b, y_2_b))
    return batches

def batch_3(x, y, y_2, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
    if isinstance(y_2, np.ndarray):
        y_2 = torch.from_numpy(y_2).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]
        y_2_m = y_2.mask
        y_2_e = y_2.emb
        y_2 = type(y_2)(y_2_m[:,rand_perm], y_2_e[:,rand_perm])

    batches = []
    y_2_m = y_2.mask
    y_2_e = y_2.emb
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]
        y_2_m_b = y_2_m[:,i * batch_size: (i + 1) * batch_size]
        y_2_e_b = y_2_e[:,i * batch_size: (i + 1) * batch_size]
        y_2_b = type(y_2)(y_2_m_b,y_2_e_b)

        batches.append((x_b, y_b, y_2_b))
    return batches


class Mine_withKey(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        if method:
            warnings.warn("arg method is deprecated")
        self.T = T

    def forward(self, x, z, z_2, z_marg=None): # x = encoder(data), z = attacker(x)
        if z_marg is None:
            perm = torch.randperm(x.shape[0])
            z_marg = z[perm]
            z_2_marg = z_2[perm]

        t = self.T(x, z, z_2).mean()
        t_marg = self.T(x, z_marg, z_2_marg) # (batch,)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, Y_2, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y, y_2 in batch_2(X, Y,Y_2, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y, y_2)
                loss.backward()
                opt.step()
                # if random.random() < 1e-2:
                # print('loss: {:.5f}\t max-abs of grad(T.FC): {:.5f}\t avg-abs: {:.5f}'.format(loss.item(), self.T.FC.weight.grad.abs().max(),self.T.FC.weight.grad.abs().mean() ))
                #     # import pdb;pdb.set_trace()
                #     print("max:",self.T.FC.weight.grad.max())
                #     print("abs avg:", self.T.FC.weight.grad.abs().mean())
                mu_mi -= loss.item()
            # if iter % (iters // 3) == 0:
            #     pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y, Y_2)
        print(f"Final MI: {final_mi}")
        return final_mi


class Mine_withKeyDiff(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        if method:
            warnings.warn("arg method is deprecated")
        self.T = T

    def forward(self, x, z, z_2, z_marg=None): # x = encoder(data), z = attacker(x)
        if z_marg is None:
            perm = torch.randperm(x.shape[0])
            z_marg = z[perm]
            z_2_m = z_2.mask
            z_2_e = z_2.emb
            z_2_marg = type(z_2)(z_2_m[:,perm], z_2_e[:,perm])

        t = self.T(x, z, z_2).mean()
        t_marg = self.T(x, z_marg, z_2_marg) # (batch,)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, Y_2, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y, y_2 in batch_3(X, Y,Y_2, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y, y_2)
                loss.backward()
                opt.step()
                # if random.random() < 1e-2:
                # print('loss: {:.5f}\t max-abs of grad(T.FC): {:.5f}\t avg-abs: {:.5f}'.format(loss.item(), self.T.FC.weight.grad.abs().max(),self.T.FC.weight.grad.abs().mean() ))
                #     # import pdb;pdb.set_trace()
                #     print("max:",self.T.FC.weight.grad.max())
                #     print("abs avg:", self.T.FC.weight.grad.abs().mean())
                mu_mi -= loss.item()
            # if iter % (iters // 3) == 0:
            #     pass
                #print(f"It {iter} - MI: {mu_mi / batch_size}")

        final_mi = self.mi(X, Y, Y_2)
        print(f"Final MI: {final_mi}")
        return final_mi
