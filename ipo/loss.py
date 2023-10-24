import torch
from ipo.utils import torch_transpose_batch


def torch_portfolio_return(z, y):
    pret = z * y
    pret = pret.sum(dim=1)
    return pret


def torch_portfolio_variance(z, cov_mat):
    z = z.unsqueeze(2)
    zt = torch_transpose_batch(z)
    var = torch.matmul(zt, cov_mat)
    var = torch.matmul(var, z)
    var = var[:, 0, 0]
    return var


def loss_mvo(z, y, cov_mat, lam=1):
    ret = torch_portfolio_return(z=z, y=y)
    var = torch_portfolio_variance(z=z, cov_mat=cov_mat)
    loss = -ret + 0.5 * lam * var
    loss = loss.sum()
    return loss


def pct_var(y, y_hat):
    num = (y - y_hat)**2
    denom = y**2
    frac = num.mean()/denom.mean()
    pct_var = 1 - frac
    return pct_var



