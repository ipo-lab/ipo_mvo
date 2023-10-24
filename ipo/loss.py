import torch


def torch_transpose_batch(x):
    if len(x.shape) > 2:
        dim0 = 1
        dim1 = 2
    else:
        dim0 = 0
        dim1 = 1

    return torch.transpose(x, dim0=dim0, dim1=dim1)


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
    loss = loss.mean()
    return loss


def pct_var(y, y_hat):
    num = (y - y_hat) ** 2
    denom = y ** 2
    frac = num.mean() / denom.mean()
    pct_var = 1 - frac
    return pct_var
