import torch
import pandas as pd


def make_matrix(x):
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    return x


def torch_uniform(*size, lower=0, upper=1, dtype=torch.float64):
    r = torch.rand(*size, dtype=dtype)
    r = r * (upper - lower) + lower
    return r


def is_batch(x):
    return len(x.shape) > 2


def torch_transpose_batch(x):
    if is_batch(x):
        dim0 = 1
        dim1 = 2
    else:
        dim0 = 0
        dim1 = 1

    return torch.transpose(x, dim0=dim0, dim1=dim1)


def torch_crossprod(x, y=None):
    if y is None:
        y = x

    return torch.matmul(torch_transpose_batch(x), y)


def torch_quad_form(x, y=None, dim_x=2, dim_y=1):
    if y is None:
        y = x
    return torch.matmul(x.unsqueeze(dim_x), y.unsqueeze(dim_y))


def torch_quad_form_mat(x, mat):
    xt = torch_transpose_batch(x)
    return torch.matmul(torch.matmul(xt, mat), x)


def torch_cov2cor(cov_mat):
    if is_batch(cov_mat):
        sigma = torch.diagonal(cov_mat, dim1=1, dim2=2)
        sigma = torch.sqrt(sigma)
        sigma = torch.diag_embed(1 / sigma)
    else:
        sigma = torch.diag(cov_mat)
        sigma = torch.sqrt(sigma)
        sigma = torch.diag(1 / sigma)

    cor_mat = torch.matmul(torch.matmul(sigma, cov_mat), sigma)
    return cor_mat


def torch_cor2cov(cor_mat, sigma):
    # --- dimension handling
    if not is_batch(cor_mat):
        cor_mat = cor_mat.unsqueeze(0)
    if is_batch(sigma):
        sigma = sigma.squeeze(2)

    # --- make sigma an embedded diagonal matrix
    sigma = torch.diag_embed(sigma)
    cov_mat = torch.matmul(torch.matmul(sigma, cor_mat), sigma)
    return cov_mat


def torch_rolling_cov(x, n=10):
    x = pd.DataFrame(x.numpy())
    covar = x.rolling(n).cov()
    covar = covar.to_numpy()
    covar = covar.reshape((x.shape[0], x.shape[1], x.shape[1]))
    return torch.tensor(covar)


def torch_ma(x, n, bias=None):
    weight = torch.ones((n, x.shape[1]), dtype=x.dtype)/n
    return torch_wma(x=x, weight=weight, bias=bias)


def torch_wma(x, weight, bias=None):
    # --- prep
    x = make_matrix(x)
    weight = make_matrix(weight)
    nr_x = x.shape[0]
    nc_x = x.shape[1]
    nr_weight = weight.shape[0]
    nc_weight = weight.shape[1]

    # --- repeat weight if necessary
    if not (nc_x == nc_weight):
        weight = weight.repeat(1, nc_x)

    # --- reshaping
    x = x.t().unsqueeze(0)
    weight = weight.t().unsqueeze(1)

    # --- conv1d:
    padding = nr_weight - 1
    groups = nc_x
    out = torch.nn.functional.conv1d(x=x, weight=weight, bias=bias, groups=groups, padding=padding)
    out = out.squeeze(0)
    out = out[:, :nr_x]
    out = out.t()

    return out


def torch_wma2(x, weight, bias=None):
    n_obs = x.shape[0]
    n_col = x.shape[1] * x.shape[2]
    out = torch_wma(x=x.view((n_obs, n_col)), weight=weight, bias=bias)
    out = out.view((n_obs, x.shape[1], x.shape[2]))
    return out


def torch_cov(x, center=True):
    if is_batch(x):
        dim = 1
        n_obs = x.shape[1]
    else:
        dim = 0
        n_obs = x.shape[0]

    # --- crossprod
    cov_mat = torch_crossprod(x)
    cov_mat = cov_mat / n_obs

    # --- centering
    if center:
        mu = torch.mean(x, dim=dim, keepdim=True)
        mu_mat = torch_crossprod(mu)
        cov_mat = cov_mat - mu_mat

    return cov_mat


def torch_cor(x, center=True):
    cov_mat = torch_cov(x, center=center)
    cor_mat = torch_cov2cor(cov_mat)
    return cor_mat