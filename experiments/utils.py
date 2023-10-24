import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_problem_data(n_x=3, n_y=5, n_obs=1000, noise_multiplier_tau=0, polys=[1], rho=0):
    # --- generate regression coefficients:
    P = generate_P(n_y=n_y, n_x=n_x)
    n_poly = len(polys)
    theta_list = []
    P_theta_list = []
    n_total_x = n_y * n_x
    for i in range(n_poly):
        theta = np.random.uniform(low=-3, high=3, size=n_total_x)
        theta_list.append(theta)
        P_theta_list.append(theta * P)  # == P.dot(np.diag(theta))

    # --- create x:
    x = np.random.normal(size=(n_obs, n_total_x))

    # --- true f(x)
    f_x = 0
    for i in range(n_poly):
        f_x += np.dot(x ** polys[i], P_theta_list[i].T)

    # --- additive errors:
    mean = np.zeros(n_y)
    if rho == 0:
        cov = np.diag(np.ones(n_y))
    else:
        v = np.arange(n_y)
        v = v[:, None]
        d = np.abs(v[:, None] - v).sum(-1)
        cov = rho ** d
    errors = np.random.multivariate_normal(mean=mean, cov=cov, size=n_obs)

    # --- generate y with correlated errors
    frac = 1 / (1 + noise_multiplier_tau)
    y = frac * f_x + (1 - frac) * errors
    y_std = y.std(axis=0)
    y = y/y_std
    x = x/y_std.repeat(n_x)

    # --- data
    data = {"x": x,
            "y": y,
            "cov": cov,
            "theta": theta_list,
            "P_theta": P_theta_list,
            "P": P}
    # np.linalg.lstsq(x, y)[0]

    return data


def generate_coef(n_y=5, n_x=3, pct_true=0.5, low=-3, high=3):
    # --- sparsity matrix:
    smat = np.random.binomial(n=1, p=pct_true, size=(n_x, n_y))
    # --- coefficients:
    b = np.random.uniform(low=low, high=high, size=(n_x, n_y))
    return smat * b


def generate_P(n_y, n_x):
    m = n_y * n_x
    P = np.zeros((n_y, m))
    for i in range(n_y):
        idx = range(i * n_x, ((i + 1) * n_x))
        P[i, idx] = 1.0
    return P


def plot_loss(results_ipo, results_ols, snr, ylabel='MVO Loss'):
    # --- plot
    mvo_mean = pd.concat({
        'IPO': results_ipo.mean(axis=0),
        'OLS': results_ols.mean(axis=0)}, axis=1)

    # --- error bars:
    mvo_error = pd.concat({
        'IPO': results_ipo.std(axis=0),
        'OLS': results_ols.std(axis=0)}, axis=1)

    color2 = ["#0000EE", "#CD3333"]
    mvo_mean.plot.line(ylabel=ylabel, xlabel='Signal-to-noise', color=color2, linewidth=4, logx=True)  # title=title
    plt.fill_between(snr, mvo_mean['IPO'] - 2 * mvo_error['IPO'],
                     mvo_mean['IPO'] + 2 * mvo_error['IPO'], alpha=0.25, color=color2[0])
    plt.fill_between(snr, mvo_mean['OLS'] - 2 * mvo_error['OLS'],
                     mvo_mean['OLS'] + 2 * mvo_error['OLS'], alpha=0.25, color=color2[1])