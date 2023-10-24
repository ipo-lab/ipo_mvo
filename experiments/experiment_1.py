import matplotlib.pyplot as plt
import torch
from ipo.loss import loss_mvo, pct_var
from ipo.popt import MVOIPOUncon, MVOOLS
from ipo.utils import torch_rolling_cov
from experiments.utils import generate_problem_data

# --- params
n_x = 3
n_y = 10
lookback = 10 * n_y
n_samples = 1000
n_obs = 2 * n_samples + lookback
noise = 10
rho = 0
data = generate_problem_data(n_x=n_x, n_y=n_y, n_obs=n_obs, noise_multiplier_tau=noise, rho=rho)
P = torch.tensor(data.get('P'))
x = torch.tensor(data.get('x'))
y = torch.tensor(data.get('y'))
V = torch.tensor(data.get('cov'))
V = V.unsqueeze(0) * torch.ones(x.shape[0], 1, 1)
V_hat = torch_rolling_cov(y, lookback)
idx_is = slice(lookback, n_samples + lookback)
idx_oos = slice(n_samples + lookback, n_obs)

# --- in samples and out of sample:
x_is = x[idx_is]
y_is = y[idx_is]
V_hat_is = V_hat[idx_is]
V_is = V[idx_is]

x_oos = x[idx_oos]
y_oos = y[idx_oos]
V_hat_oos = V_hat[idx_oos]
V_oos = V[idx_oos]

# --- modelling
model_ipo = MVOIPOUncon(P=P)
model_ols = MVOOLS(P=P)

# --- fit, predict, optimize:
model_ipo.fit(x=x_is, y=y_is, V=V_is, V_hat=V_hat_is)
y_hat_ipo = model_ipo.predict(x=x_oos)
z_ipo = model_ipo.optimize(y_hat=y_hat_ipo, V_hat=V_hat_oos)

model_ols.fit(x=x_is, y=y_is, V=V_is, V_hat=V_hat_is)
y_hat_ols = model_ols.predict(x=x_oos)
z_ols = model_ols.optimize(y_hat=y_hat_ols, V_hat=V_hat_oos)

# --- losses:
loss_mvo(z=z_ipo.squeeze(2), y=y_oos, cov_mat=V_oos)
loss_mvo(z=z_ols.squeeze(2), y=y_oos, cov_mat=V_oos)

pct_var(y=y_oos, y_hat=y_hat_ipo)
pct_var(y=y_oos, y_hat=y_hat_ols)
