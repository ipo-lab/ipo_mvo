import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from ipo.loss import loss_mvo
from ipo.popt import MVOIPOUncon, MVOIPOIneqcon
from experiments.utils import generate_problem_data, plot_loss, torch_rolling_cov

# --- params
n_sims = 5
noise = 50
snr = 1/noise
res = 20
rho = 0.0
constraint_list = [0.05, 0.10, 0.25, 0.50, 0.75, 1, 2, 5, 10]

n_x = 3
n_y = 10
lookback = res * n_y
n_samples = 1000
n_obs = 2 * n_samples + lookback

results_ipo_uncon_mvo = pd.DataFrame(np.zeros((n_sims, len(constraint_list))), columns=constraint_list)
results_ipo_mvo = pd.DataFrame(np.zeros((n_sims, len(constraint_list))), columns=constraint_list)


# --- simple loop
for constraint in constraint_list:
    lb = -torch.ones(n_y, 1) * constraint
    ub = torch.ones(n_y, 1) * constraint
    for i in range(n_sims):
        print(i)
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
        model_ipo_uncon = MVOIPOUncon(P=P, lb=lb, ub=ub, lam=0.10)
        model_ipo = MVOIPOIneqcon(P=P, lb=lb, ub=ub, lam=0.10)

        # --- fit, predict, optimize:
        model_ipo_uncon.fit(x=x_is, y=y_is, V=V_is, V_hat=V_hat_is)
        y_hat_ipo_uncon = model_ipo_uncon.predict(x=x_oos)
        z_ipo_uncon = model_ipo.optimize(y_hat=y_hat_ipo_uncon, V_hat=V_hat_oos)

        model_ipo.fit(x=x_is, y=y_is, V=V_is, V_hat=V_hat_is, lr=0.25, n_epochs=100)
        y_hat_ipo = model_ipo.predict(x=x_oos)
        z_ipo = model_ipo.optimize(y_hat=y_hat_ipo, V_hat=V_hat_oos)
        z_ipo = z_ipo.detach()

        # --- losses:
        results_ipo_uncon_mvo[constraint].loc[i] = loss_mvo(z=z_ipo_uncon.squeeze(2), y=y_oos, cov_mat=V_oos).numpy()
        results_ipo_mvo[constraint].loc[i] = loss_mvo(z=z_ipo.squeeze(2), y=y_oos, cov_mat=V_oos).numpy()


# --- plot results:
plot_loss(results_1=results_ipo_uncon_mvo,
          results_2=results_ipo_mvo,
          x=constraint_list,
          xlabel='Constraint value',
          ylabel='MVO Loss',
          columns=["IPO", "IPO-Grad"])

plt.savefig(f'images/exp_3_mvo_loss_res_{res}_rho_{int(100*rho)}.png')






