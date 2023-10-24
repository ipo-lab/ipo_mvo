import torch
import torch.nn as nn
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
from ipo.regression import fit_ipo_uncon, fit_ipo_eqcon, fit_ols
from ipo.loss import loss_mvo


class MVO:
    def __init__(self,
                 P,
                 A=None,
                 b=None,
                 lb=None,
                 ub=None,
                 control=box_qp_control(),
                 lam=1.0):
        n = P.shape[0]
        self.P = P
        self.A = A
        self.b = b
        if lb is None:
            lb = -torch.ones(n) * torch.inf
        self.lb = lb
        if ub is None:
            ub = torch.ones(n) * torch.inf
        self.ub = ub
        self.lam = lam
        self.QP = SolveBoxQP(control=control)
        self.theta = None

    def fit(self, **kwargs):
        raise NotImplementedError

    def predict(self, x):
        y_hat = x.matmul((self.theta * self.P).T)
        return y_hat

    def optimize(self, y_hat, V_hat):
        if len(y_hat.shape) < 3:
            y_hat = y_hat.unsqueeze(2)
        z = self.QP.forward(Q=self.lam * V_hat, p=-y_hat, A=self.A, b=self.b, lb=self.lb, ub=self.ub)
        return z


class MVOIPOUncon(MVO):
    def __init__(self, P, A=None, b=None, lb=None, ub=None, control=box_qp_control()):
        super().__init__(P=P, A=A, b=b, lb=lb, ub=ub, control=control)

    def fit(self, x, y, V, V_hat):
        self.theta = fit_ipo_uncon(x=x, y=y, P=self.P, V=V, V_hat=V_hat)


class MVOIPOEqcon(MVO):
    def __init__(self, P, A, b, lb=None, ub=None, z0=None, control=box_qp_control()):
        super().__init__(P=P, A=A, b=b, lb=lb, ub=ub, control=control)
        self.z0=z0

    def fit(self, x, y, V, V_hat):
        self.theta = fit_ipo_eqcon(x=x, y=y, P=self.P, V=V, V_hat=V_hat, A=self.A, z0=self.z0)


class MVOOLS(MVO):
    def __init__(self, P, A=None, b=None, lb=None, ub=None, control=box_qp_control()):
        super().__init__(P=P, A=A, b=b, lb=lb, ub=ub, control=control)

    def fit(self, x, y, V, V_hat):
        self.theta = fit_ols(x=x, y=y, P=self.P)


class MVOIPOIneqcon(MVO):
    def __init__(self, P, A, b, lb, ub, control=box_qp_control()):
        super().__init__(P=P, A=A, b=b, lb=lb, ub=ub, control=control)

    def fit(self,
            x,
            y,
            V,
            V_hat,
            lr=0.10,
            n_epochs=100):
        model = IPO(P=self.P)
        loss_hist = []
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            y_hat = model(x=x)
            z = self.optimize(y_hat=y_hat, V_hat=V_hat)
            loss = loss_mvo(z=z.squeeze(2), y=y, cov_mat=V, lam=self.lam)
            loss_hist.append(loss.item())
            optimizer.zero_grad()
            # --- compute gradients
            loss.backward()
            # --- update parameters
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))
        self.theta = model.theta


class IPO(nn.Module):
    def __init__(self, P):
        super().__init__()
        # --- init cov model:
        self.P = P
        self.theta = torch.nn.Parameter(torch.randn(P.shape[0]))

    def forward(self, x):
        y_hat = x.matmul((self.theta * self.P).T)
        return y_hat

