import torch


def fit_ols(x, y, P):
    # --- unsqueeze x and y
    x_u2 = x.unsqueeze(1)
    x_u3 = x.unsqueeze(2)
    y_u = y.unsqueeze(2)
    ones = torch.ones((x.shape[0], 1, 1))
    # --- P transpose
    Pt = P.t()
    PP = Pt.matmul(P)
    PP = PP * ones
    Pt = Pt * ones

    # --- rhs
    P_y = Pt.matmul(y_u)
    x_P_y = x_u3 * P_y
    d = torch.sum(x_P_y, 0)

    # --- hessian elements
    H = x_u2 * PP
    H = torch.transpose(H, 1, 2) * x_u2

    H = torch.sum(H, 0)

    # We may not be able to solve because H is singluar
    theta = torch.linalg.lstsq(H, d)[0]  # torch.linalg.solve(H,d)

    return theta[:, 0]


def fit_ipo_uncon(x, y, P, V, V_hat):
    # --- preamble
    Pt = P.t()

    # --- unsqueeze x and y
    x_u2 = x.unsqueeze(1)
    x_u3 = x.unsqueeze(2)
    y_u = y.unsqueeze(2)

    # --- rhs elements
    V_hat_inv = torch.linalg.inv(V_hat)
    PV_hat_inv = torch.matmul(Pt, V_hat_inv)
    PV_hat_inv_t = torch.transpose(PV_hat_inv, 1, 2)
    PV_hat_inv_y = PV_hat_inv.matmul(y_u)
    x_PV_hat_inv = x_u3 * PV_hat_inv_y
    d = torch.sum(x_PV_hat_inv, 0)

    # --- hessian elements
    M = PV_hat_inv.matmul(V)
    M = M.matmul(PV_hat_inv_t)
    H = x_u2 * M
    H = torch.transpose(H, 1, 2) * x_u2
    H = torch.sum(H, 0)

    # We may not be able to solve because H is singluar
    theta = torch.linalg.lstsq(H, d)[0]  # torch.linalg.solve(H,d)
    return theta[:, 0]


def fit_ipo_eqcon(x, y, P, V, V_hat, A, z0):
    # --- preamble
    Pt = P.t()

    # --- unsqueeze x and y
    x_u2 = x.unsqueeze(1)
    x_u3 = x.unsqueeze(2)
    y_u = y.unsqueeze(2)

    # --- Compute nullspace basis of A
    f = nullspace(A)
    ft = f.t()

    # --- fVf
    f_Vhat = ft.matmul(V_hat)
    f_Vhat_f = f_Vhat.matmul(f)
    f_Vhat_f_inv = torch.linalg.inv(f_Vhat_f)

    V_hat_inv = f.matmul(f_Vhat_f_inv).matmul(ft)

    # --- rhs elements
    PV_hat_inv = Pt.matmul(V_hat_inv)
    PV_hat_inv_t = torch.transpose(PV_hat_inv, 1, 2)

    rhs = z0 - V_hat_inv.matmul(V_hat).matmul(z0)
    rhs = -y_u + V.matmul(rhs)

    PV_hat_inv_y = PV_hat_inv.matmul(rhs)
    x_PV_hat_inv = -x_u3 * PV_hat_inv_y
    d = torch.sum(x_PV_hat_inv, 0)

    # --- hessian elements
    M = PV_hat_inv.matmul(V)
    M = M.matmul(PV_hat_inv_t)
    H = x_u2 * M
    H = torch.transpose(H, 1, 2) * x_u2

    H = torch.sum(H, 0)

    # We may not be able to solve because H is singluar
    theta = torch.linalg.lstsq(H, d)[0]  # torch.linalg.solve(H,d)
    return theta[:, 0]


def nullspace(A, rcond=None):
    u, s, vh = torch.Tensor.svd(A, some=False, compute_uv=True)
    vh = vh.T
    Mt, Nt = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = torch.finfo(s.dtype).eps * max(Mt, Nt)
    tolt = torch.max(s) * rcond
    numt = torch.sum(s > tolt)
    nspace = vh[numt:, :].T
    return nspace
