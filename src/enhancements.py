import torch
import time

def pock_chambolle_precondition(c, K, q, l, u, D_row = None, D_col = None, device='cpu', eps=1e-6, alpha=1):
    """
    Performs Pock-Chambolle equilibration (scaling) on the standard-form linear program using GPU tensors.
    """
        
    # --- Scaling Loop ---
    K_s, c_s, q_s, l_s, u_s = K.clone(), c.clone(), q.clone(), l.clone(), u.clone()
    
    if D_row is None or D_col is None:
        D_row = torch.ones((K_s.shape[0], 1), dtype=K.dtype, device=device)
        D_col = torch.ones((K_s.shape[1], 1), dtype=K.dtype, device=device)
    
    row_norms = torch.sqrt(torch.linalg.norm(K_s, ord=2-alpha, dim=1, keepdim=True))
    row_norms[row_norms < eps] = 1.0
    D_row /= row_norms
    K_s /= row_norms

    col_norms = torch.sqrt(torch.linalg.norm(K_s, ord=alpha, dim=0, keepdim=True))
    col_norms[col_norms < eps] = 1.0
    D_col /= col_norms.T
    K_s /= col_norms
    
    c_s *= D_col
    q_s *= D_row
    l_s /= D_col
    u_s /= D_col
    
    return K_s, c_s, q_s, l_s, u_s, D_col, D_row

def ruiz_precondition(c, K, q, l, u, device='cpu', max_iter=10, eps=1e-6):
    """
    Performs Ruiz equilibration (scaling) on the standard-form linear program using GPU tensors.

    This is done to improve the numerical stability of iterative solvers, especially for
    ill-conditioned problems.

    Standard form of the LP:
        minimize     cᵀx
        subject to   Gx ≥ h
                     Ax = b
                     l ≤ x ≤ u

    Inputs:
    -------
    c  : (n x 1) torch tensor — objective function vector
    K  : ((m_ineq + m_eq) x n) torch tensor — constraint matrix (stacked G and A)
    q  : ((m_ineq + m_eq) x 1) torch tensor — RHS vector (stacked h and b)
    l  : (n x 1) torch tensor — lower bounds on variables
    u  : (n x 1) torch tensor — upper bounds on variables
    max_iter : int — number of scaling iterations to perform (default: 20)

    Outputs:
    --------
    K_s : ((m_ineq + m_eq) x n) torch tensor — scaled constraint matrix (stacked G and A)
    c_s : (n x 1) torch tensor — scaled objective vector
    q_s : ((m_ineq + m_eq) x 1) torch tensor — scaled RHS vector (stacked h and b)
    l_s : (n x 1) torch tensor — scaled lower bounds
    u_s : (n x 1) torch tensor — scaled upper bounds
    D_col : (n x 1) torch tensor — final column scaling factors (for rescaling solution)
    m_ineq : int — number of inequality constraints (used for slicing G vs A in K_s if needed)

    Notes:
    ------
    - The scaling preserves feasibility and optimality but improves numerical conditioning.
    - You must rescale your solution after solving using D_col (and D_row if needed).
    """
    # --- Scaling Loop ---
    K_s, c_s, q_s, l_s, u_s = K.clone(), c.clone(), q.clone(), l.clone(), u.clone()
    m, n = K_s.shape

    D_row = torch.ones((m, 1), dtype=K.dtype, device=device)
    D_col = torch.ones((n, 1), dtype=K.dtype, device=device)

    for i in range(max_iter):
        row_norms = torch.sqrt(torch.linalg.norm(K_s, ord=torch.inf, dim=1, keepdim=True))
        row_norms[row_norms < eps] = 1.0
        D_row /= row_norms
        K_s /= row_norms

        col_norms = torch.sqrt(torch.linalg.norm(K_s, ord=torch.inf, dim=0, keepdim=True))
        col_norms[col_norms < eps] = 1.0
        D_col /= col_norms.T
        K_s /= col_norms

        if (torch.max(torch.abs(1 - row_norms)) < eps and
            torch.max(torch.abs(1 - row_norms)) < eps):
            break
    
    c_s *= D_col
    q_s *= D_row
    l_s /= D_col
    u_s /= D_col
    
    return K_s, c_s, q_s, l_s, u_s, D_col, D_row

def preconditioning(c, K, q, l, u, device='cpu', max_iter=10, eps=1e-6):
    time_start = time.time()
    K_s, c_s, q_s, l_s, u_s, D_col, D_row = ruiz_precondition(c, K, q, l, u, device=device, max_iter=max_iter, eps=eps)
    # K_s, c_s, q_s, l_s, u_s, D_col, D_row = pock_chambolle_precondition(c_s, K_s, q_s, l_s, u_s, D_row=D_row, D_col=D_col, device=device, eps=eps)
    time_used = time.time() - time_start
    return K_s, c_s, q_s, l_s, u_s, (D_col, D_row, K, c, q, l, u), time_used
    
def primal_weight_update(x_prev, x, y_prev, y, omega, smooth_theta):
    diff_y_norm = torch.linalg.norm(y_prev - y, 2)
    diff_x_norm = torch.linalg.norm(x_prev - x, 2)
    if diff_x_norm > 0 and diff_y_norm > 0:
        omega = torch.exp(smooth_theta * (torch.log(diff_y_norm/diff_x_norm)) + (1-smooth_theta)*torch.log(omega))
    return omega
