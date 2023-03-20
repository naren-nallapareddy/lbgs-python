import numpy as np
from typing import Callable
from utils import Funcd
import scipy.optimize as spo


def bfgs(
    func: Callable,
    x_0: np.ndarray,
    gtol: float,
    max_iters: int,
):
    """
    BFGS algorithm taken from page 521, numerical recipes: the art of scientific computing

    Parameters:
    ----------
    p: n-dimensional point [0,...,n-1]
    gtol: convergence criterion on \nabla x
    iter: maximum number of iterations
    """
    # Define constants

    ITMAX: int = 20000  # maximum allowed number of iteratios
    EPS: float = np.finfo(float).eps  # type: ignore

    assert max_iters <= ITMAX, "Too many iterations in BFGS"

    x_k = np.copy(x_0)  # Initial point is point at x_k
    n: int = len(x_k)

    if x_k.dtype != np.float64:
        x_k = x_k.astype(np.float64)

    y_k = np.zeros(
        (n,), dtype=float
    )  # y_k is the vector of grad difference \nabla f(x_k_1) - \nabla f(x_k)
    s_k = np.zeros(
        (n,), dtype=float
    )  # s_k is the vector of point difference x_k_1 - x_k, notation taken from wright and nocedal book
    gamma_k = np.dot(s_k, y_k) / np.dot(y_k, y_k)
    hessian_k = gamma_k * np.identity(
        n, dtype=float
    )  # Initialize hessian to a identity matrix

    func_derivative = Funcd(func)
    f_k = func(x_k)  # caclulate initial gradient and value
    g_k = func_derivative(x_k)
    g_k_plus_1 = g_k
    f_k_plus_1 = f_k
    f_k_minus_1 = f_k + np.linalg.norm(g_k) / 2.0

    for itr in range(0, max_iters):
        if np.linalg.norm(g_k) < gtol:
            return x_k, f_k_minus_1, itr + 1
        p_k = -np.dot(
            hessian_k, g_k
        )  # Search direction as defined in wright and nocedal book
        alpha_k, _, _, f_k, f_k_minus_1, new_slope = spo.line_search(
            func,
            func_derivative,
            x_k,
            p_k,
            g_k,
            old_fval=f_k,
            old_old_fval=f_k_minus_1,
        )
        s_k = alpha_k * p_k  # Line search step
        x_k_plus_1 = x_k + s_k  # Line search step

        g_k_plus_1 = func_derivative(x_k_plus_1)
        y_k = g_k_plus_1 - g_k
        if alpha_k * np.linalg.norm(p_k) <= 0:
            break
        if not np.isfinite(f_k_minus_1):
            break

        # Update step for Hessian
        rho_k = 1.0 / np.dot(
            y_k, s_k
        )  # Again using notation from wright and nocedal book
        W_k_left = (
            np.identity(n) - rho_k * s_k[:, None] * y_k[None, :]
        )  # W is the weight matrix which is the average of the inverse hessian matrix
        W_k_right = np.identity(n) - rho_k * y_k[:, None] * s_k[None, :]
        sum_term = rho_k * s_k[:, None] * s_k[None, :]
        hessian_k = np.dot(W_k_left, np.dot(hessian_k, W_k_right)) + sum_term

        g_k = g_k_plus_1
        x_k = x_k_plus_1

    return x_k, f_k, itr + 1
