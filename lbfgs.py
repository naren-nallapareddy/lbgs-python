import numpy as np
import scipy.optimize as spo
from typing import Callable
from utils import Funcd


EPS = np.finfo(float).eps
MAXITER = 100_000
MAXCORR = 30


def inv_hessian(
    sk_memo,
    yk_memo,
    rho_memo,
    q,
    max_corr=None,
):
    if max_corr is None:
        max_corr = MAXCORR

    if len(sk_memo) > max_corr:
        raise ValueError("Number of stored vectors exceeds max_corr")

    # using the same formula names as in Nocedal and Wright for ease of reference
    alphas = []
    for s, rho, y in zip(sk_memo[::-1], rho_memo[::-1], yk_memo):
        alpha = rho * np.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)  # keeping track of alpha

    r = np.dot(sk_memo[-1], yk_memo[-1]) / np.dot(yk_memo[-1], yk_memo[-1]) * q

    for y, s, alpha in zip(yk_memo, sk_memo, alphas[::-1]):
        beta = rho * np.dot(y, r)
        r = r + s * (alpha - beta)

    return r


def lbfgs(func, x_0, g_tol, max_iters, max_corr=None):
    assert isinstance(x_0, np.ndarray), "Input x should be numpy array"
    if x_0.ndim == 0:
        x_0.shape = (1,)  # Converting 0-d array into 1-d array
    if x_0.ndim == 2:
        x_0 = x_0.squeeze()
        x_0 = x_0.reshape((-1,))
    if max_iters > MAXITER:
        max_iters = MAXITER
    if max_corr is None:
        max_corr = MAXCORR

    g_tol = np.sqrt(EPS)

    # initializing memory
    s_memo = []
    y_memo = []
    rho_memo = []

    funcd = Funcd(
        func
    )  # function that outputs both numerical differential as well as function value

    # Nocedal uses k for iteration
    fval_k = func(x_0)
    grad_k = funcd(x_0)  # first calculation of function value and gradient

    number_params = len(x_0)
    EYE = np.eye(number_params, dtype=int)

    fval_k_m_1 = (
        fval_k + np.linalg.norm(grad_k) / 2
    )  # This is a trick based on the idea that initial step guess to dx ~ 1.

    x_k = x_0  # First guess is the initialized value

    def initial_inv_hessian():
        x_k_m_1 = x_k - np.cbrt(EPS)
        _, grad_k_m_1 = funcd(x_k_m_1)
        s_k_m_1 = x_k - x_k_m_1
        y_k_m_1 = grad_k - grad_k_m_1
        gamma_k = np.dot(y_k_m_1, s_k_m_1) / np.dot(y_k_m_1, y_k_m_1)
        inv_hessian_k = gamma_k * EYE
        return inv_hessian_k

    # Given in nocedal and wright, implementaion details of BFGS
    inv_hessian_k = initial_inv_hessian()

    sigma_3 = 0.01

    g_norm = np.amax(np.abs(grad_k))  # inf norm of gradient
    # initializations
    k = 0
    p_k = -grad_k  # the initial search direction is just negative gradient
    while g_norm > g_tol and k < max_iters:
        deltak = np.dot(grad_k, grad_k)
        try:
            alpha_k, _, _, fval_k, fval_k_m_1, grad_k_p_1 = spo.line_search(
                func,
                funcd,
                x_k,
                p_k,
                grad_k,
                old_fval=fval_k,
                old_old_fval=fval_k_m_1,
            )
        except __import__("scipy")._LineSearchError:
            Warning("Line search failed")
            break
        # Calculating s_k and y_k
        s_k = alpha_k * p_k  # This is also equal to x_k_p_1 - x_k
        x_k_p_1 = x_k + s_k

        grad_k_p_1 = funcd(x_k)
        y_k = grad_k_p_1 - grad_k

        # Precalculating rho_k
        rho_k = 1 / np.dot(y_k, s_k)

        # storing values s_k and y_k
        s_memo.append(s_k)
        y_memo.append(y_k)
        rho_memo.append(rho_k)

        assert (
            len(s_memo) == len(y_memo) == len(rho_memo)
        ), "Memo lengths are not equal"

        if len(s_memo) > max_corr:
            s_memo.pop(0)
            y_memo.pop(0)
            rho_memo.pop(0)

        # incrementing values and reassinging x_k and grad_k
        k += 1  # incrementing iteration
        x_k = x_k_p_1
        grad_k = grad_k_p_1

        g_norm = np.amax(np.abs(grad_k))  # inf norm of gradient

        # Calculating the new search direction
        p_k = -inv_hessian(s_memo, y_memo, rho_memo, grad_k, max_corr=max_corr)

    optimization_dict = {"fval": fval_k, "x": x_k, "nit": k, "gnorm": g_norm}
    return optimization_dict
