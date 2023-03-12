import numpy as np
from typing import Callable, TypeVar, Generic
from utils import Funcd, linesearch


def bfgs(
    current_point: np.ndarray,
    gtol: float,
    max_iters: int,
    func: Callable,
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
    TOLX: float = 4 * EPS  # convergence criterion on x
    STEP_MAX: float = 100.0  # scaled max step length allowed in line searches

    assert max_iters <= ITMAX, "Too many iterations in BFGS"

    n: int = len(current_point)

    d_grad = np.zeros((n,))
    hessian_d_grad = np.zeros((n,))
    search_direction_new = np.zeros((n,))

    hessian = np.identity(n)  # Initialize hessian to a identity matrix

    funcd: Funcd = Funcd(func)
    f_current_point = funcd(
        current_point
    )  # caclulate initial gradient and value
    f_return = f_current_point
    grad = funcd.df(current_point)
    sum = np.sum(current_point**2)
    x_initial = -grad  # initial line direction

    step_max = STEP_MAX * np.maximum(
        np.sqrt(sum), float(n)
    )  # initial step length

    for iteration in range(0, max_iters):
        _, search_direction_new, fp_new, _ = linesearch(current_point, f_current_point, grad, x_initial, step_max, funcd)  # type: ignore
        f_current_point = fp_new
        x_initial = search_direction_new - current_point
        test_for_convergence = np.max(
            np.abs(x_initial) / np.maximum(np.abs(current_point), 1.0)
        )
        ##! Literal translation of the for loop below
        # for ix in range(0, n):
        # temp = abs(x_initial[ix]) / np.maximum(abs(current_point[ix]), 1.0)
        # if temp > test_for_convergence:
        # test_for_convergence = temp
        ##!
        if test_for_convergence < TOLX:
            return current_point, iteration, f_return
        d_grad = grad
        grad = funcd.df(search_direction_new)
        test_for_convergence = 0.0
        den = np.maximum(f_return, 1.0)  # type: ignore
        test_for_convergence = np.max(
            (np.abs(grad) * np.maximum(np.abs(search_direction_new), 1.0))
            / den
        )  # type: ignore
        ##! Literal translation of the for loop below
        # for ix in range(0, n):
        # temp = (
        # np.abs(grad[ix])
        # * np.maximum(abs(current_point[ix]), 1.0)
        # / den
        # )
        # if temp > test_for_convergence:
        # test_for_convergence = temp
        ##!
        if test_for_convergence < gtol:
            return current_point, iteration, f_return

        d_grad = grad - d_grad

        hessian_d_grad = np.squeeze(d_grad[None] @ hessian)
        ##! Literal translation of the for loop below
        # for ix in range(0, n):
        # for jx in range(0, n):
        # hessian_d_grad[ix] += hessian[ix, jx] * d_grad[jx]
        ##!
        fac = np.dot(d_grad, x_initial)
        fae = np.dot(d_grad, hessian_d_grad)
        sum_dgrad = np.sum(d_grad**2)
        sum_x_initial = np.sum(x_initial**2)

        if fac > np.sqrt(EPS * sum_dgrad * sum_x_initial):
            fac = 1.0 / fac
            fad = 1.0 / fae
            d_grad = fac * x_initial - fad * hessian_d_grad
            for ix in range(0, n):
                for jx in range(ix, n):
                    hessian[ix, jx] += (
                        fac * x_initial[ix] ** 2
                        - fad * hessian_d_grad[ix] ** 2
                        + fae * d_grad[ix] ** 2
                    )
                    hessian[jx, ix] = hessian[ix, jx]

        for ix in range(0, n):
            for jx in range(0, n):
                x_initial[ix] -= (
                    hessian[ix, jx] * grad[jx]
                )  # Next calculate the new search direction
