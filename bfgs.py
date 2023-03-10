import numpy as np
from typing import Callable, TypeVar, Generic
from utils import Funcd, linesearch



def bfgs(
    current_point: np.ndarray, gtol: float, iter: int, fret: float, func: Callable,
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

    ITMAX: int = 2000  # maximum allowed number of iteratios
    EPS: float = np.finfo(float).eps  # type: ignore
    TOLX: float = 4 * EPS  # convergence criterion on x
    STEP_MAX: float = 100.0  # scaled max step length allowed in line searches

    n: int = len(current_point)

    dg = np.zeros((n,))
    hdg = np.zeros((n,))
    pnew = np.zeros((n,))

    hessian = np.identity(n)  # Initialize hessian to a identity matrix

    funcd: Funcd = Funcd(func, EPS)
    f_current_point = funcd(current_point)  # caclulate initial gradient and value
    grad = funcd.df(current_point)
    sum = np.sum(current_point**2)
    x_initial = -grad  # initial line direction

    step_max = STEP_MAX * np.maximum(np.sqrt(sum), float(n))  # initial step length

    for iteration in range(0, ITMAX):
        _, _, fp_new, check = linesearch(current_point, f_current_point, grad, x_initial, step_max, check, func)  # type: ignore
        f_current_point = fp_new
        x_initial = pnew - current_point

        # for ix in range(0, n):
            # x_initial[ix] = pnew[ix] - current_point[ix]
            # current_point[ix] = pnew[ix]
        test = 0.0
        for ix in range(0, n):
            temp = abs(x_initial[ix]) / np.maximum(abs(current_point[ix]), 1.0)
            if temp > test:
                test = temp
        if test < TOLX:
            return current_point, iteration, fret
        dg = grad
        grad, fret = funcd(current_point)
        test = 0.0
        den = np.maximum(fret, 1.0)
        for ix in range(0, n):
            temp = np.abs(grad[ix]) * np.maximum(abs(current_point[ix]), 1.0) / den
            if temp > test:
                test = temp
        if test < gtol:
            return current_point, iteration, fret

        for ix in range(0, n):
            dg[ix] = grad[ix] - dg[ix]

        for ix in range(0, n):
            hdg[ix] = 0.0
            for jx in range(0, n):
                hdg[ix] += hessian[ix, jx] * dg[jx]

        fac = fae = sumdg = sumxi = 0.0
        fac = np.sum(dg * x_initial)
        fae = np.sum(dg * hdg)
        sumdg = np.sum(dg * dg)
        sumxi = np.sum(x_initial * x_initial)

        if fac > np.sqrt(EPS * sumdg * sumxi):
            fac = 1.0 / fac
            fad = 1.0 / fae
            for ix in range(0, n):
                dg[ix] = fac * x_initial[ix] - fad * hdg[ix]
            for ix in range(0, n):
                for jx in range(ix, n):
                    hessian[ix, jx] += (
                        fac * x_initial[ix] * x_initial[jx]
                        - fad * hdg[ix] * hdg[jx]
                        + fae * dg[ix] * dg[jx]
                    )
                    hessian[jx, ix] = hessian[ix, jx]

        for ix in range(0, n):
            x_initial[ix] = 0.0
            for jx in range(0, n):
                x_initial[ix] -= hessian[ix, jx] * grad[jx]





