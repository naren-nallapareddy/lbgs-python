# LBFGS utilities for optimization from Numerical recipes: the art of scientific computing
# Author: Naren Nallapareddy
import numpy as np
from typing import TypeVar, Generic, Callable, Any
import math


def linesearch(
    x_old: np.ndarray,
    f_old: float,
    grad: np.ndarray,
    search_direction: np.ndarray,
    step_max: float,
    func,
):
    """
    Line search method taken from page 479, numerical recipes: the art of scientific computing

    Parameters:
    ----------
    x_old: n-dimensional ppoint [0,...,n-1]
    f_old: function value at xold
    g: gradient at xold
    p: search direction
    step_max: maximum step length
    func: callable function

    Return:
    ------
    a_lambda: step length
    x: n-dimensional point [0,...,n-1]
    f: function value at x
    check: output check is false on normal exit, and it is true when x is too close to xold
    """
    ALF = 1e-4
    # TOLX = np.finfo(
    # float
    # ).eps  # ALF ensures sufficient decrease in function value, TOLX is the convergence criterion on \nabla x
    TOLX = 1e-4
    a_lambda_2 = 0.0
    f2 = 0.0
    slope = 0.0
    check = False

    n = len(x_old)  # Length of the old input vector

    sum = np.sqrt(
        np.sum(search_direction**2)
    )  # sum = sum over search direction

    if sum > step_max:
        search_direction = (
            search_direction * step_max / sum
        )  # rescale if attempted step is too big

    slope = np.dot(
        grad, search_direction
    )  # slope = dot product of gradient and search direction
    assert slope < 0, "Roundoff problem in linesearch."

    test = np.max(np.abs(search_direction) / np.maximum(np.abs(x_old), 1.0))
    ##! Literal translation of the for loop below
    # test = 0.0  # computing lambda min
    # for ix in range(0, n):
    # temp = np.abs(search_direction[ix]) / np.maximum(
    # np.abs(x_old[ix]), 1.0
    # )
    # if temp > test:
    # test = temp
    ##!
    a_lambda_min = TOLX / test  # type: ignore
    a_lambda = 1.0  # try full Newton step first
    while True:
        x = x_old + a_lambda * search_direction
        f = func(x)
        if a_lambda < a_lambda_min:
            x = np.copy(x_old)
            check = True
            return a_lambda, x, f, check  # check is true

        elif f <= f_old + ALF * a_lambda * slope:
            check = False
            return a_lambda, x, f, check

        else:
            if a_lambda == 1.0:
                temp_lambda = -slope / (2.0 * (f - f_old - slope))
            else:
                rhs1 = f - f_old - a_lambda * slope
                rhs2 = f2 - f_old - a_lambda_2 * slope  # type: ignore
                a = (rhs1 / (a_lambda**2) - rhs2 / (a_lambda_2**2)) / (
                    a_lambda - a_lambda_2
                )
                b = (
                    -a_lambda_2 * rhs1 / (a_lambda**2)
                    + a_lambda * rhs2 / (a_lambda_2**2)
                ) / (a_lambda - a_lambda_2)
                if a == 0.0:
                    temp_lambda = -slope / (2.0 * b)
                else:
                    discriminant = b**2 - 3.0 * a * slope
                    if discriminant < 0.0:
                        temp_lambda = 0.5 * a_lambda
                    elif b <= 0.0:
                        temp_lambda = (-b + np.sqrt(discriminant)) / (3.0 * a)
                    else:
                        temp_lambda = -slope / (b + np.sqrt(discriminant))
                temp_lambda = np.minimum(temp_lambda, 0.5 * a_lambda)
        a_lambda_2 = a_lambda
        f2 = f
        a_lambda = np.maximum(
            temp_lambda, 0.01 * a_lambda
        )  # lambda is at least 10% of the old lambda


T = TypeVar("T", bound=Callable)


class Funcd(Generic[T]):
    """
    Class for function evaluation and gradient evaluation using finite difference.
    """

    def __init__(self, func: Callable):
        self.func = func
        self.EPS = np.sqrt(np.finfo(float).eps)
        self.f = None

    def __call__(self, x: np.ndarray):
        self.f = self.func(x)
        return self.f

    def df(self, x: np.ndarray):
        n = len(x)
        xh = np.copy(x)
        fold = self.f
        df = np.zeros_like(x)
        for ix in range(0, n):
            temp = x[ix]
            h = self.EPS * math.fabs(
                temp
            )  # temp is a float, no need to introduce numpy
            if h == 0.0:
                h = self.EPS
            xh[ix] = temp + h
            h = xh[ix] - temp
            fh = self.__call__(xh)
            xh[ix] = temp
            df[ix] = (fh - fold) / h
        return df
