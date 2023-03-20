import numpy as np
from utils import linesearch
from lbfgs import lbfgs
from scipy.optimize import rosen, minimize
from rich.pretty import pprint
import matplotlib.pyplot as plt

# def simple_convex_function_2d(x):
# return x[0] ** 2 + x[1] ** 2

if __name__ == "__main__":
    # x = np.linspace(-2, 2, 100)
    # y = np.linspace(-2, 2, 100)
    # X, Y = np.meshgrid(x, y)
    # Z = simple_convex_function_2d([X, Y])
    # plt.contourf(X, Y, Z, 100)
    # plt.colorbar()
    # plt.show()

    x0 = np.array([3, 3, 3, 3, 3])
    res = minimize(
        rosen,
        x0,
        method="BFGS",
        jac=False,
        options={"disp": False},
    )
    res_lbfgs = lbfgs(rosen, x0, 1e-5, 100)

    pprint(res)
    pprint(res_lbfgs)
