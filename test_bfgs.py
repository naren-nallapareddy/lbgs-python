import numpy as np
from utils import linesearch
from bfgs import bfgs
from scipy.optimize import rosen, minimize
from rich.pretty import pprint

if __name__ == "__main__":
    x0 = np.array([1.2, 0.7, 0.8, 1.9, 1.2])
    res = minimize(
        rosen, x0, method="BFGS", jac=False, options={"disp": False}
    )
    curr_point, itr, fvalue = bfgs(x0, 1e-5, 100, rosen)
    res_bfgs = {"x": curr_point, "nfev": itr, "fun": fvalue}
    pprint(res)
    pprint(res_bfgs)
