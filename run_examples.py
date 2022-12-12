from functools import partial

import cvxpy as cp
import numpy as np

from src.full_example import (
    in_ellipse,
    plot_heatmap,
    plot_historic,
    portfolio_construction,
    worst_case_analysis,
)
from src.robust_optimization import UncertaintySet

try:
    import dsp  # noqa: F401

    RUN_CONSTRUCTION = True
except ImportError:
    RUN_CONSTRUCTION = False


def run() -> None:
    # load data
    Cash_flows = np.loadtxt("data/C.csv")
    target_weights = np.loadtxt("data/target_weights.csv")
    F = np.loadtxt("data/F.csv")
    P_hat_inv = np.loadtxt("data/sigma_joint_inverse.csv")
    x_hat = np.loadtxt("data/mu_joint.csv")
    prices = np.loadtxt("data/p.csv")
    y_nominal = np.loadtxt("data/y_nominal.csv")
    s_nominal = np.loadtxt("data/s_nominal.csv")
    mean_yields = np.loadtxt("data/mean_yields.csv")
    mean_spreads = np.loadtxt("data/mean_spreads.csv")

    plot_heatmap(target_weights)
    plot_historic(mean_yields, mean_spreads, y_nominal, s_nominal)

    T = Cash_flows.shape[1]
    n = Cash_flows.shape[0]

    y_tilde = cp.Variable(T, name="y_tilde")
    s_tilde = cp.Variable(n, name="s_tilde")

    alpha_modest = 0.5
    ellipse_callable = partial(
        in_ellipse, P_hat_inv=P_hat_inv, x_hat=x_hat, F=F, alpha=alpha_modest
    )
    U_modest = UncertaintySet(y_s_const=[ellipse_callable])

    alpha_extreme = 0.01
    ellipse_callable = partial(
        in_ellipse, P_hat_inv=P_hat_inv, x_hat=x_hat, F=F, alpha=alpha_extreme
    )
    U_extreme = UncertaintySet(y_s_const=[ellipse_callable])

    ys_nominal = (y_nominal, s_nominal)
    worst_case_analysis(
        Cash_flows,
        U_extreme,
        U_modest,
        prices,
        s_nominal,
        target_weights,
        y_nominal,
        ys_nominal,
        plot=True,
    )

    if RUN_CONSTRUCTION:
        portfolio_construction(
            Cash_flows, prices, s_tilde, target_weights, y_tilde, U_extreme, U_modest, ys_nominal
        )
    else:
        print("Skipping portfolio construction example, requires dsp package to be installed.")


if __name__ == "__main__":
    run()
