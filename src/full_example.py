from __future__ import annotations

import cvxpy as cp
import dspp
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from cvxpy.constraints.constraint import Constraint
from matplotlib.ticker import FuncFormatter
from scipy.stats import chi2

from robust_optimization import UncertaintySet, get_Delta_lin, run_worst_case_analysis


def plot_yields_and_spreads(
    y_wc_m: np.ndarray,
    s_wc_m: np.ndarray,
    y_wc_e: np.ndarray,
    s_wc_e: np.ndarray,
    y_hat_wc_m: np.ndarray,
    s_hat_wc_m: np.ndarray,
    y_hat_wc_e: np.ndarray,
    s_hat_wc_e: np.ndarray,
    y_nominal: np.ndarray,
    s_nominal: np.ndarray,
) -> None:
    n = len(s_nominal)

    plt.rc("ytick", labelsize=10)
    plt.rc("ytick", labelsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), width_ratios=[9, 4], dpi=100)

    axes[0].plot(y_wc_m * 2, label=r"exact $\alpha=50\%$", color="tab:green")
    axes[0].plot(
        y_hat_wc_m * 2, label=r"linearized $\alpha=50\%$", color="tab:green", linestyle="--"
    )
    axes[0].plot(y_wc_e * 2, label=r"exact $\alpha=99\%$", color="tab:orange")
    axes[0].plot(
        y_hat_wc_e * 2, label=r"linearized $\alpha=99\%$", color="tab:orange", linestyle="--"
    )
    axes[0].plot(y_nominal * 2, label="nominal", color="tab:blue")
    axes[0].set_ylabel("Annualized yield")
    axes[0].set_xlabel("Period")
    axes[0].yaxis.set_major_formatter(FuncFormatter("{:.1%}".format))

    for i in range(4):
        bond_range = list(range(n))[i * 5 : (i + 1) * 5]
        axes[1].plot(bond_range, s_wc_m[bond_range] * 2, color="tab:green")
        axes[1].plot(bond_range, s_hat_wc_m[bond_range] * 2, color="tab:green", linestyle="--")
        axes[1].plot(bond_range, s_wc_e[bond_range] * 2, color="tab:orange")
        axes[1].plot(bond_range, s_hat_wc_e[bond_range] * 2, color="tab:orange", linestyle="--")
        axes[1].plot(bond_range, s_nominal[bond_range] * 2, color="tab:blue")

    axes[1].set_ylabel("Spread")
    axes[1].set_xlabel("Rating")
    axes[1].yaxis.set_major_formatter(FuncFormatter("{:.1%}".format))

    axes[1].set_xticks([2, 7, 12, 17])
    axes[1].set_xticklabels(["AAA", "AA", "A", "BBB"])

    fig.subplots_adjust(bottom=0.25, wspace=0.3)
    fig.legend(loc="lower center", ncol=3)

    plt.savefig("figures/worst_case_yields.pdf")
    plt.show()
    plt.rcdefaults()


def plot_historic(
    mean_yields: np.ndarray, mean_spreads: np.ndarray, y_nominal: np.ndarray, s_nominal: np.ndarray
) -> None:
    n = len(s_nominal)

    plt.rc("ytick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    lw = 3

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), width_ratios=[9, 4], dpi=100)

    axes[0].plot(mean_yields * 2, label=r"$\mu^{\mathrm{hist}}$", color="darkgrey", linewidth=lw)
    axes[0].plot(y_nominal * 2, label="nominal yields", color="tab:blue", linewidth=lw)
    axes[0].set_ylabel("Annualized yield")
    axes[0].set_xlabel("Period")
    axes[0].yaxis.set_major_formatter(FuncFormatter("{:.1%}".format))
    axes[0].legend()

    for i in range(4):
        bond_range = list(range(n))[i * 5 : (i + 1) * 5]
        label = r"$\mu^{\mathrm{hist}}$" if i == 0 else None
        axes[1].plot(
            bond_range, mean_spreads[bond_range] * 2, color="darkgrey", label=label, linewidth=lw
        )
        label = "nominal\nspreads" if i == 0 else None
        axes[1].plot(
            bond_range, s_nominal[bond_range] * 2, color="tab:blue", label=label, linewidth=lw
        )

    axes[1].set_ylabel("Spread")
    axes[1].set_xlabel("Rating")
    axes[1].yaxis.set_major_formatter(FuncFormatter("{:.1%}".format))
    axes[1].set_xticks([2, 7, 12, 17])
    axes[1].set_xticklabels(["AAA", "AA", "A", "BBB"])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("figures/mean_yields_spreads.pdf")
    plt.show()
    plt.rcdefaults()


def plot_turnover_distance(
    modest: pd.DataFrame,
    extreme: pd.DataFrame,
    modest_linearized: pd.DataFrame,
    extreme_linearized: pd.DataFrame,
) -> None:

    plt.figure(figsize=(6, 4), dpi=100)

    plt.plot(modest["lambda"], modest.phi, label=r"exact $\alpha=50\%$", color="tab:green")
    plt.plot(
        modest_linearized["lambda"],
        modest_linearized.phi,
        label=r"linearized $\alpha=50\%$",
        color="tab:green",
        linestyle="--",
    )
    plt.plot(extreme["lambda"], extreme.phi, label=r"exact $\alpha=99\%$", color="tab:orange")
    plt.plot(
        extreme_linearized["lambda"],
        extreme_linearized.phi,
        label=r"linearized $\alpha=99\%$",
        color="tab:orange",
        linestyle="--",
    )

    plt.ylabel(r"$\phi$")
    plt.xlabel(r"$\lambda$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/turnover_distance.pdf")
    plt.show()


def plot_heatmaps(modest: pd.DataFrame, extreme: pd.DataFrame) -> None:

    factor = 1.5
    fig, axes = plt.subplots(3, 2, figsize=(12 / factor, 14 / factor), dpi=100)

    sorted_rat = ["AAA", "AA", "A", "BBB"]
    sorted_mat = ["<3", "3-5", "5-10", "10-20", ">20"]
    lambdas = [1, 5, 15]

    new_cmap = colors.LinearSegmentedColormap.from_list(
        "new_cmap",
        [
            (0, (0.267004, 0.004874, 0.329415, 1.0)),
            (0.1, (0.127568, 0.566949, 0.550556, 1.0)),
            (1.0, (0.993248, 0.906157, 0.143936, 1.0)),
        ],
    )

    for i, lamb in enumerate(lambdas):
        wgts_modest = modest.loc[modest["lambda"] == lamb, "weights"].values[0].reshape((4, -1))

        ax = axes[i, 0]
        ax = sns.heatmap(
            wgts_modest,
            vmin=0,
            vmax=1,
            cmap=new_cmap,
            xticklabels=sorted_mat,
            yticklabels=sorted_rat,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Maturity in years")
        ax.set_ylabel("Rating")
        ax.set_title(r"$\alpha=50\%$, $\lambda={}$".format(lamb))

        wgts_extreme = extreme.loc[extreme["lambda"] == lamb, "weights"].values[0].reshape((4, -1))
        ax = axes[i, 1]
        ax = sns.heatmap(
            wgts_extreme,
            vmin=0,
            vmax=1,
            cmap=new_cmap,
            xticklabels=sorted_mat,
            yticklabels=sorted_rat,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Maturity in years")
        ax.set_ylabel("Rating")
        ax.set_title(r"$\alpha=99\%$, $\lambda={}$".format(lamb))

    def formatter(x: float, _pos: int) -> str:
        return "{:.0%}".format(x)

    fig.subplots_adjust(right=0.8, hspace=0.5)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mpl.cm.ScalarMappable(cmap=new_cmap), cax=cbar_ax, format=FuncFormatter(formatter))

    plt.savefig("figures/weight_heatmaps.pdf", bbox_inches="tight")
    plt.show()


def portfolio_construction(
    Cash_flows: np.ndarray,
    prices: np.ndarray,
    s_tilde: np.ndarray,
    target_weights: np.ndarray,
    y_tilde: cp.Variable,
    U_extreme: UncertaintySet,
    U_modest: UncertaintySet,
    ys_nominal: tuple[np.ndarray, np.ndarray],
) -> None:

    print("modest change")
    modest = run_portfolio_construction(
        Cash_flows, prices, s_tilde, target_weights, y_tilde, U_modest
    )

    print("extreme change")
    extreme = run_portfolio_construction(
        Cash_flows, prices, s_tilde, target_weights, y_tilde, U_extreme
    )

    print("modest change linear")
    modest_linear = run_portfolio_construction(
        Cash_flows,
        prices,
        s_tilde,
        target_weights,
        y_tilde,
        U_modest,
        linearized=True,
        ys_nominal=ys_nominal,
    )

    print("extreme change linear")
    extreme_linear = run_portfolio_construction(
        Cash_flows,
        prices,
        s_tilde,
        target_weights,
        y_tilde,
        U_extreme,
        linearized=True,
        ys_nominal=ys_nominal,
    )

    plot_turnover_distance(modest, extreme, modest_linear, extreme_linear)
    plot_heatmaps(modest, extreme)


def run_portfolio_construction(
    Cash_flows: np.ndarray,
    prices: np.ndarray,
    s_tilde: cp.Variable,
    target_weights: np.ndarray,
    y_tilde: cp.Variable,
    U: UncertaintySet,
    linearized: bool = False,
    ys_nominal: tuple[np.ndarray, np.ndarray] = None,
) -> pd.DataFrame:

    n, T = Cash_flows.shape
    h = cp.Variable(n, name="h")

    # Objective
    phi = cp.norm1(target_weights - cp.multiply(h, prices)) / 2

    weight_map, y_tilde_map, s_tilde_map, exponent_offset = get_cashflow_mapping(Cash_flows)
    n_exponents = len(exponent_offset)

    exponents = cp.Variable(n_exponents, name="exponents")
    weights = cp.Variable(n_exponents, nonneg=True, name="weights")

    exponent_constraints = [
        *U.contains((y_tilde, s_tilde)),
        exponents == y_tilde_map @ y_tilde + s_tilde_map @ s_tilde + exponent_offset,
    ]

    B = 1

    weight_constraints = [h >= 0, h @ prices == B, weights == weight_map @ h]

    res = []
    for lambda_val in np.linspace(0.0, 20.0, 41):
        if not linearized:
            Delta = dspp.weighted_log_sum_exp(exponents, weights)
        else:
            Delta = get_Delta_lin(
                h, y_tilde, s_tilde, Cash_flows, *ys_nominal, B, construction=True
            )
        saddle_problem = dspp.SaddleProblem(
            dspp.MinimizeMaximize(phi - lambda_val * Delta),
            weight_constraints + exponent_constraints,
        )
        saddle_problem.solve(solver=cp.MOSEK, eps=1e-2)

        print(f"{lambda_val:.2f}, " f"{phi.value=:.2f}")
        res.append(
            {
                "lambda": lambda_val,
                "phi": phi.value,
                "weights": h.value * prices,
            }
        )

    return pd.DataFrame(res)


def get_cashflow_mapping(
    Cash_flows: np.ndarray,
) -> tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix, np.ndarray]:
    n, T = Cash_flows.shape

    sprs_C = sp.coo_matrix(Cash_flows)
    nnz = sprs_C.nnz

    h_inds, t_inds, C_vals = sprs_C.row, sprs_C.col, sprs_C.data

    weight_map = sp.coo_matrix((np.ones(nnz), (np.arange(nnz), h_inds)), shape=(nnz, n))

    neg_t_plus_1 = -(t_inds + 1)

    y_tilde_map = sp.coo_matrix((neg_t_plus_1, (np.arange(nnz), t_inds)), shape=(nnz, T))
    s_tilde_map = sp.coo_matrix((neg_t_plus_1, (np.arange(nnz), h_inds)), shape=(nnz, n))
    exponent_offset = np.log(C_vals)

    return weight_map, y_tilde_map, s_tilde_map, exponent_offset


def worst_case_analysis(
    Cash_flows: np.ndarray,
    U_extreme: UncertaintySet,
    U_modest: UncertaintySet,
    prices: np.ndarray,
    s_nominal: np.ndarray,
    target_weights: np.ndarray,
    y_nominal: np.ndarray,
    ys_nominal: tuple[np.ndarray, np.ndarray],
    plot: bool = False,
) -> None:
    print(" Worst case analysis ".center(40, "#"), end="\n\n")

    print(" modest change ".center(40, "#"))
    Delta_wc_m, y_wc_m, s_wc_m, Delta_hat_wc_m, y_hat_wc_m, s_hat_wc_m = run_worst_case_analysis(
        Cash_flows, U_modest, target_weights / prices, prices, ys_nominal
    )

    print(" extreme change ".center(40, "#"))
    Delta_wc_e, y_wc_e, s_wc_e, Delta_hat_wc_e, y_hat_wc_e, s_hat_wc_e = run_worst_case_analysis(
        Cash_flows, U_extreme, target_weights / prices, prices, ys_nominal
    )
    if plot:
        plot_yields_and_spreads(
            y_wc_m,
            s_wc_m,
            y_wc_e,
            s_wc_e,
            y_hat_wc_m,
            s_hat_wc_m,
            y_hat_wc_e,
            s_hat_wc_e,
            y_nominal,
            s_nominal,
        )


def plot_heatmap(weights: np.array) -> None:
    matrix = weights.reshape((4, -1))
    plt.figure()
    sorted_rat = ["AAA", "AA", "A", "BBB"]
    sorted_mat = ["<3", "3-5", "5-10", "10-20", ">20"]
    ax = sns.heatmap(
        matrix,
        vmin=0,
        cmap="viridis",
        xticklabels=sorted_mat,
        yticklabels=sorted_rat,
        annot=True,
        fmt=".1%",
        cbar=False,
    )
    ax.set_xlabel("Maturity in years")
    ax.set_ylabel("Rating")
    plt.tight_layout()
    plt.savefig("figures/nominal_portfolio.pdf")
    plt.show()


def in_ellipse(
    y_tilde: cp.Variable,
    s_tilde: cp.Variable,
    P_hat_inv: np.ndarray,
    x_hat: np.ndarray,
    F: np.ndarray,
    alpha: float = 0.05,
) -> list[Constraint]:
    rhs = chi2.ppf(1 - alpha, df=P_hat_inv.shape[0])
    x = cp.Variable(P_hat_inv.shape[0])
    return [cp.quad_form((x - x_hat), P_hat_inv) <= rhs, F @ x == cp.hstack([y_tilde, s_tilde])]
