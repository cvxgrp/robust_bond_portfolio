from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

if TYPE_CHECKING:
    from dsp import inner


@dataclass
class UncertaintySet:
    y_const: list[Callable] | None = None
    s_const: list[Callable] | None = None
    y_s_const: list[Callable] | None = None

    def __post_init__(self) -> None:
        assert any([self.y_const is not None, self.s_const is not None, self.y_s_const is not None])

    def contains(self, ys: tuple[cp.Variable, cp.Variable]) -> list[Constraint]:
        y_tilde, s_tilde = ys
        c_constraints = [c(y_tilde) for c in self.y_const] if self.y_const is not None else []
        s_constraints = [c(s_tilde) for c in self.s_const] if self.s_const is not None else []
        y_s_constraints = (
            [c(y_tilde, s_tilde) for c in self.y_s_const] if self.y_s_const is not None else []
        )
        all_constraints = c_constraints + s_constraints + y_s_constraints
        all_iterable_constraints = [make_iterable(x) for x in all_constraints]
        flat_constraints = list(itertools.chain.from_iterable(all_iterable_constraints))
        return flat_constraints

    def has_maximum_element(self, ys: tuple[cp.Variable, cp.Variable]) -> bool:
        y_tilde, s_tilde = ys
        T, n = y_tilde.shape[0], s_tilde.shape[0]

        y_tilde_componentwise_max = np.zeros(T)
        for t in range(T):
            problem = cp.Problem(cp.Maximize(y_tilde[t]), self.contains((y_tilde, s_tilde)))
            problem.solve()
            y_tilde_componentwise_max[t] = y_tilde.value[t]

        s_tilde_componentwise_max = np.zeros(n)
        for i in range(n):
            problem = cp.Problem(cp.Maximize(s_tilde[i]), self.contains((y_tilde, s_tilde)))
            problem.solve()
            s_tilde_componentwise_max[i] = s_tilde.value[i]

        problem = cp.Problem(
            cp.Maximize(0),
            self.contains((y_tilde, s_tilde))
            + [y_tilde == y_tilde_componentwise_max, s_tilde == s_tilde_componentwise_max],
        )

        problem.solve()
        assert problem.status in {cp.OPTIMAL, cp.INFEASIBLE}
        return problem.status != cp.INFEASIBLE

    def get_A_c(self, ys: tuple[cp.Variable, cp.Variable]) -> tuple[np.ndarray, np.ndarray]:
        y_tilde, s_tilde = ys
        prob = cp.Problem(cp.Minimize(0), self.contains((y_tilde, s_tilde)))
        problem_data = prob.get_problem_data(solver=cp.SCS)
        return problem_data[0]["A"].toarray(), problem_data[0]["b"]


def make_iterable(x: Any | Iterable[Any]) -> Iterable[Any]:
    if not isinstance(x, Iterable):
        return [x]
    else:
        return x


def exact_worst_case_analysis(
    C: np.array, U: UncertaintySet, h: np.array, p: np.array
) -> tuple[float, np.array, np.array]:
    n, T = C.shape

    y_tilde = cp.Variable(T)
    s_tilde = cp.Variable(n)

    V_y_s = h @ p

    exponents = []
    for i in range(n):
        for t in range(T):
            c_it = C[i, t]
            if c_it > 0:
                exponents.append(-(t + 1) * (y_tilde[t] + s_tilde[i]) + np.log(h[i] * c_it))

    Delta_y_tilde_s_tilde = cp.log_sum_exp(cp.hstack(exponents)) - np.log(V_y_s)
    obj = cp.Minimize(Delta_y_tilde_s_tilde)
    prob = cp.Problem(obj, U.contains((y_tilde, s_tilde)))
    prob.solve(solver=cp.MOSEK)
    assert prob.status == cp.OPTIMAL
    return prob.value, y_tilde.value, s_tilde.value


def linearized_worst_case_analysis(
    C: np.array, U: UncertaintySet, h: np.array, p: np.array, ys_nominal: tuple[np.array, np.array]
) -> tuple[float, np.array, np.array]:
    n, T = C.shape

    y_nominal, s_nominal = ys_nominal

    y_tilde = cp.Variable(T)
    s_tilde = cp.Variable(n)

    Delta_hat_y_tilde_s_tilde = get_Delta_lin(h, y_tilde, s_tilde, C, y_nominal, s_nominal, p @ h)
    obj = cp.Minimize(Delta_hat_y_tilde_s_tilde)
    prob = cp.Problem(obj, U.contains((y_tilde, s_tilde)))
    prob.solve(solver=cp.MOSEK)
    assert prob.status == cp.OPTIMAL
    return prob.value, y_tilde.value, s_tilde.value


def get_Delta_lin(
    h: np.ndarray | cp.Variable,
    y_tilde: cp.Variable,
    s_tilde: cp.Variable,
    C: np.ndarray,
    y_nominal: np.ndarray,
    s_nominal: np.ndarray,
    V_y_s: float,
    construction: bool = False,
) -> inner | cp.Expression:
    n, T = C.shape

    # TODO: Vectorize
    tmp = []
    for t in range(T):
        tmp.append(
            (
                -(t + 1)
                * cp.sum(
                    [h[i] * C[i, t] * np.exp(-t * (y_nominal[t] + s_nominal[i])) for i in range(n)]
                )
            )
            / V_y_s
        )
    D_yld = cp.hstack(tmp)

    tmp = []
    for i in range(n):
        tmp.append(
            (
                cp.sum(
                    [
                        -(t + 1) * h[i] * C[i, t] * np.exp(-t * (y_nominal[t] + s_nominal[i]))
                        for t in range(T)
                    ]
                )
            )
            / V_y_s
        )
    D_spr = cp.hstack(tmp)

    if construction:
        from dsp import inner

        return inner(y_tilde - y_nominal, D_yld) + inner(s_tilde - s_nominal, D_spr)
    else:
        return D_yld.value @ (y_tilde - y_nominal) + D_spr.value @ (s_tilde - s_nominal)


def run_worst_case_analysis(
    C: np.array, U: UncertaintySet, h: np.array, p: np.array, ys_nominal: tuple[np.array, np.array]
) -> tuple[float, np.array, np.array, float, np.array, np.array]:

    Delta_wc, y_wc, s_wc = exact_worst_case_analysis(C, U, h, p)
    print("Exact worst case analysis:")
    print(f"log return: {Delta_wc:.4%}, simple return {np.exp(Delta_wc)-1:.4%}")
    Delta_hat_wc, y_hat_wc, s_hat_wc = linearized_worst_case_analysis(C, U, h, p, ys_nominal)
    print("Linearized worst case analysis:")
    print(f"log return: {Delta_hat_wc:.4%}, simple return {np.exp(Delta_hat_wc)-1:.4%}")
    return Delta_wc, y_wc, s_wc, Delta_hat_wc, y_hat_wc, s_hat_wc
