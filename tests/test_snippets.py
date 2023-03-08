def test_dsp_construction() -> None:
    # Setup ###
    import numpy as np

    C = np.loadtxt("data/C.csv")
    n, T = C.shape

    # Snippet start ###
    import cvxpy as cp
    import dsp

    y = cp.Variable(T)
    s = cp.Variable(n)
    h = cp.Variable(n, nonneg=True)

    exponents = []
    weights = []
    for i in range(n):
        for t in range(T):
            if C[i, t] > 0:
                exponents.append(-(t + 1) * (y[t] + s[i]))
                weights.append(h[i] * C[i, t])

    Delta = dsp.weighted_log_sum_exp(cp.hstack(exponents), cp.hstack(weights))

    # Define phi, lamb, H, U ###
    phi = cp.norm1(h - np.ones(n) / n)
    lamb = 1
    H = [cp.sum(h) == 1]
    U = [0 <= y, y <= 0.2, 0 <= s, s <= 0.01]
    # ###

    obj = dsp.MinimizeMaximize(phi - lamb * Delta)

    constraints = H + U

    saddle_problem = dsp.SaddlePointProblem(obj, constraints)
    saddle_problem.solve()

    # Snippet end ###
    assert np.allclose(y.value, 0.2)
    assert np.allclose(s.value, 0.01)
    assert np.isclose(phi.value, 0.17965100137657167)


def test_worse_case_analysis() -> None:
    # Setup ###
    import numpy as np

    C = np.loadtxt("data/C.csv")
    n, T = C.shape
    h = np.ones(n) / n
    p = np.ones(n) * 100
    A = np.vstack([np.eye(n + T), -np.eye(n + T)])
    b = np.hstack([np.ones(n + T) * 0.05, np.zeros(n + T)])

    # Snippet start ###
    import cvxpy as cp
    import numpy as np

    y = cp.Variable(T)
    s = cp.Variable(n)

    V = h @ p

    exponents = []
    for i in range(n):
        for t_idx in range(T):
            t = t_idx + 1  # account for 0-indexing
            w_it = h[i] * C[i, t_idx]
            if w_it > 0:
                exponents.append(-t * (y[t_idx] + s[i]) + np.log(w_it))

    Delta = cp.log_sum_exp(cp.hstack(exponents)) - np.log(V)
    obj = cp.Minimize(Delta)
    prob = cp.Problem(obj, [A @ cp.hstack([y, s]) <= b])
    prob.solve()

    # Snippet end ###
    assert np.allclose(y.value, 0.05)
    assert np.allclose(s.value, 0.05)


def test_explicit_dual() -> None:
    # Setup ###
    import numpy as np

    C = np.loadtxt("data/C.csv")
    n, T = C.shape
    p = np.ones(n) * 100
    A = np.vstack([np.eye(n + T), -np.eye(n + T)])
    b = np.hstack([np.ones(n + T) * 0.05, np.zeros(n + T)])

    # Snippet start ###
    import cvxpy as cp
    import numpy as np

    F_1 = np.tile(np.eye(T), (n, 1))
    F_2 = np.repeat(np.eye(n), repeats=T, axis=0)
    F = np.hstack([F_1, F_2])

    lam = cp.Variable(len(b), nonneg=True)
    nu = cp.Variable(n * T, nonneg=True)

    h = cp.Variable(n, nonneg=True)

    B = 1

    term = 0
    for i in range(n):
        for t in range(1, T):
            nu_i_t = nu[i * T + t]
            term -= cp.rel_entr(C[i, t] * h[i], nu_i_t / t)

    obj = cp.Maximize(-lam @ b + term - np.log(B))
    constraints = [
        A.T @ lam == F.T @ nu,
        cp.sum(nu) == 1,
        p @ h == B,
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    # Snippet end ###
    assert np.isclose(h.value[-5], 0.01)
    assert np.isclose(h.value.sum(), 0.01)
