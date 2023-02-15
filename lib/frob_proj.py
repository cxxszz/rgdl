import numpy as np
import warnings
from copy import deepcopy as dc
import timeit


def dual_ascent(X: np.ndarray, p: np.ndarray, q: np.ndarray, eta=1e-1, max_iter=10000, stop_prec=2e-4):
    u, v = np.zeros(p.shape), np.zeros(q.shape)
    trans = np.outer(p, q)
    trans_max = max(np.max(p), np.max(q))
    for t in range(max_iter):
        trans_new = np.clip(X - u.reshape(-1, 1) - v, 0, trans_max)
        # print(t, trans)
        u = u + eta * (np.sum(trans, 1) - p)
        v = v + eta * (np.sum(trans, 0) - q)
        convergence = np.max(np.abs(
            np.sum(trans_new, 1) - p
        ))
        convergence += np.max(np.abs(
            np.sum(trans_new, 0) - q
        ))
        if convergence < stop_prec:
            return trans_new
        else:
            trans = trans_new
    if t == max_iter - 1:
        warnings.warn("Dual ascent not converged")
    return trans


def dual_ascent_bcd(X: np.ndarray, p: np.ndarray, q: np.ndarray, eta=1e-1, max_iter=10000, stop_prec=2e-4):
    u, v = np.zeros(p.shape), np.zeros(q.shape)
    trans_max = min(np.max(p), np.max(q))
    for t in range(max_iter):
        trans = np.clip(X - u.reshape(-1, 1) - v, 0, trans_max)
        # print(t, trans)
        u = u + eta * (np.sum(trans, 1) - p)
        v = v + eta * (np.sum(trans, 0) - q)
        convergence = np.max(np.abs(
            np.sum(trans, 1) - p
        ))
        convergence += np.max(np.abs(
            np.sum(trans, 0) - q
        ))
        if convergence < stop_prec:
            # print("normalize takes {} iterations".format(normalize_iter))
            break
    if t == max_iter - 1:
        warnings.warn("Dual ascent bcd not converged")
    return trans


def alm(X: np.ndarray, p: np.ndarray, q: np.ndarray, pen_wei=100, max_iter=10000, stop_prec=2e-4, eta_dual=1,
        max_iter_inner=10000, prec_inner=1e-6, eta_inner=None):
    def grad(trans: np.ndarray, u: np.ndarray, v: np.ndarray):
        vio1 = np.sum(trans, 1) - p
        vio2 = np.sum(trans, 0) - q
        return trans - X + u.reshape(-1, 1) + v + pen_wei * vio1.reshape(-1, 1) + pen_wei * vio2

    def nesterov(trans: np.ndarray, u: np.ndarray, v: np.ndarray):
        alpha, q = mu / L, mu / L
        trans_hat = trans
        for tau in range(max_iter_inner):
            trans_new = np.clip(trans_hat - eta * grad(trans=trans_hat, u=u, v=v), 0, trans_max)
            # print(trans_new)
            alpha_new = q - alpha ** 2 + np.sqrt((q - alpha ** 2) ** 2 + 4 * alpha ** 2)
            alpha_new = alpha_new / 2
            trans_hat_new = trans_new + alpha * (1 - alpha) / (alpha ** 2 + alpha_new) * (trans_new - trans)
            if np.max(np.abs(trans_new - trans)) < prec_inner:
                break
            else:
                trans = trans_new
                trans_hat = trans_hat_new
                alpha = alpha_new
        if tau == max_iter_inner - 1:
            warnings.warn("Nesterov subroutine has not converged")
        return trans

    ns, nt = len(p), len(q)
    mu, L = 1, 1 + pen_wei * (ns + nt)
    trans_max = min(np.max(p), np.max(q))
    if eta_inner is None:
        eta = 1 / L
    else:
        eta = eta_inner
    trans = dc(X)
    vio1 = np.sum(trans, 1) - p
    vio2 = np.sum(trans, 0) - q
    u, v = eta_dual * vio1, eta_dual * vio2
    for t in range(max_iter):
        trans_new = nesterov(trans=trans, u=u, v=v)
        # input("One call of nesterov finished")
        vio1 = np.sum(trans, 1) - p
        vio2 = np.sum(trans, 0) - q
        u_new = u + eta_dual * vio1
        v_new = v + eta_dual * vio2
        if np.max(np.abs(vio1)) + np.max(np.abs(vio2)) < stop_prec:
            break
        else:
            trans = trans_new
            u = u_new
            v = v_new
    if t == max_iter - 1:
        warnings.warn("ALM-based projection has not converged")
    return trans


if __name__ == '__main__':
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    p = np.ones(2) / 2
    q = np.ones(3) / 3
    print(dual_ascent(X=X, p=p, q=q))
    print(dual_ascent_bcd(X=X, p=p, q=q))
    print(alm(X=X, p=p, q=q, max_iter=10000))
    loop = 10
    print(timeit.timeit("dual_ascent(X=X, p=p, q=q)", globals=globals(), number=loop))
    print(timeit.timeit("dual_ascent_bcd(X=X, p=p, q=q)", globals=globals(), number=loop))
    print(timeit.timeit("alm(X=X, p=p, q=q, max_iter=10000)", globals=globals(), number=loop))
