import numpy as np
import ot
from ot.utils import unif
import warnings
from lib.matmul import diag_matmul_np, matmul_diag_np
from lib.frob_proj import dual_ascent_bcd as dual_ascent


def sinkhorn(C: np.ndarray, p: np.ndarray, q: np.ndarray, reg: float, max_iter=100, stop_prec=1e-6, add_prec=1e-30):
    C_norm = C / np.max(np.abs(C))  # normalized
    K = np.exp(-C_norm / reg)
    return scaling(K=K, p=p, q=q, max_iter=max_iter, stop_prec=stop_prec, add_prec=add_prec)


def scaling(K: np.ndarray, p: np.ndarray, q: np.ndarray, max_iter=100, stop_prec=1e-6, add_prec=1e-30):
    """
    has to ensure K is non-negative
    """
    u = np.ones(p.shape)
    for scaling_iter in range(max_iter):
        v = q / (K.T @ u + add_prec)
        u = p / (K @ v + add_prec)
        tmp = diag_matmul_np(u, K)
        trans = matmul_diag_np(tmp, v)
        convergence = np.max(np.abs(
            np.sum(trans, 1) - p
        ))
        convergence += np.max(np.abs(
            np.sum(trans, 0) - q
        ))
        if convergence < stop_prec:
            break
    if scaling_iter == max_iter - 1:
        warnings.warn("Scaling has not converged")
    return trans


def sinkhorn_logsum(C: np.ndarray, p: np.ndarray, q: np.ndarray, reg: float, max_iter=100, stop_prec=1e-6,
                    add_prec=1e-30):
    def soft_min(x: np.ndarray, eps: float):
        tmp = np.exp(-x / eps)
        return -eps * np.log(np.sum(tmp) + add_prec)

    C_norm = C / np.max(np.abs(C))  # normalized
    ns, nt = len(p), len(q)
    K = np.exp(-C_norm / reg)
    f = np.random.uniform(0, 1, ns)
    g = np.random.uniform(0, 1, nt)
    for t in range(max_iter):
        for i in range(ns):
            f[i] = soft_min(C_norm[i] - g, eps=reg) + reg * np.log(p[i])
        for j in range(nt):
            g[j] = soft_min(C_norm[:, j] - f, eps=reg) + reg * np.log(q[j])

        u = np.exp(f / reg)
        v = np.exp(g / reg)
        tmp = np.diag(u) @ K
        trans = tmp @ np.diag(v)
        convergence = np.max(np.abs(
            np.sum(trans, 1) - p
        ))
        convergence += np.max(np.abs(
            np.sum(trans, 0) - q
        ))
        if convergence < stop_prec:
            break
    if t == max_iter - 1:
        warnings.warn("Sinkhorn logsum has not converged")
    return trans


def peyre_expon(trans: np.ndarray, ps: np.ndarray, pt: np.ndarray, ws: np.array, wt: np.array):
    """

    :param ws:
    :param wt:
    :param trans:
    :param ps:
    :param pt:
    :return:
    """
    ns, nt = len(ps), len(pt)
    deg_terms = np.outer(ws ** 2 @ ps, np.ones(nt))
    deg_terms += np.outer(np.ones(ns), wt ** 2 @ pt)
    num = ws @ trans @ wt
    return deg_terms - 2 * num


def gromov(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, stepsize=100, max_iter=100000,
           add_prec=1e-30, stop_prec=5e-7, init=None, use_gd=False):
    if init is None:
        trans = np.outer(ps, pt)
    else:
        trans = init
    for iter_num in range(max_iter):
        # print("calculating grad")
        grad = peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=wt)
        # print("grad calculated")
        if use_gd:
            tmp = trans - stepsize * grad
        else:
            tmp = trans * np.exp(-1 - stepsize * grad) + add_prec
        del grad
        trans_new = scaling(tmp, ps, pt)
        if np.max(np.abs(trans_new - trans)) < stop_prec:
            break
        else:
            trans = trans_new
    if iter_num == max_iter - 1:
        print("has not converged")
    return trans


def gromov_loss(trans: np.ndarray, ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray):
    grad = peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=wt)
    return np.sum(grad * trans)


if __name__ == '__main__':
    C = np.array([[1, 2, 3],
                  [4, 5, 6]])
    ps, pt = unif(2), unif(3)
    reg = 1e-3
    print(sinkhorn(C=C, p=ps, q=pt, reg=reg))
    print(sinkhorn_logsum(C=C, p=ps, q=pt, reg=reg))
