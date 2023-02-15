import numpy as np
from copy import deepcopy as dc
import warnings
from ot.optim import emd, solve_linesearch
from ot.gromov import fused_gromov_wasserstein
from ot.utils import unif
from lib.sinkhorn import peyre_expon, normalize, gromov_loss, gromov
from lib.util import pearson
from methods_ot.game import inner

add_prec = 1e-30
print_interval = 10


def cg_alternating(M: np.ndarray, ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, alpha: float,
                   n_stage=10, bound=0.2, trans_star=None, stop_prec=2e-4):
    trans = np.outer(ps, pt)
    for cg_iter in range(n_stage):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        # cg_call += 1
        # print("{}-th calculating the transport plan".format(cg_call))
        trans_new = fused_gromov_wasserstein(M=M, C1=ws, C2=wt + pert, p=ps, q=pt, loss_fun='square_loss', alpha=alpha,
                                             G0=trans, log=False, armijo=False)
        if np.max(np.abs(trans_new - trans)) < stop_prec:
            trans = trans_new
            yield trans, inner(trans=trans, ws=ws, wt=wt, bound=bound)
            break
        else:
            trans = trans_new
            yield trans, inner(trans=trans, ws=ws, wt=wt, bound=bound)


def cg(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, n_stage=200, bound=0.2, trans_star=None,
       stop_prec=2e-4):
    """
    CG utilizing the closed-form solution is inefficient.
    And because it takes many iterations for the transport plan sequence to converge, this further causes inaccuracy for the unmixing step.
    """
    ns, nt = len(ps), len(pt)
    numItermaxEmd = 100000
    trans = np.outer(ps, pt)
    f_val = gromov_loss(trans=trans, ws=ws, wt=wt, ps=ps, pt=pt)
    for stage_i in range(n_stage):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        r_wt = wt + pert

        def f(trans):
            return gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)

        def df(trans):
            return peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)

        old_fval = f_val
        Mi = df(trans)  # problem linearization
        Mi -= np.min(Mi)  # set Mi as positive
        trans_ora = emd(ps, pt, Mi, numItermax=numItermaxEmd, log=False)  # solve linear program
        delta_trans = trans_ora - trans
        constC = np.outer(ws ** 2 @ ps, np.ones(nt)) + np.outer(np.ones(ns), r_wt ** 2 @ pt)
        alpha, _, f_val = solve_linesearch(cost=f, G=trans, deltaG=delta_trans, Mi=Mi, f_val=f_val, armijo=False, C1=ws,
                                           C2=r_wt, reg=1, Gc=trans_ora, constC=constC, M=0, alpha_min=0, alpha_max=1)
        trans_new = trans + alpha * delta_trans
        if stage_i % print_interval == 0:
            # loss = gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)
            # print("loss={}".format(loss))
            if trans_star is not None:
                corr = pearson(trans, trans_star)
                print("corr={}".format(corr))
            yield trans_new, inner(trans_new, ws=ws, wt=wt, bound=bound)
        abs_convergence = abs(old_fval - f_val)
        rel_convergence = abs_convergence / f_val
        if np.max(np.abs(trans_new - trans)) < stop_prec or abs_convergence < stop_prec or rel_convergence < 1e-3:
            yield trans_new, inner(trans_new, ws=ws, wt=wt, bound=bound)
            break
        else:
            trans = trans_new
    if stage_i == n_stage - 1:
        print("has not converged")
    return trans, inner(trans, ws=ws, wt=wt, bound=bound)


def gd(ws: np.ndarray, wt: np.ndarray, Ys: np.ndarray, Yt: np.ndarray, alpha: float, ps: np.ndarray, pt: np.ndarray,
       n_stage=10000, bound=0.2, eta_trans=1, trans_star=None, stop_prec=2e-4, use_gd=False):
    # print("eta_trans={}".format(eta_trans))
    ns, nt = Ys.shape[0], Yt.shape[0]
    d = Ys.shape[1]
    assert d == Yt.shape[1]
    grad_w = (Ys ** 2) @ np.ones([d, nt]) + np.ones([ns, d]) @ (Yt.T ** 2) - 2 * Ys @ Yt.T
    # print("ws shape {}, wt shape {}, Ys shape {}, Yt shape {}, grad_w shape {}".format(ws.shape, wt.shape, Ys.shape,
    #                                                                                    Yt.shape, grad_w.shape))
    alpha_grad_w = (1 - alpha) * grad_w
    trans = np.outer(ps, pt)
    for stage_i in range(n_stage):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        # print("pert {}, {}, {}".format(np.min(pert), np.median(pert), np.max(pert)))
        # print("pert={}".format(pert))
        r_wt = wt + pert
        grad = peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)
        grad = alpha * grad + alpha_grad_w
        # print("grad calculated")
        if use_gd:
            tmp = np.clip(trans - eta_trans * grad, 0, 1) + add_prec
            trans_new = normalize(tmp, ps, pt, proj="fro", max_iter=10, stop_prec=2e-4)
        else:
            tmp = trans * np.exp(-1 - eta_trans * grad) + add_prec
            trans_new = normalize(tmp, ps, pt, max_iter=10, stop_prec=2e-4)
            # tmp = trans * np.exp(-eta_trans * grad) + add_prec
        # print("grad={}".format(grad))
        # input("tmp={}".format(tmp))
        del grad

        if np.max(np.abs(trans_new - trans)) < stop_prec:
            trans = trans_new
            yield trans, inner(trans, ws=ws, wt=wt, bound=bound)
            break
        else:
            trans = trans_new
            if stage_i % print_interval == 0:
                # loss = gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)
                # print("loss={}".format(loss))
                if trans_star is not None:
                    corr = pearson(trans, trans_star)
                    print("corr={}".format(corr))
                yield trans, inner(trans, ws=ws, wt=wt, bound=bound)
    if stage_i == n_stage - 1:
        warnings.warn("GD has not converged")


if __name__ == '__main__':
    ws = np.array([[0, 4, 9],
                   [4, 0, 1],
                   [9, 1, 0]])
    wt = dc(ws)
    Ys = np.array([[1, 1],
                   [0, 0],
                   [-1, -1]])
    Yt = np.array([[-1, -1],
                   [0, 0],
                   [1, 1]])
    ps = unif(3)
    pt = unif(3)
    alpha = 0.01
    for trans, pert in gd(ws=ws, wt=wt, Ys=Ys, Yt=Yt, alpha=alpha, ps=ps, pt=pt):
        pass
    print(trans)
    print("===" * 10)
    for trans, pert in gd(ws=ws, wt=wt, Ys=Ys, Yt=Yt, alpha=alpha, ps=ps, pt=pt, bound=2):
        pass
    print(trans)
    print(wt + pert)
    print("===" * 10)
    for trans, pert in gd(ws=ws, wt=wt, Ys=Ys, Yt=Yt, alpha=alpha, ps=ps, pt=pt, bound=2e10):
        pass
    print(trans)
    print(wt + pert)
