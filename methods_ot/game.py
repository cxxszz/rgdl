import numpy as np
import warnings
from copy import deepcopy as dc
import ot
from ot.optim import emd, solve_linesearch
from ot.gromov import gromov_wasserstein
from lib.sinkhorn import peyre_expon, scaling, gromov_loss, sinkhorn
from lib.util import pearson
from lib.frob_proj import dual_ascent, dual_ascent_bcd

add_prec = 1e-30
print_interval = 1


def inner(trans: np.ndarray, ws: np.ndarray, wt: np.ndarray, bound: float):
    turn = trans.T @ ws @ trans - wt * (trans.T @ trans)
    return bound * (turn <= 0).astype(float) - bound * (turn > 0).astype(float)


def cg_alternating(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, n_stage=10, bound=0.2,
                   trans_star=None, stop_prec=2e-4):
    trans = np.outer(ps, pt)
    for cg_iter in range(n_stage):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        # cg_call += 1
        # print("{}-th calculating the transport plan".format(cg_call))
        trans_new = gromov_wasserstein(C1=ws, C2=wt + pert, p=ps, q=pt, loss_fun='square_loss',
                                       G0=trans, log=False, armijo=False)
        if np.max(np.abs(trans_new - trans)) < stop_prec:
            trans = trans_new
            yield trans, inner(trans=trans, ws=ws, wt=wt, bound=bound)
            break
        else:
            trans = trans_new
            yield trans, inner(trans=trans, ws=ws, wt=wt, bound=bound)


def cg(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, max_iter=200, bound=0.2, trans_star=None,
       stop_prec=2e-4, lp_ora=True, max_proj_iter=100000, proj_prec=1e-7, reg=1e-2):
    """
    CG utilizing the closed-form solution is inefficient.
    And because it takes many iterations for the transport plan sequence to converge, this further causes inaccuracy for the unmixing step.
    """
    ns, nt = len(ps), len(pt)
    trans = np.outer(ps, pt)
    f_val = gromov_loss(trans=trans, ws=ws, wt=wt, ps=ps, pt=pt)
    for iter_num in range(max_iter):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        r_wt = wt + pert

        def f(trans):
            return gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)

        def df(trans):
            return peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)

        old_fval = f_val
        Mi = df(trans)  # problem linearization
        Mi -= np.min(Mi)  # set Mi as positive
        if lp_ora:
            trans_ora = emd(ps, pt, Mi, numItermax=max_proj_iter, log=False)  # solve linear program
        else:
            Mi = Mi / np.max(Mi)
            # trans_ora = ot.bregman.sinkhorn(a=ps, b=pt, M=Mi, reg=1e-2)  # solve linear program
            trans_ora = sinkhorn(C=Mi, p=ps, q=pt, reg=reg, max_iter=max_proj_iter,
                                 stop_prec=proj_prec)  # solve linear program
        delta_trans = trans_ora - trans
        constC = np.outer(ws ** 2 @ ps, np.ones(nt)) + np.outer(np.ones(ns), r_wt ** 2 @ pt)
        alpha, _, f_val = solve_linesearch(cost=f, G=trans, deltaG=delta_trans, Mi=Mi, f_val=f_val, armijo=False, C1=ws,
                                           C2=r_wt, reg=1, Gc=trans_ora, constC=constC, M=0, alpha_min=0, alpha_max=1)
        trans_new = trans + alpha * delta_trans
        if iter_num % print_interval == 0:
            # loss = gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)
            # print("loss={}".format(loss))
            if trans_star is not None:
                corr = pearson(trans, trans_star)
                print("corr={}".format(corr))
            yield trans_new, inner(trans_new, ws=ws, wt=wt, bound=bound)
        abs_convergence = abs(old_fval - f_val)
        rel_convergence = abs_convergence / f_val
        # if np.max(np.abs(trans_new - trans)) < stop_prec or abs_convergence < stop_prec or rel_convergence < 1e-3:
        if np.max(np.abs(trans_new - trans)) < stop_prec:
            yield trans_new, inner(trans_new, ws=ws, wt=wt, bound=bound)
            break
        else:
            trans = trans_new
    if iter_num == max_iter - 1:
        warnings.warn("CG has not converged")


def gd(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, n_stage=10000, bound=0.2, eta_trans=1,
       trans_star=None, stop_prec=2e-4, use_gd=False, max_proj_iter=1000, proj_prec=2e-4, eta_proj=1e-2):
    # print("eta_trans={}".format(eta_trans))
    trans_max = min(np.max(ps), np.max(pt))
    trans = np.outer(ps, pt)
    for stage_i in range(n_stage):
        pert = inner(trans=trans, ws=ws, wt=wt, bound=bound)
        # print("pert {}, {}, {}".format(np.min(pert), np.median(pert), np.max(pert)))
        # print("pert={}".format(pert))
        r_wt = wt + pert
        grad = peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)
        # print("grad {}".format(grad))
        if use_gd:
            tmp = np.clip(trans - eta_trans * grad, 0, trans_max) + add_prec
            if np.max(tmp) == add_prec:
                warnings.warn("The stepsize for Gradient is too large")
            # trans_new = dual_ascent(tmp, ps, pt, eta=eta_proj, max_iter=max_proj_iter, stop_prec=proj_prec)
            trans_new = dual_ascent_bcd(tmp, ps, pt, eta=eta_proj, max_iter=max_proj_iter, stop_prec=proj_prec)
        else:
            tmp = trans * np.exp(-1 - eta_trans * grad) + add_prec
            trans_new = scaling(tmp, ps, pt, max_iter=max_proj_iter, stop_prec=proj_prec)
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
