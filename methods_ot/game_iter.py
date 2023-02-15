import numpy as np
from copy import deepcopy as dc
from lib.sinkhorn import peyre_expon, normalize, gromov_loss, gromov
from lib.util import pearson

add_prec = 1e-30
print_interval = 10


def rgw(ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, n_stage=100, bound=0.2, eta_trans=1,
        eta_pert=1e-2, trans_star=None, stop_prec=5e-7, max_iter=100000):
    def first_order_inner(trans, descending=True, init=None):
        def inner_grad(pert, trans):  # will always be symmetric
            return (pert + wt) * (trans.T @ trans) - trans.T @ ws @ trans

        nt = len(wt)
        if init is None:
            pert = np.zeros([nt, nt])
        else:
            pert = init
        for t in range(max_iter):
            if descending:
                pert_new = np.clip(pert - eta_pert * inner_grad(pert=pert, trans=trans), -bound, bound)
            else:
                pert_new = np.clip(pert + eta_pert * inner_grad(pert=pert, trans=trans), -bound, bound)
            if np.mean(np.abs(pert_new - pert)) < stop_prec:
                break
            else:
                pert = pert_new
        if t == max_iter - 1:
            print("has not converged")
        # print("pert_new={}".format(pert_new))
        return pert_new

    print("eta_trans={}".format(eta_trans))
    trans = gromov(ws=ws, wt=wt, ps=ps, pt=pt, stepsize=eta_trans, max_iter=max_iter // 100)
    if trans_star is not None:
        corr = pearson(trans, trans_star)
        print("corr={}".format(corr))
    loss = gromov_loss(trans=trans, ws=ws, wt=wt, ps=ps, pt=pt)
    print("loss={}".format(loss))
    pert = None
    for stage_i in range(n_stage):
        # r_wt = wt + first_order_inner(trans=trans, descending=True)
        pert = first_order_inner(trans=trans, descending=False, init=pert)
        print("pert {}, {}, {}".format(np.min(pert), np.median(pert), np.max(pert)))
        r_wt = wt + pert
        trans = gromov(ws=ws, wt=r_wt, ps=ps, pt=pt, stepsize=eta_trans, init=trans)
        yield trans
        if trans_star is not None:
            corr = pearson(trans, trans_star)
            print("corr={}".format(corr))
        loss = gromov_loss(trans=trans, ws=ws, wt=r_wt, ps=ps, pt=pt)
        print("loss={}".format(loss))
    return trans


def robust_grad(trans: np.ndarray, ps: np.ndarray, pt: np.ndarray, ws: np.ndarray, wt: np.array, pert: np.ndarray):
    r_wt = wt + pert
    return peyre_expon(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)


def robust_loss(trans: np.ndarray, ws: np.ndarray, wt: np.ndarray, ps: np.ndarray, pt: np.ndarray, pert: np.ndarray):
    r_wt = wt + pert
    return gromov_loss(trans=trans, ps=ps, pt=pt, ws=ws, wt=r_wt)


def rgw_rand(Ts: np.ndarray, weis: np.ndarray, pert: np.ndarray, eta_trans: float, eta_wei: float, eta_pert: float,
             ent_reg: float, bound: float, gaussian_scale: float, n_iter: int, ps: np.ndarray, pt: np.ndarray,
             ws: np.array, wt: np.array, trans_star=None):
    def pert_grad(pert: np.ndarray, trans: np.ndarray):
        return (pert + wt) * (trans.T @ trans) - trans.T @ ws @ trans

    def pert_grad_mean(pert: np.ndarray, Ts: np.ndarray, weis: np.ndarray):
        n_trans = len(Ts)
        grad = weis[0] * pert_grad(pert=pert, trans=Ts[0])
        for k in range(1, n_trans):
            grad = grad + weis[k] * pert_grad(pert=pert, trans=Ts[k])
        return grad

    def projected_langevin(pert: np.ndarray, Ts: np.ndarray, weis: np.ndarray):
        tmp = pert + eta_pert / (2 * ent_reg) * pert_grad_mean(pert=pert, Ts=Ts, weis=weis)
        tmp = tmp + np.sqrt(eta_pert) * np.random.normal(0, gaussian_scale)
        return np.clip(tmp, -bound, bound)

    assert len(Ts) == len(weis)
    n_trans = len(Ts)
    wei_unnor = np.zeros(n_trans)
    for t in range(n_iter):
        # =========================== updating the weighted mixture ========================
        for k in range(n_trans):  # calculating new weights
            loss = robust_loss(trans=Ts[k], ps=ps, pt=pt, ws=ws, wt=wt, pert=pert)
            wei_unnor[k] = weis[k] * np.exp(-eta_wei * loss)
        weis_new = wei_unnor / np.sum(wei_unnor)
        for k in range(n_trans):  # updating positions
            grad = robust_grad(trans=Ts[k], ps=ps, pt=pt, ws=ws, wt=wt, pert=pert)
            tmp = Ts[k] * np.exp(-1 - eta_trans * grad) + add_prec
            Ts[k] = normalize(tmp, ps, pt)
        weis = dc(weis_new)  # updating weights
        pert = projected_langevin(pert=pert, Ts=Ts, weis=weis)  # updating perturbation
        if t % print_interval == 0:
            print("weis={}".format(weis))
            print("pert {}, {}, {}".format(np.min(pert), np.median(pert), np.max(pert)))
            yield Ts, weis
            if trans_star is not None:
                index = np.random.choice(n_trans, p=weis)
                corr = pearson(Ts[index], trans_star)
                print("corr={}".format(corr))
    return Ts, weis
