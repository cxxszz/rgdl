import numpy as np
from copy import deepcopy as dc
from ot.utils import check_random_state, unif
from ot.gromov import gromov_wasserstein, _adam_stochastic_updates, _initialize_adam_optimizer
from lib.sinkhorn import gromov_loss
from methods_ot.game_fused import gd, cg_alternating as cg_gw


def linesearch(w, grad_w, x, Y, Cdict, Ydict, Cembedded, Yembedded, trans, const_p, const_TCET, one_nt_d, alpha, reg):
    """
    Compute optimal steps for the line search problem
    Parameters
    ----------

    w : array-like, shape (n_atom,)
        Unmixing.
    grad_w : array-like, shape (n_atom, n_atom)
        Gradient of the reconstruction loss with respect to w.
    x: array-like, shape (n_atom,)
        Conditional gradient direction.
    Cdict : array-like, shape (n_atom, atom_size, atom_size)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
        Each matrix in the dictionary must have the same size (atom_size,atom_size).
    Cembedded: array-like, shape (atom_size, atom_size)
        Embedded structure :math:`(\sum_d w[d]*Cdict[d],pt)` of :math:`(\mathbf{C},\mathbf{ps})` onto the dictionary.
        Used to avoid redundant computations.
    const_p: array-like, shape (n,n). Used to avoid redundant computations.
    const_TCET: array-like, shape (atom_size, atom_size). Used to avoid redundant computations.
    Returns
    -------
    gamma: float
        Optimal value for the line-search step
    a: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    b: float
        Constant factor appearing in the factorization :math:`a \gamma^2 + b \gamma +c` of the reconstruction loss
    Cembedded_diff: numpy array, shape (atom_size, atom_size)
        Difference between models evaluated in :math:`\mathbf{w}` and in :math:`\mathbf{w}`.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of :math:`\mathbf{w}`.
    """

    # 3) Line-search step: solve \argmin_{\gamma \in [0,1]} a*gamma^2 + b*gamma + c
    Cembedded_x = np.sum(x[:, None, None] * Cdict, axis=0)
    Cembedded_diff = Cembedded_x - Cembedded
    trace_diffx = np.sum(Cembedded_diff * Cembedded_x * const_p)
    trace_diffw = np.sum(Cembedded_diff * Cembedded * const_p)
    a1 = trace_diffx - trace_diffw
    b1 = 2 * (trace_diffw - np.sum(Cembedded_diff * const_TCET))

    Yembedded_x = np.sum(x[:, None, None] * Ydict, axis=0)
    Yembedded_diff = Yembedded_x - Yembedded
    a2 = np.sum((Yembedded_diff ** 2) @ one_nt_d.T * trans)
    # b2 = 2 * np.sum(T * (ones_ns_d.dot((Yembedded * Yembedded_diff).T) - Y.dot(Yembedded_diff.T)))
    b2 = 2 * np.sum(((Yembedded_diff * Yembedded) @ one_nt_d.T - Yembedded_diff @ Y.T) * trans)
    a, b = alpha * a1 + (1 - alpha) * a2, alpha * b1 + (1 - alpha) * b2
    if reg != 0:
        a -= reg * np.sum((x - w) ** 2)
        b -= 2 * reg * np.sum(w * (x - w))

    if a > 0:
        gamma = min(1, max(0, - b / (2 * a)))
    elif a + b < 0:
        gamma = 1
    else:
        gamma = 0

    return gamma, a, b, Cembedded_diff, Yembedded_diff


def cg(C, Y, Cdict, Ydict, Cembedded, Yembedded, w, trans, ps, pt, const_p, diag_p, pert, starting_loss, alpha, reg=0.,
       tol=10 ** (-5), max_iter=200):
    """
    Parameters
    ----------

    C : array-like, shape (n, n)
        Metric/Graph cost matrix.
    Cdict : array-like, shape (n_atom, atom_size, atom_size)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
        Each matrix in the dictionary must have the same size (atom_size,atom_size).
    Cembedded: array-like, shape (atom_size, atom_size)
        Embedded structure :math:`(\sum_d w[d]*Cdict[d],pt)` of :math:`(\mathbf{C},\mathbf{ps})` onto the dictionary.
        Used to avoid redundant computations.
    w: array-like, shape (n_atom,)
        Linear unmixing of the input structure onto the dictionary
    const_p: array-like, shape (atom_size, atom_size). Used to avoid redundant computations.
    T: array-like, shape (atom_size,n)
        fixed transport plan between the input structure and its representation in the dictionary.
    ps : array-like, shape (atom_size,)
        Distribution in the source space.
    pt : array-like, shape (n,)
        Distribution in the embedding space depicted by the dictionary.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of w. Default is 0.

    Returns
    -------
    w: ndarray (n_atom,)
        optimal unmixing of :math:`(\mathbf{C},\mathbf{ps})` onto the dictionary span given OT starting from previously optimal unmixing.
    """
    convergence_criterion = np.inf
    current_loss = starting_loss
    count = 0
    const_TCET = trans @ (C + pert) @ trans.T
    n_atom = len(Cdict)
    while (convergence_criterion > tol) and (count < max_iter):

        previous_loss = current_loss
        # 1) Compute gradient at current point w
        grad_w = np.zeros(n_atom)
        for i in range(n_atom):
            tmp1 = 2 * np.sum(Cdict[i] * Cembedded * const_p) - 2 * np.sum(Cdict[i] * const_TCET)
            tmp2 = 2 * np.sum(Yembedded @ Ydict[i].T * diag_p) - 2 * np.sum(Ydict[i] @ Y.T * trans)
            grad_w[i] = alpha * tmp1 + (1 - alpha) * tmp2
        grad_w -= 2 * reg * w

        # 2) Conditional gradient direction finding: x= \argmin_x x^T.grad_w
        min_ = np.min(grad_w)
        x = (grad_w == min_).astype(np.float64)
        x /= np.sum(x)

        # 3) Line-search step: solve \argmin_{\gamma \in [0,1]} a*gamma^2 + b*gamma + c
        one_nt_d = np.ones(Y.shape)
        gamma, a, b, Cembedded_diff, Yembedded_diff = linesearch(w, grad_w, x, Y, Cdict, Ydict, Cembedded, Yembedded,
                                                                 trans, const_p, const_TCET, one_nt_d, alpha, reg)

        # 4) Updates: w <-- (1-gamma)*w + gamma*x
        w += gamma * (x - w)
        Cembedded += gamma * Cembedded_diff
        Yembedded += gamma * Yembedded_diff
        current_loss += a * (gamma ** 2) + b * gamma

        if previous_loss != 0:  # not that the loss can be negative if reg >0
            convergence_criterion = abs(previous_loss - current_loss) / abs(previous_loss)
        else:  # handle numerical issues around 0
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-15)
        count += 1

    return w, Cembedded, Yembedded, current_loss


def unmixing(C, Y, Cdict, Ydict, alpha, reg=0., ps=None, pt=None, tol_outer=10 ** (-5), tol_inner=10 ** (-5),
             max_iter_outer=20, max_iter_inner=200, eta_trans=100, bound=1e-3, use_gd=False, use_cg=False):
    """
    C : array-like, shape (atom_size, atom_size)
        Metric/Graph cost matrix.
    Cdict : n_atom array-like, shape (n_atom, atom_size, atom_size)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of w. Default is 0.
    ps : array-like, shape (atom_size,), optional
        Distribution in the space depicted by the dictionary. Default is None and corresponds to uniform distribution.
    pt : array-like, shape (n,), optional
        Distribution in the source space C. Default is None and corresponds to uniform distribution.
    tol_outer : float, optional
        Solver precision for the BCD algorithm.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.

    Returns
    -------
    w: array-like, shape (n_atom,)
        gromov-wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{ps})` onto the span of the dictionary.
    Cembedded: array-like, shape (atom_size, atom_size)
        embedded structure of :math:`(\mathbf{C},\mathbf{ps})` onto the dictionary, :math:`\sum_d w_d\mathbf{C_{dict}[d]}`.
    T: array-like (atom_size, n)
        Gromov-Wasserstein transport plan between :math:`(\mathbf{C},\mathbf{ps})` and :math:`(\sum_d w_d\mathbf{C_{dict}[d]}, \mathbf{pt})`
    current_loss: float
        reconstruction error
    """
    if ps is None:
        ps = unif(Cdict.shape[-1])
    if pt is None:
        pt = unif(C.shape[0])
    trans = ps[:, None] * pt[None, :]  # np.outer(ps, pt)
    n_atom = len(Cdict)
    d = Y.shape[1]
    w = unif(n_atom)  # Initialize uniformly the unmixing w
    ns, nt = len(ps), Y.shape[0]
    Cembedded = np.sum(w[:, None, None] * Cdict, axis=0)
    Yembedded = np.sum(w[:, None, None] * Ydict, axis=0)
    const_p = ps[:, None] * ps[None, :]
    diag_p = np.diag(ps)
    convergence_criterion = np.inf  # Trackers for BCD convergence
    current_loss = 10 ** 15
    outer_count = 0

    const_grad_w = np.ones([ns, d]) @ (Y.T ** 2)

    while (convergence_criterion > tol_outer) and (outer_count < max_iter_outer):
        grad_w = (Yembedded ** 2) @ np.ones([d, nt]) + const_grad_w - 2 * Yembedded @ Y.T
        previous_loss = current_loss
        if use_cg:
            for trans, pert in cg_gw(M=grad_w, ws=Cembedded, wt=C, ps=ps, pt=pt, alpha=alpha, bound=bound,
                                     stop_prec=tol_inner):
                pass
        else:
            for trans, pert in gd(ws=Cembedded, wt=C, Ys=Yembedded, Yt=Y, alpha=alpha, ps=ps, pt=pt,
                                  eta_trans=eta_trans, bound=bound, use_gd=use_gd, stop_prec=tol_inner):
                pass
        current_loss = gromov_loss(trans=trans, ws=Cembedded, wt=C + pert, ps=ps, pt=pt)
        # input("ns {}, nt {}, Yembdded {}, const_grad_w {}, Y {}".format(ns, nt, Yembedded.shape, const_grad_w.shape,
        #                                                                 Y.shape))

        current_loss = alpha * current_loss + (1 - alpha) * np.sum(grad_w * trans)
        if reg != 0:
            current_loss -= reg * np.sum(w ** 2)
        # 2. Solve linear unmixing problem over w with a fixed transport plan T
        w, Cembedded, Yembedded, current_loss = cg(
            C=C, Y=Y, Cdict=Cdict, Ydict=Ydict, Cembedded=Cembedded, Yembedded=Yembedded, w=w, trans=trans, ps=ps,
            pt=pt, const_p=const_p, diag_p=diag_p, pert=pert, starting_loss=current_loss, alpha=alpha, reg=reg,
            tol=tol_inner, max_iter=max_iter_inner
        )

        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(previous_loss)
        else:  # handle numerical issues around 0
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-15)
        if use_cg:
            convergence_criterion /= 1e3
        outer_count += 1

    return w, Cembedded, Yembedded, trans, pert, current_loss


def dictionary_learning(C_list, Y_list, n_atom, atom_size, alpha, reg=0., ps=None, pt=None, epochs=20, batch_size=16,
                        learning_rate=1., Cdict_init=None, Ydict_init=None, projection='nonnegative_symmetric',
                        use_log=True, tol_outer=10 ** (-5), tol_inner=10 ** (-5), max_iter_outer=20, max_iter_inner=200,
                        use_adam_optimizer=True, verbose=True, eta_trans=100, bound=1e-3, use_gd=False, use_cg=False):
    """

    Parameters
    ----------
    Cs : list of K symmetric array-like, shape (n, n)
    n_atom: int
        Number of dictionary atoms to learn
    atom_size: int
        Number of samples within each dictionary atoms
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
    ps : list of M array-like, shape (atom_size,), optional
        Distribution in the embedding space whose structure will be learned. Default is None and corresponds to uniform distributions.
        Distribution in each source space C of Cs. Default is None and corresponds to uniform distibutions.
    pt : array-like, shape (n,), optional
        Distribution in the embedding space whose structure will be learned. Default is None and corresponds to uniform distributions.
    epochs: int, optional
        Number of epochs used to learn the dictionary. Default is 32.
    batch_size: int, optional
        Batch size for each stochastic gradient update of the dictionary. Set to the dataset size if the provided batch_size is higher than the dataset size. Default is 32.
    learning_rate: float, optional
        Learning rate used for the stochastic gradient descent. Default is 1.
    Cdict_init: list of n_atom array-like with shape (atom_size, atom_size), optional
        Used to initialize the dictionary.
        If set to None (Default), the dictionary will be initialized randomly.
        Else Cdict must have shape (n_atom, atom_size, atom_size) i.e match provided shape features.
    projection: str , optional
        If 'nonnegative' and/or 'symmetric' is in projection, the corresponding projection will be performed at each stochastic update of the dictionary
        Else the set of atoms is :math:`R^{atom_size * atom_size}`. Default is 'nonnegative_symmetric'
    log: bool, optional
        If set to True, losses evolution by batches and epochs are tracked. Default is False.
    use_adam_optimizer: bool, optional
        If set to True, adam optimizer with default settings is used as adaptative learning rate strategy.
        Else perform SGD with fixed learning rate. Default is True.
    tol_outer : float, optional
        Solver precision for the BCD algorithm, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    tol_inner : float, optional
        Solver precision for the Conjugate Gradient algorithm used to get optimal w at a fixed transport, measured by absolute relative error on consecutive losses. Default is :math:`10^{-5}`.
    max_iter_outer : int, optional
        Maximum number of iterations for the BCD. Default is 20.
    max_iter_inner : int, optional
        Maximum number of iterations for the Conjugate Gradient. Default is 200.
    verbose : bool, optional
        Print the reconstruction loss every epoch. Default is False.

    Returns
    -------

    Cdict_best_state : n_atom array-like, shape (n_atom, atom_size, atom_size)
        Metric/Graph cost matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    """
    dataset_size = len(C_list)
    d = Y_list[0].shape[1]
    # Handle backend of optional arguments
    if ps is None:
        ps = unif(atom_size)
    if pt is None:
        pt = [unif(C.shape[0]) for C in C_list]
    if Cdict_init is None:
        # Initialize randomly structures of dictionary atoms based on samples
        dataset_means = [C.mean() for C in C_list]
        Cdict = np.random.normal(loc=np.mean(dataset_means), scale=np.std(dataset_means),
                                 size=(n_atom, atom_size, atom_size))
    else:
        Cdict = dc(Cdict_init)
    if Ydict_init is None:
        # Initialize randomly features of dictionary atoms based on samples distribution by feature component
        dataset_feature_means = np.stack([F.mean(axis=0) for F in Y_list])
        Ydict = np.random.normal(loc=dataset_feature_means.mean(axis=0), scale=dataset_feature_means.std(axis=0),
                                 size=(n_atom, atom_size, d))
    else:
        Ydict = dc(Ydict_init)
    if 'symmetric' in projection:
        Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
    if 'nonnegative' in projection:
        Cdict[Cdict < 0.] = 0
    if use_adam_optimizer:
        adam_moments_C = _initialize_adam_optimizer(Cdict)
        adam_moments_Y = _initialize_adam_optimizer(Ydict)

    log = {'loss_batches': [], 'loss_epochs': []}
    const_p = ps[:, None] * ps[None, :]
    diag_p = np.diag(ps)
    Cdict_best_state = Cdict.copy()
    Ydict_best_state = Ydict.copy()
    loss_best_state = np.inf
    if batch_size > dataset_size:
        batch_size = dataset_size
    iter_by_epoch = dataset_size // batch_size + int((dataset_size % batch_size) > 0)

    for epoch in range(epochs):
        cumulated_loss_over_epoch = 0.

        for _ in range(iter_by_epoch):
            # batch sampling
            batch = np.random.choice(range(dataset_size), size=batch_size, replace=False)
            cumulated_loss_over_batch = 0.
            unmixings = np.zeros((batch_size, n_atom))
            Cs_embedded = np.zeros((batch_size, atom_size, atom_size))
            Ys_embedded = np.zeros((batch_size, atom_size, d))
            trans_list = [None] * batch_size
            pert_list = [None] * batch_size

            for batch_idx, C_idx in enumerate(batch):
                # BCD solver for Gromov-Wassersteisn linear unmixing used independently on each structure of the sampled batch
                unmixings[batch_idx], Cs_embedded[batch_idx], Ys_embedded[batch_idx], trans_list[batch_idx], pert_list[
                    batch_idx], current_loss = unmixing(
                    C_list[C_idx], Y_list[C_idx], Cdict, Ydict, alpha, reg=reg, ps=ps, pt=pt[C_idx],
                    tol_outer=tol_outer, tol_inner=tol_inner, max_iter_outer=max_iter_outer,
                    max_iter_inner=max_iter_inner, eta_trans=eta_trans, bound=bound, use_gd=use_gd, use_cg=use_cg
                )
                cumulated_loss_over_batch += current_loss
            cumulated_loss_over_epoch += cumulated_loss_over_batch

            if use_log:
                log['loss_batches'].append(cumulated_loss_over_batch)

            # Stochastic projected gradient step over dictionary atoms
            grad_Cdict = np.zeros_like(Cdict)
            grad_Ydict = np.zeros_like(Ydict)
            for batch_idx, C_idx in enumerate(batch):
                shared_term_structures = Cs_embedded[batch_idx] * const_p - trans_list[batch_idx] @ (
                        C_list[C_idx] + pert_list[batch_idx]) @ trans_list[batch_idx].T
                shared_term_features = diag_p @ Ys_embedded[batch_idx] - trans_list[batch_idx] @ Y_list[C_idx]
                grad_Cdict += alpha * unmixings[batch_idx][:, None, None] * shared_term_structures[None, :, :]
                grad_Ydict += (1 - alpha) * unmixings[batch_idx][:, None, None] * shared_term_features[None, :, :]
            grad_Cdict *= 2 / batch_size
            grad_Ydict *= 2 / batch_size
            if use_adam_optimizer:
                Cdict, adam_moments_C = _adam_stochastic_updates(Cdict, grad_Cdict, learning_rate, adam_moments_C)
                Ydict, adam_moments_Y = _adam_stochastic_updates(Ydict, grad_Ydict, learning_rate, adam_moments_Y)
            else:
                Cdict -= learning_rate * grad_Cdict
                Ydict -= learning_rate * grad_Ydict
            if 'symmetric' in projection:
                Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
            if 'nonnegative' in projection:
                Cdict[Cdict < 0.] = 0.

        if use_log:
            log['loss_epochs'].append(cumulated_loss_over_epoch)
        if np.abs(loss_best_state - cumulated_loss_over_epoch) < tol_outer:
            flag = True
        else:
            flag = False
        if loss_best_state > cumulated_loss_over_epoch:
            loss_best_state = cumulated_loss_over_epoch
            Cdict_best_state = Cdict.copy()
            Ydict_best_state = Ydict.copy()
        if verbose:
            print('--- epoch =', epoch, ' cumulated reconstruction error: ', cumulated_loss_over_epoch)
        yield Cdict_best_state, Ydict_best_state
        if flag:
            break
