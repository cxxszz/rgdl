import numpy as np
from copy import deepcopy as dc
from ot.utils import check_random_state, unif
from ot.gromov import gromov_wasserstein, _cg_gromov_wasserstein_unmixing, _adam_stochastic_updates, \
    _initialize_adam_optimizer


def gromov_wasserstein_linear_unmixing(C, Cdict, reg=0., p=None, q=None, tol_outer=10 ** (-5), tol_inner=10 ** (-5),
                                       max_iter_outer=20, max_iter_inner=200, **kwargs):
    """
    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Cdict : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed C.
    reg : float, optional.
        Coefficient of the negative quadratic regularization used to promote sparsity of w. Default is 0.
    p : array-like, shape (ns,), optional
        Distribution in the source space C. Default is None and corresponds to uniform distribution.
    q : array-like, shape (nt,), optional
        Distribution in the space depicted by the dictionary. Default is None and corresponds to uniform distribution.
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
    w: array-like, shape (D,)
        gromov-wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{p})` onto the span of the dictionary.
    Cembedded: array-like, shape (nt,nt)
        embedded structure of :math:`(\mathbf{C},\mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{C_{dict}[d]}`.
    T: array-like (ns, nt)
        Gromov-Wasserstein transport plan between :math:`(\mathbf{C},\mathbf{p})` and :math:`(\sum_d w_d\mathbf{C_{dict}[d]}, \mathbf{q})`
    current_loss: float
        reconstruction error
    """
    C0, Cdict0 = C, Cdict
    C = dc(C0)
    Cdict = dc(Cdict0)
    if p is None:
        p = unif(C.shape[0])

    if q is None:
        q = unif(Cdict.shape[-1])

    T = p[:, None] * q[None, :]  # np.outer(p, q)
    D = len(Cdict)

    w = unif(D)  # Initialize uniformly the unmixing w
    Cembedded = np.sum(w[:, None, None] * Cdict, axis=0)

    const_q = q[:, None] * q[None, :]
    # Trackers for BCD convergence
    convergence_criterion = np.inf
    current_loss = 10 ** 15
    outer_count = 0

    while (convergence_criterion > tol_outer) and (outer_count < max_iter_outer):
        previous_loss = current_loss
        # 1. Solve GW transport between (C,p) and (\sum_d Cdictionary[d],q) fixing the unmixing w
        T, log = gromov_wasserstein(C1=C, C2=Cembedded, p=p, q=q, loss_fun='square_loss', G0=T, log=True, armijo=False,
                                    **kwargs)
        current_loss = log['gw_dist']
        if reg != 0:
            current_loss -= reg * np.sum(w ** 2)

        # 2. Solve linear unmixing problem over w with a fixed transport plan T
        w, Cembedded, current_loss = _cg_gromov_wasserstein_unmixing(
            C=C, Cdict=Cdict, Cembedded=Cembedded, w=w, const_q=const_q, T=T,
            starting_loss=current_loss, reg=reg, tol=tol_inner, max_iter=max_iter_inner, **kwargs
        )

        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(previous_loss)
        else:  # handle numerical issues around 0
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-15)
        outer_count += 1

    return w, Cembedded, T, current_loss


def gromov_wasserstein_dictionary_learning(Cs, D, nt, reg=0., ps=None, q=None, epochs=20, batch_size=32,
                                           learning_rate=1., Cdict_init=None, projection='nonnegative_symmetric',
                                           use_log=True, tol_outer=10 ** (-5), tol_inner=10 ** (-5), max_iter_outer=20,
                                           max_iter_inner=200, use_adam_optimizer=True, verbose=False, **kwargs):
    """

    Parameters
    ----------
    Cs : list of S symmetric array-like, shape (ns, ns)
        List of Metric/Graph cost matrices of variable size (ns, ns).
    D: int
        Number of dictionary atoms to learn
    nt: int
        Number of samples within each dictionary atoms
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
    ps : list of S array-like, shape (ns,), optional
        Distribution in each source space C of Cs. Default is None and corresponds to uniform distibutions.
    q : array-like, shape (nt,), optional
        Distribution in the embedding space whose structure will be learned. Default is None and corresponds to uniform distributions.
    epochs: int, optional
        Number of epochs used to learn the dictionary. Default is 32.
    batch_size: int, optional
        Batch size for each stochastic gradient update of the dictionary. Set to the dataset size if the provided batch_size is higher than the dataset size. Default is 32.
    learning_rate: float, optional
        Learning rate used for the stochastic gradient descent. Default is 1.
    Cdict_init: list of D array-like with shape (nt, nt), optional
        Used to initialize the dictionary.
        If set to None (Default), the dictionary will be initialized randomly.
        Else Cdict must have shape (D, nt, nt) i.e match provided shape features.
    projection: str , optional
        If 'nonnegative' and/or 'symmetric' is in projection, the corresponding projection will be performed at each stochastic update of the dictionary
        Else the set of atoms is :math:`R^{nt * nt}`. Default is 'nonnegative_symmetric'
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

    Cdict_best_state : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    """
    Cs0 = dc(Cs)
    dataset_size = len(Cs)
    # Handle backend of optional arguments
    if ps is None:
        ps = [unif(C.shape[0]) for C in Cs]
    if q is None:
        q = unif(nt)
    if Cdict_init is None:
        # Initialize randomly structures of dictionary atoms based on samples
        dataset_means = [C.mean() for C in Cs]
        Cdict = np.random.normal(loc=np.mean(dataset_means), scale=np.std(dataset_means), size=(D, nt, nt))
    else:
        Cdict = dc(Cdict_init)

    if 'symmetric' in projection:
        Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
    if 'nonnegative' in projection:
        Cdict[Cdict < 0.] = 0
    if use_adam_optimizer:
        adam_moments = _initialize_adam_optimizer(Cdict)

    log = {'loss_batches': [], 'loss_epochs': []}
    const_q = q[:, None] * q[None, :]
    Cdict_best_state = Cdict.copy()
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
            unmixings = np.zeros((batch_size, D))
            Cs_embedded = np.zeros((batch_size, nt, nt))
            Ts = [None] * batch_size

            for batch_idx, C_idx in enumerate(batch):
                # BCD solver for Gromov-Wassersteisn linear unmixing used independently on each structure of the sampled batch
                unmixings[batch_idx], Cs_embedded[batch_idx], Ts[
                    batch_idx], current_loss = gromov_wasserstein_linear_unmixing(
                    Cs[C_idx], Cdict, reg=reg, p=ps[C_idx], q=q, tol_outer=tol_outer, tol_inner=tol_inner,
                    max_iter_outer=max_iter_outer, max_iter_inner=max_iter_inner
                )
                cumulated_loss_over_batch += current_loss
            cumulated_loss_over_epoch += cumulated_loss_over_batch

            if use_log:
                log['loss_batches'].append(cumulated_loss_over_batch)

            # Stochastic projected gradient step over dictionary atoms
            grad_Cdict = np.zeros_like(Cdict)
            for batch_idx, C_idx in enumerate(batch):
                shared_term_structures = Cs_embedded[batch_idx] * const_q - (Cs[C_idx].dot(Ts[batch_idx])).T.dot(
                    Ts[batch_idx])
                grad_Cdict += unmixings[batch_idx][:, None, None] * shared_term_structures[None, :, :]
            grad_Cdict *= 2 / batch_size
            if use_adam_optimizer:
                Cdict, adam_moments = _adam_stochastic_updates(Cdict, grad_Cdict, learning_rate, adam_moments)
            else:
                Cdict -= learning_rate * grad_Cdict
            if 'symmetric' in projection:
                Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
            if 'nonnegative' in projection:
                Cdict[Cdict < 0.] = 0.

        if use_log:
            log['loss_epochs'].append(cumulated_loss_over_epoch)
        if loss_best_state > cumulated_loss_over_epoch:
            loss_best_state = cumulated_loss_over_epoch
            Cdict_best_state = Cdict.copy()
        if verbose:
            print('--- epoch =', epoch, ' cumulated reconstruction error: ', cumulated_loss_over_epoch)
        yield Cdict_best_state
