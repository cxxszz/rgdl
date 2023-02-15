import numpy as np
from copy import deepcopy as dc
from ot.utils import check_random_state, unif
from ot.gromov import fused_gromov_wasserstein, _cg_fused_gromov_wasserstein_unmixing, _adam_stochastic_updates, \
    _initialize_adam_optimizer


def fused_gromov_wasserstein_dictionary_learning(Cs, Ys, D, nt, alpha, reg=0., ps=None, q=None, epochs=20,
                                                 batch_size=16, learning_rate_C=1., learning_rate_Y=1.,
                                                 Cdict_init=None, Ydict_init=None, projection='nonnegative_symmetric',
                                                 use_log=False,
                                                 tol_outer=10 ** (-5), tol_inner=10 ** (-5), max_iter_outer=20,
                                                 max_iter_inner=200, use_adam_optimizer=True, verbose=True, **kwargs):
    r"""
    Infer Fused Gromov-Wasserstein linear dictionary :math:`\{ (\mathbf{C_{dict}[d]}, \mathbf{Y_{dict}[d]}, \mathbf{q}) \}_{d \in [D]}`  from the list of S attributed structures :math:`\{ (\mathbf{C_s}, \mathbf{Y_s},\mathbf{p_s}) \}_s`

    .. math::
        \min_{\mathbf{C_{dict}},\mathbf{Y_{dict}}, \{\mathbf{w_s}\}_{s}} \sum_{s=1}^S  FGW_{2,\alpha}(\mathbf{C_s}, \mathbf{Y_s}, \sum_{d=1}^D w_{s,d}\mathbf{C_{dict}[d]},\sum_{d=1}^D w_{s,d}\mathbf{Y_{dict}[d]}, \mathbf{p_s}, \mathbf{q}) \\ - reg\| \mathbf{w_s}  \|_2^2


    Such that :math:`\forall s \leq S` :

    - :math:`\mathbf{w_s}^\top \mathbf{1}_D = 1`
    - :math:`\mathbf{w_s} \geq \mathbf{0}_D`

    Where :

    - :math:`\forall s \leq S, \mathbf{C_s}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\forall s \leq S, \mathbf{Y_s}` is a (ns,d) features matrix of variable size ns and fixed dimension d.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\mathbf{Y_{dict}}` is a (D, nt, d) tensor of D features matrix of fixed size nt and fixed dimension d.
    - :math:`\forall s \leq S, \mathbf{p_s}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - :math:`\alpha` is the trade-off parameter of Fused Gromov-Wasserstein
    - reg is the regularization coefficient.


    The stochastic algorithm used for estimating the attributed graph dictionary atoms as proposed in [38]

    Parameters
    ----------
    Cs : list of S symmetric array-like, shape (ns, ns)
        List of Metric/Graph cost matrices of variable size (ns,ns).
    Ys : list of S array-like, shape (ns, d)
        List of feature matrix of variable size (ns,d) with d fixed.
    D: int
        Number of dictionary atoms to learn
    nt: int
        Number of samples within each dictionary atoms
    alpha : float
        Trade-off parameter of Fused Gromov-Wasserstein
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
    learning_rate_C: float, optional
        Learning rate used for the stochastic gradient descent on Cdict. Default is 1.
    learning_rate_Y: float, optional
        Learning rate used for the stochastic gradient descent on Ydict. Default is 1.
    Cdict_init: list of D array-like with shape (nt, nt), optional
        Used to initialize the dictionary structures Cdict.
        If set to None (Default), the dictionary will be initialized randomly.
        Else Cdict must have shape (D, nt, nt) i.e match provided shape features.
    Ydict_init: list of D array-like with shape (nt, d), optional
        Used to initialize the dictionary features Ydict.
        If set to None, the dictionary features will be initialized randomly.
        Else Ydict must have shape (D, nt, d) where d is the features dimension of inputs Ys and also match provided shape features.
    projection: str, optional
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
    Ydict_best_state : D array-like, shape (D,nt,d)
        Feature matrices composing the dictionary.
        The dictionary leading to the best loss over an epoch is saved and returned.
    log: dict
        If use_log is True, contains loss evolutions by batches and epoches.
    References
    -------

    ..[38]  Cédric Vincent-Cuaz, Titouan Vayer, Rémi Flamary, Marco Corneli, Nicolas Courty.
            "Online Graph Dictionary Learning"
            International Conference on Machine Learning (ICML). 2021.
    """

    d = Ys[0].shape[-1]
    dataset_size = len(Cs)

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
    if Ydict_init is None:
        # Initialize randomly features of dictionary atoms based on samples distribution by feature component
        dataset_feature_means = np.stack([F.mean(axis=0) for F in Ys])
        Ydict = np.random.normal(loc=dataset_feature_means.mean(axis=0), scale=dataset_feature_means.std(axis=0),
                                 size=(D, nt, d))
    else:
        Ydict = dc(Ydict_init)

    if 'symmetric' in projection:
        Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
    if 'nonnegative' in projection:
        Cdict[Cdict < 0.] = 0.

    if use_adam_optimizer:
        adam_moments_C = _initialize_adam_optimizer(Cdict)
        adam_moments_Y = _initialize_adam_optimizer(Ydict)

    log = {'loss_batches': [], 'loss_epochs': []}
    const_q = q[:, None] * q[None, :]
    diag_q = np.diag(q)
    Cdict_best_state = Cdict.copy()
    Ydict_best_state = Ydict.copy()
    loss_best_state = np.inf
    if batch_size > dataset_size:
        batch_size = dataset_size
    iter_by_epoch = dataset_size // batch_size + int((dataset_size % batch_size) > 0)

    for epoch in range(epochs):
        cumulated_loss_over_epoch = 0.

        for _ in range(iter_by_epoch):

            # Batch iterations
            batch = np.random.choice(range(dataset_size), size=batch_size, replace=False)
            cumulated_loss_over_batch = 0.
            unmixings = np.zeros((batch_size, D))
            Cs_embedded = np.zeros((batch_size, nt, nt))
            Ys_embedded = np.zeros((batch_size, nt, d))
            Ts = [None] * batch_size

            for batch_idx, C_idx in enumerate(batch):
                # BCD solver for Gromov-Wassersteisn linear unmixing used independently on each structure of the sampled batch
                unmixings[batch_idx], Cs_embedded[batch_idx], Ys_embedded[batch_idx], Ts[
                    batch_idx], current_loss = fused_gromov_wasserstein_linear_unmixing(
                    Cs[C_idx], Ys[C_idx], Cdict, Ydict, alpha, reg=reg, p=ps[C_idx], q=q,
                    tol_outer=tol_outer, tol_inner=tol_inner, max_iter_outer=max_iter_outer,
                    max_iter_inner=max_iter_inner
                )
                cumulated_loss_over_batch += current_loss
            cumulated_loss_over_epoch += cumulated_loss_over_batch
            if use_log:
                log['loss_batches'].append(cumulated_loss_over_batch)

            # Stochastic projected gradient step over dictionary atoms
            grad_Cdict = np.zeros_like(Cdict)
            grad_Ydict = np.zeros_like(Ydict)

            for batch_idx, C_idx in enumerate(batch):
                shared_term_structures = Cs_embedded[batch_idx] * const_q - (Cs[C_idx].dot(Ts[batch_idx])).T.dot(
                    Ts[batch_idx])
                shared_term_features = diag_q.dot(Ys_embedded[batch_idx]) - Ts[batch_idx].T.dot(Ys[C_idx])
                grad_Cdict += alpha * unmixings[batch_idx][:, None, None] * shared_term_structures[None, :, :]
                grad_Ydict += (1 - alpha) * unmixings[batch_idx][:, None, None] * shared_term_features[None, :, :]
            grad_Cdict *= 2 / batch_size
            grad_Ydict *= 2 / batch_size

            if use_adam_optimizer:
                Cdict, adam_moments_C = _adam_stochastic_updates(Cdict, grad_Cdict, learning_rate_C, adam_moments_C)
                Ydict, adam_moments_Y = _adam_stochastic_updates(Ydict, grad_Ydict, learning_rate_Y, adam_moments_Y)
            else:
                Cdict -= learning_rate_C * grad_Cdict
                Ydict -= learning_rate_Y * grad_Ydict

            if 'symmetric' in projection:
                Cdict = 0.5 * (Cdict + Cdict.transpose((0, 2, 1)))
            if 'nonnegative' in projection:
                Cdict[Cdict < 0.] = 0.

        if use_log:
            log['loss_epochs'].append(cumulated_loss_over_epoch)
        if loss_best_state > cumulated_loss_over_epoch:
            loss_best_state = cumulated_loss_over_epoch
            Cdict_best_state = Cdict.copy()
            Ydict_best_state = Ydict.copy()
        if verbose:
            print('--- epoch: ', epoch, ' cumulated reconstruction error: ', cumulated_loss_over_epoch)
        yield Cdict_best_state, Ydict_best_state


def fused_gromov_wasserstein_linear_unmixing(C, Y, Cdict, Ydict, alpha, reg=0., p=None, q=None, tol_outer=10 ** (-5),
                                             tol_inner=10 ** (-5), max_iter_outer=20, max_iter_inner=200, **kwargs):
    r"""
    Returns the Fused Gromov-Wasserstein linear unmixing of :math:`(\mathbf{C},\mathbf{Y},\mathbf{p})` onto the attributed dictionary atoms :math:`\{ (\mathbf{C_{dict}[d]},\mathbf{Y_{dict}[d]}, \mathbf{q}) \}_{d \in [D]}`

    .. math::
        \min_{\mathbf{w}}  FGW_{2,\alpha}(\mathbf{C},\mathbf{Y}, \sum_{d=1}^D w_d\mathbf{C_{dict}[d]},\sum_{d=1}^D w_d\mathbf{Y_{dict}[d]}, \mathbf{p}, \mathbf{q}) - reg \| \mathbf{w}  \|_2^2

    such that, :math:`\forall s \leq S` :

        - :math:`\mathbf{w_s}^\top \mathbf{1}_D = 1`
        - :math:`\mathbf{w_s} \geq \mathbf{0}_D`

    Where :

    - :math:`\mathbf{C}` is a (ns,ns) pairwise similarity matrix of variable size ns.
    - :math:`\mathbf{Y}` is a (ns,d) features matrix of variable size ns and fixed dimension d.
    - :math:`\mathbf{C_{dict}}` is a (D, nt, nt) tensor of D pairwise similarity matrix of fixed size nt.
    - :math:`\mathbf{Y_{dict}}` is a (D, nt, d) tensor of D features matrix of fixed size nt and fixed dimension d.
    - :math:`\mathbf{p}` is the source distribution corresponding to :math:`\mathbf{C_s}`
    - :math:`\mathbf{q}` is the target distribution assigned to every structures in the embedding space.
    - :math:`\alpha` is the trade-off parameter of Fused Gromov-Wasserstein
    - reg is the regularization coefficient.

    The algorithm used for solving the problem is a Block Coordinate Descent as discussed in [38], algorithm 6.

    Parameters
    ----------
    C : array-like, shape (ns, ns)
        Metric/Graph cost matrix.
    Y : array-like, shape (ns, d)
        Feature matrix.
    Cdict : D array-like, shape (D,nt,nt)
        Metric/Graph cost matrices composing the dictionary on which to embed (C,Y).
    Ydict : D array-like, shape (D,nt,d)
        Feature matrices composing the dictionary on which to embed (C,Y).
    alpha: float,
        Trade-off parameter of Fused Gromov-Wasserstein.
    reg : float, optional
        Coefficient of the negative quadratic regularization used to promote sparsity of w. The default is 0.
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
        fused gromov-wasserstein linear unmixing of (C,Y,p) onto the span of the dictionary.
    Cembedded: array-like, shape (nt,nt)
        embedded structure of :math:`(\mathbf{C},\mathbf{Y}, \mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{C_{dict}[d]}`.
    Yembedded: array-like, shape (nt,d)
        embedded features of :math:`(\mathbf{C},\mathbf{Y}, \mathbf{p})` onto the dictionary, :math:`\sum_d w_d\mathbf{Y_{dict}[d]}`.
    T: array-like (ns,nt)
        Fused Gromov-Wasserstein transport plan between :math:`(\mathbf{C},\mathbf{p})` and :math:`(\sum_d w_d\mathbf{C_{dict}[d]}, \sum_d w_d\mathbf{Y_{dict}[d]},\mathbf{q})`.
    current_loss: float
        reconstruction error
    References
    -------

    ..[38]  Cédric Vincent-Cuaz, Titouan Vayer, Rémi Flamary, Marco Corneli, Nicolas Courty.
            "Online Graph Dictionary Learning"
            International Conference on Machine Learning (ICML). 2021.
    """

    if p is None:
        p = unif(C.shape[0])
    if q is None:
        q = unif(Cdict.shape[-1])

    T = p[:, None] * q[None, :]
    D = len(Cdict)
    d = Y.shape[-1]
    w = unif(D)  # Initialize with uniform weights
    ns = C.shape[-1]
    nt = Cdict.shape[-1]

    # modeling (C,Y)
    Cembedded = np.sum(w[:, None, None] * Cdict, axis=0)
    Yembedded = np.sum(w[:, None, None] * Ydict, axis=0)

    # constants depending on q
    const_q = q[:, None] * q[None, :]
    diag_q = np.diag(q)
    # Trackers for BCD convergence
    convergence_criterion = np.inf
    current_loss = 10 ** 15
    outer_count = 0
    Ys_constM = (Y ** 2).dot(np.ones((d, nt)))  # constant in computing euclidean pairwise feature matrix

    while (convergence_criterion > tol_outer) and (outer_count < max_iter_outer):
        previous_loss = current_loss

        # 1. Solve GW transport between (C,p) and (\sum_d Cdictionary[d],q) fixing the unmixing w
        Yt_varM = (np.ones((ns, d))).dot((Yembedded ** 2).T)
        M = Ys_constM + Yt_varM - 2 * Y.dot(Yembedded.T)  # euclidean distance matrix between features
        T, log = fused_gromov_wasserstein(M, C, Cembedded, p, q, loss_fun='square_loss', alpha=alpha, armijo=False,
                                          G0=T, log=True)
        current_loss = log['fgw_dist']
        if reg != 0:
            current_loss -= reg * np.sum(w ** 2)

        # 2. Solve linear unmixing problem over w with a fixed transport plan T
        w, Cembedded, Yembedded, current_loss = _cg_fused_gromov_wasserstein_unmixing(C, Y, Cdict, Ydict, Cembedded,
                                                                                      Yembedded, w,
                                                                                      T, p, q, const_q, diag_q,
                                                                                      current_loss, alpha, reg,
                                                                                      tol=tol_inner,
                                                                                      max_iter=max_iter_inner, **kwargs)
        if previous_loss != 0:
            convergence_criterion = abs(previous_loss - current_loss) / abs(previous_loss)
        else:
            convergence_criterion = abs(previous_loss - current_loss) / 10 ** (-12)
        outer_count += 1

    return w, Cembedded, Yembedded, T, current_loss
