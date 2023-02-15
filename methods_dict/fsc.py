import ot
from ot.utils import unif
from sklearn.cluster import SpectralClustering
import numpy as np


def sc(dataset: list, Ys: list, alpha: float, n_clu=2, seed=1):
    n_graph = len(dataset)
    M = np.zeros([n_graph, n_graph])
    for i in range(n_graph):
        for j in range(i + 1, n_graph):
            wi, wj = dataset[i], dataset[j]
            ni, nj = len(wi), len(wj)
            Yi, Yj = Ys[i], Ys[j]
            d = Yi.shape[1]
            grad_w = (Yi ** 2) @ np.ones([d, nj]) + np.ones([ni, d]) @ (Yj.T ** 2) - 2 * Yi @ Yj.T
            pi, pj = unif(ni), unif(nj)
            M[i, j] = ot.gromov.fused_gromov_wasserstein2(grad_w, wi, wj, pi, pj, alpha=alpha)
            M[j, i] = M[i, j]
    print("GW matrix constructed")
    clustering = SpectralClustering(n_clusters=n_clu, assign_labels='discretize', affinity="precomputed",
                                    random_state=seed).fit(M)
    return clustering.labels_
