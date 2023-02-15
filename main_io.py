import numpy as np
from numpy import linalg as la
from copy import deepcopy as dc
import networkx as nx
import json, csv
from networkx.generators.community import stochastic_block_model as sbm
from lib.matmul import diag_matmul_np, matmul_diag_np
from data_io.real_dataload import build_dataset_withoutfeatures, build_dataset_realfeatures


def random_walk_sim(w: np.ndarray, walk_len=2):
    d = np.sum(w, axis=1)
    rw = diag_matmul_np(1 / d, w)
    B1, B2 = dc(rw), dc(rw.T)
    for i in range(2, walk_len + 1):
        B1 += B1 @ rw
        B2 += B2 @ rw.T
    B = B1 + B2
    # B = B - np.diag(np.diag(B))
    return B


def heat_kernel(A: np.ndarray, diff_time=10):
    d = np.sum(A, axis=1)
    tmps = diag_matmul_np((d + 1e-32) ** -0.5, A)
    n = A.shape[0]
    L = np.eye(n) - matmul_diag_np(tmps, (d + 1e-32) ** -0.5)
    lda, Phi = la.eig(L)
    C = Phi @ np.diag(-diff_time * lda) @ Phi.T
    return C


def sbm_balanced_adj(n_graph=60, clusters=(1, 2, 3)):
    Nc = n_graph // len(clusters)  # number of graphs by cluster
    nlabels = len(clusters)
    dataset = []
    labels = []

    p_inter = 0.1
    p_intra = 0.9
    for n_cluster in clusters:
        for i in range(Nc):
            n_nodes = int(np.random.uniform(low=30, high=50))

            if n_cluster > 1:
                P = p_inter * np.ones((n_cluster, n_cluster))
                np.fill_diagonal(P, p_intra)
            else:
                P = p_intra * np.eye(1)
            sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
            G = sbm(sizes, P, seed=i, directed=False)
            C = nx.to_numpy_array(G)
            dataset.append(C)
            labels.append(n_cluster)
    A1 = np.zeros([6, 6])
    A1[0, 1] = 2
    A1[2, 3] = 2
    A1[4, 5] = 2
    A1 = 0.5 * (A1 + A1.T)

    A2 = np.zeros([6, 6])
    A2[0, 1] = 2
    A2[1, 2] = 2
    A2[2, 0] = 2
    A2[3, 4] = 2
    A2[4, 5] = 2
    A2[5, 3] = 2
    A2 = 0.5 * (A2 + A2.T)

    Cdict = np.zeros([3, 6, 6])
    Cdict[0] = nx.to_numpy_array(nx.complete_graph(6))
    Cdict[1] = A1
    Cdict[2] = A2
    return dataset, labels, Cdict, dataset


def modify_adj(A: np.ndarray, ratio=0.05):
    A_modify = dc(A)
    n = len(A)
    n_edge = np.count_nonzero(A) // 2
    n_modify = int(n_edge * ratio)
    add_count, rm_count = 0, 0
    while add_count < n_modify:
        i, j = np.random.choice(n, 2, replace=False)
        if A_modify[i, j] == 0:
            A_modify[i, j] = 1
            A_modify[j, i] = 1
            add_count += 1
    while rm_count < n_modify:
        candidates = np.nonzero(A_modify)
        edge = np.random.choice(len(candidates[0]))
        i, j = candidates[0][edge], candidates[1][edge]
        A_tmp = dc(A_modify)
        A_tmp[i, j] = 0
        A_tmp[j, i] = 0
        if nx.is_connected(nx.from_numpy_array(A_tmp)):
            A_modify = dc(A_tmp)
            rm_count += 1
    return A_modify


def sbm_balanced_edge(n_graph=60, clusters=(1, 2, 3), scale=0.1):
    Nc = n_graph // len(clusters)  # number of graphs by cluster
    nlabels = len(clusters)
    dataset_clean, dataset = [], []
    labels = []

    p_inter = 0.1
    p_intra = 0.9
    for n_cluster in clusters:
        for i in range(Nc):
            n_nodes = int(np.random.uniform(low=30, high=50))

            if n_cluster > 1:
                P = p_inter * np.ones((n_cluster, n_cluster))
                np.fill_diagonal(P, p_intra)
            else:
                P = p_intra * np.eye(1)
            sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
            G = sbm(sizes, P, seed=i, directed=False)
            A = nx.to_numpy_array(G)
            C = heat_kernel(A)
            dataset_clean.append(C)
            if scale > 0:
                A_noisy = modify_adj(A, scale)
            else:
                A_noisy = A
            C_noisy = heat_kernel(A_noisy)
            dataset.append(C_noisy)
            labels.append(n_cluster)
    A1 = np.zeros([6, 6])
    A1[0, 1] = 2
    A1[2, 3] = 2
    A1[4, 5] = 2
    A1 = 0.5 * (A1 + A1.T)

    A2 = np.zeros([6, 6])
    A2[0, 1] = 2
    A2[1, 2] = 2
    A2[2, 0] = 2
    A2[3, 4] = 2
    A2[4, 5] = 2
    A2[5, 3] = 2
    A2 = 0.5 * (A2 + A2.T)

    Cdict = np.zeros([3, 6, 6])
    Cdict[0] = heat_kernel(nx.to_numpy_array(nx.complete_graph(6)))
    Cdict[1] = heat_kernel(A1)
    Cdict[2] = heat_kernel(A2)
    return dataset, labels, Cdict, dataset_clean


def sbm_unbalanced_edge(n_graph=60, clusters=(1, 2, 3), scale=0.1):
    tmp = [int(n_graph * 0.2), int(n_graph * 0.3)]
    Nc = dc(tmp)
    Nc.append(n_graph - tmp[0] - tmp[1])
    nlabels = len(clusters)
    dataset_clean, dataset = [], []
    labels = []

    p_inter = 0.1
    p_intra = 0.9
    for index, n_cluster in enumerate(clusters):
        for i in range(Nc[index]):
            n_nodes = int(np.random.uniform(low=30, high=50))

            if n_cluster > 1:
                P = p_inter * np.ones((n_cluster, n_cluster))
                np.fill_diagonal(P, p_intra)
            else:
                P = p_intra * np.eye(1)
            sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
            G = sbm(sizes, P, seed=i, directed=False)
            A = nx.to_numpy_array(G)
            C = heat_kernel(A)
            dataset_clean.append(C)
            if scale > 0:
                A_noisy = modify_adj(A, scale)
            else:
                A_noisy = A
            C_noisy = heat_kernel(A_noisy)
            dataset.append(C_noisy)
            labels.append(n_cluster)
    A1 = np.zeros([6, 6])
    A1[0, 1] = 2
    A1[2, 3] = 2
    A1[4, 5] = 2
    A1 = 0.5 * (A1 + A1.T)

    A2 = np.zeros([6, 6])
    A2[0, 1] = 2
    A2[1, 2] = 2
    A2[2, 0] = 2
    A2[3, 4] = 2
    A2[4, 5] = 2
    A2[5, 3] = 2
    A2 = 0.5 * (A2 + A2.T)

    Cdict = np.zeros([3, 6, 6])
    Cdict[0] = heat_kernel(nx.to_numpy_array(nx.complete_graph(6)))
    Cdict[1] = heat_kernel(A1)
    Cdict[2] = heat_kernel(A2)
    return dataset, labels, Cdict, dataset_clean


def sbm_balanced_gaussian(n_graph=60, clusters=(1, 2, 3), scale=0.1):
    Nc = n_graph // len(clusters)  # number of graphs by cluster
    nlabels = len(clusters)
    dataset_clean, dataset = [], []
    labels = []

    p_inter = 0.1
    p_intra = 0.9
    for n_cluster in clusters:
        for i in range(Nc):
            n_nodes = int(np.random.uniform(low=30, high=50))

            if n_cluster > 1:
                P = p_inter * np.ones((n_cluster, n_cluster))
                np.fill_diagonal(P, p_intra)
            else:
                P = p_intra * np.eye(1)
            sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
            G = sbm(sizes, P, seed=i, directed=False)
            A = nx.to_numpy_array(G)
            C = heat_kernel(A)
            dataset_clean.append(C)
            if scale > 0:
                tmp = np.random.normal(0, scale, C.shape)
            else:
                tmp = np.zeros(C.shape)
            C_noisy = C + tmp + tmp.T
            dataset.append(C_noisy)
            labels.append(n_cluster)
    A1 = np.zeros([6, 6])
    A1[0, 1] = 2
    A1[2, 3] = 2
    A1[4, 5] = 2
    A1 = 0.5 * (A1 + A1.T)

    A2 = np.zeros([6, 6])
    A2[0, 1] = 2
    A2[1, 2] = 2
    A2[2, 0] = 2
    A2[3, 4] = 2
    A2[4, 5] = 2
    A2[5, 3] = 2
    A2 = 0.5 * (A2 + A2.T)

    Cdict = np.zeros([3, 6, 6])
    Cdict[0] = heat_kernel(nx.to_numpy_array(nx.complete_graph(6)))
    Cdict[1] = heat_kernel(A1)
    Cdict[2] = heat_kernel(A2)
    return dataset, labels, Cdict, dataset_clean


def sbm_unbalanced_gaussian(n_graph=60, clusters=(1, 2, 3), scale=0.1):
    tmp = [int(n_graph * 0.2), int(n_graph * 0.3)]
    Nc = dc(tmp)
    Nc.append(n_graph - tmp[0] - tmp[1])
    nlabels = len(clusters)
    dataset_clean, dataset = [], []
    labels = []

    p_inter = 0.1
    p_intra = 0.9
    for index, n_cluster in enumerate(clusters):
        for i in range(Nc[index]):
            n_nodes = int(np.random.uniform(low=30, high=50))

            if n_cluster > 1:
                P = p_inter * np.ones((n_cluster, n_cluster))
                np.fill_diagonal(P, p_intra)
            else:
                P = p_intra * np.eye(1)
            sizes = np.round(n_nodes * np.ones(n_cluster) / n_cluster).astype(np.int32)
            G = sbm(sizes, P, seed=i, directed=False)
            A = nx.to_numpy_array(G)
            C = heat_kernel(A)
            dataset_clean.append(C)
            if scale > 0:
                tmp = np.random.normal(0, scale, C.shape)
            else:
                tmp = np.zeros(C.shape)
            C_noisy = C + tmp + tmp.T
            dataset.append(C_noisy)
            labels.append(n_cluster)
    A1 = np.zeros([6, 6])
    A1[0, 1] = 2
    A1[2, 3] = 2
    A1[4, 5] = 2
    A1 = 0.5 * (A1 + A1.T)

    A2 = np.zeros([6, 6])
    A2[0, 1] = 2
    A2[1, 2] = 2
    A2[2, 0] = 2
    A2[3, 4] = 2
    A2[4, 5] = 2
    A2[5, 3] = 2
    A2 = 0.5 * (A2 + A2.T)

    Cdict = np.zeros([3, 6, 6])
    Cdict[0] = heat_kernel(nx.to_numpy_array(nx.complete_graph(6)))
    Cdict[1] = heat_kernel(A1)
    Cdict[2] = heat_kernel(A2)
    return dataset, labels, Cdict, dataset_clean


def imdb_b(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('IMDB-BINARY', "data/IMDB-BINARY/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def imdb_m(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('IMDB-MULTI', "data/IMDB-MULTI/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def mutag(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('MUTAG', "data/MUTAG_2/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def bzr(rtype, diff_time=2):
    X = build_dataset_realfeatures('BZR', "data/BZR/", type_attr='real')
    dataset = []
    attributes = []
    labels = []
    for x in X:
        g = x[0]
        # input(g.nodes())
        A = nx.to_numpy_array(g)  # the ordering is produced by G.nodes().
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(0.5 * (A + A.T))
            dataset.append(C.astype(np.float))
        tmp = nx.get_node_attributes(g, "att")
        attribute = []
        for node in g.nodes():
            attribute.append(tmp[node])
        attribute = np.array(attribute)
        attributes.append(attribute)
        labels.append(x[1])
    return dataset, labels, None, dc(dataset), attributes


def cox(rtype, diff_time=2):
    X = build_dataset_realfeatures('COX2', "data/COX2/", type_attr='real')
    dataset = []
    attributes = []
    labels = []
    for x in X:
        g = x[0]
        # input(g.nodes())
        A = nx.to_numpy_array(g)  # the ordering is produced by G.nodes().
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(0.5 * (A + A.T))
            dataset.append(C.astype(np.float))
        tmp = nx.get_node_attributes(g, "att")
        attribute = []
        for node in g.nodes():
            attribute.append(tmp[node])
        attribute = np.array(attribute)
        attributes.append(attribute)
        labels.append(x[1])
    return dataset, labels, None, dc(dataset), attributes


def enzymes(rtype, diff_time=2):
    X = build_dataset_realfeatures('ENZYMES', "data/ENZYMES_2/", type_attr='real')
    dataset = []
    attributes = []
    labels = []
    for x in X:
        g = x[0]
        # input(g.nodes())
        A = nx.to_numpy_array(g)  # the ordering is produced by G.nodes().
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(0.5 * (A + A.T))
            dataset.append(C.astype(np.float))
        tmp = nx.get_node_attributes(g, "att")
        attribute = []
        for node in g.nodes():
            attribute.append(tmp[node])
        attribute = np.array(attribute)
        attributes.append(attribute)
        labels.append(x[1])
    return dataset, labels, None, dc(dataset), attributes


def msrc(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('MSRC_9', "data/MSRC_9/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def kki(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('KKI', "data/KKI/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            # dataset.append(A)
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def deezer(rtype, diff_time=2):
    adj_path = "data/deezer_ego_nets/deezer_edges.json"
    f = open(adj_path)

    # returns JSON object as a dictionary
    d = json.load(f)
    dataset = []
    for k in d.keys():
        g = nx.Graph()
        g.add_edges_from(d[k])
        A = nx.to_numpy_array(g)
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
    path = "data/deezer_ego_nets/deezer_target.csv"
    with open(path, newline="") as csvfile:
        tmp = list(csv.reader(csvfile))
    # print(tmp)
    labels = []
    i = 0
    while i < len(tmp) - 1:
        # print(i, tmp[i][0])
        assert i == int(tmp[i + 1][0])
        labels.append(int(tmp[i + 1][1]))
        i += 1
    assert len(dataset) == len(labels)
    return dataset, labels, None, dataset


def github(rtype, diff_time=2):
    adj_path = "data/github_stargazers/git_edges.json"
    f = open(adj_path)

    # returns JSON object as a dictionary
    d = json.load(f)
    dataset = []
    for k in d.keys():
        g = nx.Graph()
        g.add_edges_from(d[k])
        A = nx.to_numpy_array(g)
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            dataset.append(A)
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
    path = "data/github_stargazers/git_target.csv"
    with open(path, newline="") as csvfile:
        tmp = list(csv.reader(csvfile))
    # print(tmp)
    labels = []
    i = 0
    while i < len(tmp) - 1:
        # print(i, tmp[i][0])
        assert i == int(tmp[i + 1][0])
        labels.append(int(tmp[i + 1][1]))
        i += 1
    assert len(dataset) == len(labels)
    return dataset, labels, None, dataset


def ohsu(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('OHSU', "data/OHSU/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            # dataset.append(A)
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


def peking(rtype, diff_time=2):
    X = build_dataset_withoutfeatures('Peking_1', "data/Peking_1/")
    dataset = []
    labels = []
    for x in X:
        A = nx.to_numpy_array(x[0])
        A = A.astype(np.float)
        if rtype == "heat":
            C = heat_kernel(0.5 * (A + A.T), diff_time=diff_time)
            dataset.append(C.astype(np.float))
        elif rtype == "adj":
            # dataset.append(A)
            dataset.append(0.5 * (A + A.T))
        elif rtype == "rw":
            C = random_walk_sim(A)
            dataset.append(C.astype(np.float))
        labels.append(x[1])
    return dataset, labels, None, dc(dataset)


if __name__ == '__main__':
    dataset, labels, _, _, attributes = bzr("adj")
    print(dataset)
    print(attributes)
