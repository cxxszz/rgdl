import copy
import networkx as nx
import time, os, pandas, pickle, random, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from sklearn.manifold import MDS, TSNE


class StructuralDataSampler(Dataset):
    """Sampling point sets via minbatch"""

    def __init__(self, data: List):
        """
        Args:
            data: a list of data include [[edges, #nodes, (optional label)], ...]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        adj = self.data[idx][0]
        num_nodes = self.data[idx][1]
        dist = np.ones((num_nodes, 1))
        dist /= np.sum(dist)

        features = self.data[idx][2]

        features = torch.from_numpy(features).type(torch.FloatTensor)
        dist = torch.from_numpy(dist).type(torch.FloatTensor)
        adj = torch.from_numpy(adj).type(torch.FloatTensor)
        label = torch.LongTensor([self.data[idx][-1]])

        return [adj, dist, features, label]


def cost_mat(cost_s: torch.Tensor, cost_t: torch.Tensor, p_s: torch.Tensor, p_t: torch.Tensor, tran: torch.Tensor,
             emb_s: torch.Tensor = None, emb_t: torch.Tensor = None) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
        emb_s: (ns, d) matrix
        emb_t: (nt, d) matrix
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = ((cost_s ** 2) @ p_s).repeat(1, tran.size(1))
    f2_st = (torch.t(p_t) @ torch.t(cost_t ** 2)).repeat(tran.size(0), 1)
    cost_st = f1_st + f2_st
    cost = cost_st - 2 * cost_s @ tran @ torch.t(cost_t)
    # if emb_s is not None and emb_t is not None:
    #     tmp1 = emb_s @ torch.t(emb_t)
    #     tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
    #     tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
    #     cost += 0.5 * (1 - tmp1 / (tmp2 @ torch.t(tmp3)))
    #     # tmp1 = 2 * emb_s @ torch.t(emb_t)
    #     # tmp2 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, tran.size(1))
    #     # tmp3 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, tran.size(0))
    #     # tmp = 0.1 * (tmp2 + torch.t(tmp3) - tmp1) / (emb_s.size(1) ** 2)
    #     # cost += tmp
    return cost


def fgwd(graph1, embedding1, prob1, graph2, embedding2, prob2, tran):
    # input(graph1)
    # input(embedding1)
    # input(prob1)
    # input(graph2)
    # input(embedding2)
    # input(prob2)
    # input(tran)
    cost = cost_mat(graph1, graph2, prob1, prob2, tran, embedding1, embedding2)

    return (cost * tran).sum()


class FGWF(nn.Module):
    """
    A simple PyTorch implementation of Fused Gromov-Wasserstein factorization model
    The feed-forward process imitates the proximal point algorithm or bregman admm
    """

    def __init__(self, num_samples: int, num_classes: int, size_atoms: List, dim_embedding: int = 1,
                 ot_method: str = 'ppa', gamma: float = 1e-1, gwb_layers: int = 5, ot_layers: int = 5, prior=None):
        """
        Args:
            num_samples: the number of samples
            size_atoms: a list, its length is the number of atoms, each element is the size of the corresponding atom
            dim_embedding: the dimension of embedding
            ot_method: ppa or b-admm
            gamma: the weight of Bregman divergence term
            gwb_layers: the number of gwb layers in each gwf module
            ot_layers: the number of ot layers in each gwb module
        """
        super(FGWF, self).__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size_atoms = size_atoms
        self.num_atoms = len(self.size_atoms)
        self.dim_embedding = dim_embedding
        self.ot_method = ot_method
        self.gwb_layers = gwb_layers
        self.ot_layers = ot_layers
        self.gamma = gamma

        # weights of atoms
        self.weights = nn.Parameter(torch.randn(self.num_atoms, self.num_samples))
        self.softmax = nn.Softmax(dim=0)

        # basis and their node distribution
        if prior is None:
            self.ps = []
            self.atoms = nn.ParameterList()
            self.embeddings = nn.ParameterList()
            for k in range(self.num_atoms):
                atom = nn.Parameter(torch.randn(self.size_atoms[k], self.size_atoms[k]))
                embedding = nn.Parameter(torch.randn(self.size_atoms[k], self.dim_embedding) / self.dim_embedding)
                dist = torch.ones(self.size_atoms[k], 1) / self.size_atoms[k]  # .type(torch.FloatTensor)
                self.ps.append(dist)
                self.atoms.append(atom)
                self.embeddings.append(embedding)
        else:
            # num_atoms_per_class = int(self.num_atoms / self.num_classes)
            # counts = np.zeros((self.num_classes,))
            # self.ps = []
            # self.atoms = []
            # self.size_atoms = []
            # self.embeddings = []
            # base_label = []
            # for n in range(prior.__len__()):
            #     data = prior.__getitem__(n)
            #     graph = data[0]
            #     prob = data[1]
            #     emb = data[2]
            #     gt = int(data[3][0])
            #     if counts[gt] < num_atoms_per_class:
            #         self.size_atoms.append(graph.size(0))
            #         atom = nn.Parameter(graph)
            #         embedding = nn.Parameter(emb)
            #         self.ps.append(prob)
            #         self.atoms.append(atom)
            #         self.embeddings.append(embedding)
            #         base_label.append(gt)
            #         counts[gt] += 1

            num_samples = prior.__len__()
            index_samples = list(range(num_samples))
            random.shuffle(index_samples)
            self.ps = []
            self.atoms = nn.ParameterList()
            self.embeddings = nn.ParameterList()
            base_label = []
            for k in range(self.num_atoms):
                idx = index_samples[k]
                data = prior.__getitem__(idx)
                graph = data[0]
                prob = data[1]
                emb = data[2]
                gt = data[3]
                self.size_atoms[k] = graph.size(0)
                atom = nn.Parameter(graph)
                embedding = nn.Parameter(emb)
                self.ps.append(prob)
                self.atoms.append(atom)
                self.embeddings.append(embedding)
                base_label.append(gt[0])

            print(self.size_atoms)
            print(base_label)
        self.sigmoid = nn.Sigmoid()

    def output_weights(self, idx: int = None):
        if idx is not None:
            return self.softmax(self.weights[:, idx])
        else:
            return self.softmax(self.weights)

    def output_atoms(self, idx: int = None):
        if idx is not None:
            return self.sigmoid(self.atoms[idx])
        else:
            return [self.sigmoid(self.atoms[idx]) for idx in range(len(self.atoms))]

    def fgwb(self, pb: torch.Tensor, trans: List, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve GW Barycetner problem
        barycenter = argmin_{B} sum_k w[k] * d_gw(atom[k], B) via proximal point-based alternating optimization:

        step 1: Given current barycenter, for k = 1:K, we calculate trans[k] by the OT-PPA layer.
        step 2: Given new trans, we update barycenter by
            barycenter = sum_k trans[k] * atom[k] * trans[k]^T / (pb * pb^T)

        Args:
            pb: (nb, 1) vector (torch tensor), the empirical distribution of the nodes/samples of the barycenter
            trans: a dictionary {key: index of atoms, value: the (ns, nb) initial optimal transport}
            weights: (K,) vector (torch tensor), representing the weights of the atoms

        Returns:
            barycenter: (nb, nb) matrix (torch tensor) representing the updated GW barycenter
        """
        tmp1 = pb @ torch.t(pb)
        tmp2 = pb @ torch.ones(1, self.dim_embedding)
        graph = torch.zeros(pb.size(0), pb.size(0))
        embedding = torch.zeros(pb.size(0), self.dim_embedding)
        for k in range(self.num_atoms):
            graph_k = self.output_atoms(k)
            graph += weights[k] * (torch.t(trans[k]) @ graph_k @ trans[k])
            # input(graph)
            embedding += weights[k] * (torch.t(trans[k]) @ self.embeddings[k])
        graph = graph / tmp1

        embedding = embedding / tmp2
        return graph, embedding

    def forward(self, graph: torch.Tensor, prob: torch.Tensor, embedding: torch.Tensor,
                index: int, trans: List, tran: torch.Tensor):
        """
        For "n" unknown samples, given their disimilarity/adjacency matrix "cost" and distribution "p", we calculate
        "d_gw(barycenter(atoms, weights), cost)" approximately.

        Args:
            graph: (n, n) matrix (torch.Tensor), representing disimilarity/adjacency matrix
            prob: (n, 1) vector (torch.Tensor), the empirical distribution of the nodes/samples in "graph"
            embedding: (n, d) matrix (torch.Tensor)
            index: the index of the "cost" in the dataset
            trans: a list of (ns, nb) OT matrices
            tran: a (n, nb) OT matrix

        Returns:
            d_gw: the value of loss function
            barycenter: the proposed GW barycenter
            tran0: the optimal transport between barycenter and cost
            trans: the optimal transports between barycenter and atoms
            weights: the weights of atoms
        """
        # variables
        weights = self.softmax(self.weights[:, index])
        graph_b, embedding_b = self.fgwb(prob, trans, weights)
        d_fgw = fgwd(graph, embedding, prob, graph_b, embedding_b, prob, tran)
        # input(d_fgw)
        return d_fgw, self.weights[:, index], graph_b, embedding_b


def tsne_weights(model) -> np.ndarray:
    """
    Learn the 2D embeddings of the weights associated with atoms via t-SNE
    Returns:
        embeddings: (num_samples, 2) matrix representing the embeddings of weights
    """
    model.eval()
    features = model.weights.cpu().data.numpy()
    features = features.T
    if features.shape[1] == 2:
        embeddings = features
    else:
        embeddings = TSNE(n_components=2).fit_transform(features)
    return embeddings


def ot_fgw(cost_s: torch.Tensor, cost_t: torch.Tensor, p_s: torch.Tensor, p_t: torch.Tensor, ot_method: str,
           gamma: float, num_layer: int, emb_s: torch.Tensor = None, emb_t: torch.Tensor = None):
    tran = p_s @ torch.t(p_t)
    if ot_method == 'ppa':
        dual = torch.ones(p_s.size()) / p_s.size(0)
        for m in range(num_layer):
            cost = cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t)
            # cost /= torch.max(cost)
            # kernel = torch.exp(-cost / gamma) * tran
            kernel = torch.exp(-cost / gamma) * tran
            # input(kernel)
            b = p_t / (torch.t(kernel) @ dual + 1e-7)
            for i in range(5):
                dual = p_s / (kernel @ b + 1e-7)
                b = p_t / (torch.t(kernel) @ dual + 1e-7)
            tran = (dual @ torch.t(b)) * kernel

    elif ot_method == 'b-admm':
        all1_s = torch.ones(p_s.size())
        all1_t = torch.ones(p_t.size())
        dual = torch.zeros(p_s.size(0), p_t.size(0))
        for m in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a

            dual = dual + gamma * (tran - aux)

            cost = cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t)
            # cost /= torch.max(cost)
            kernel_t = torch.exp(-(cost + dual) / gamma) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) * tran).sum()
    return d_gw, tran


def train_usl(model, database, size_batch: int = 16, epochs: int = 10, lr: float = 1e-1, weight_decay: float = 0,
              shuffle_data: bool = True, zeta: float = None, mode: str = 'fit', visualize_prefix: str = None):
    """
    training a FGWF model
    Args:
        model: a FGWF model
        database: a list of data, each element is a list representing [cost, distriubtion, feature, label]
        size_batch: the size of batch, deciding the frequency of backpropagation
        epochs: the number epochs
        lr: learning rate
        weight_decay: the weight of the l2-norm regularization of parameters
        shuffle_data: whether shuffle data in each epoch
        zeta: the weight of the regularizer enhancing the diversity of atoms
        mode: fit or transform
        visualize_prefix: display learning result after each epoch or not
    """
    if mode == 'fit':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        n = 0
        for param in model.parameters():
            if n > 0:
                param.requires_grad = False
            n += 1

        # only update partial model's parameters
        optimizer = optim.Adam([list(model.parameters())[0]], lr=lr, weight_decay=weight_decay)
    model.train()

    data_sampler = StructuralDataSampler(database)
    num_samples = data_sampler.__len__()
    index_samples = list(range(num_samples))
    index_atoms = list(range(model.num_atoms))

    best_loss = float("Inf")
    best_model = None
    for epoch in range(epochs):
        counts = 0
        t_start = time.time()
        loss_epoch = 0
        loss_total = 0
        d_fgw_total = 0
        reg_total = 0
        optimizer.zero_grad()

        if shuffle_data:
            random.shuffle(index_samples)

        for idx in index_samples:
            data = data_sampler.__getitem__(idx)
            graph = data[0]
            prob = data[1]
            emb = data[2]

            # Envelop Theorem
            # feed-forward computation of barycenter B({Ck}, w) and its transports {Trans_k}
            trans = []
            for k in range(model.num_atoms):
                graph_k = model.output_atoms(k).data
                emb_k = model.embeddings[k].data
                _, tran_k = ot_fgw(graph_k, graph, model.ps[k], prob,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb_k, emb)
                # input(tran_k)
                trans.append(tran_k)
            tran = torch.diag(prob[:, 0])
            # trans = []
            # graph_b = graph
            # emb_b = emb
            # weights = model.output_weights(idx).data
            # tmp1 = prob @ torch.t(prob)
            # tmp2 = prob @ torch.ones(1, model.dim_embedding)
            # for n in range(model.gwb_layers):
            #     graph_b_tmp = 0
            #     emb_b_tmp = 0
            #     trans = []
            #     for k in range(model.num_atoms):
            #         graph_k = model.output_atoms(k).data
            #         emb_k = model.embeddings[k].data
            #         _, tran_k = ot_fgw(graph_k, graph_b, model.ps[k], prob,
            #                            model.ot_method, model.gamma, model.ot_layers,
            #                            emb_k, emb_b)
            #         trans.append(tran_k)
            #         graph_b_tmp += weights[k] * (torch.t(tran_k) @ graph_k @ tran_k)
            #         emb_b_tmp += weights[k] * (torch.t(tran_k) @ emb_k)
            #     graph_b = graph_b_tmp / tmp1
            #     emb_b = emb_b_tmp / tmp2

            # _, tran = ot_fgw(graph, graph_b, prob, prob,
            #                  model.ot_method, model.gamma, model.ot_layers,
            #                  emb, emb_b)

            d_fgw, _, _, _ = model(graph, prob, emb, idx, trans, tran)

            d_fgw_total += d_fgw
            loss_total += d_fgw

            if zeta is not None and mode == 'fit':
                random.shuffle(index_atoms)
                graph1 = model.output_atoms(index_atoms[0])
                emb1 = model.embeddings[index_atoms[0]]
                p1 = model.ps[index_atoms[0]]

                graph2 = model.output_atoms(index_atoms[1])
                emb2 = model.embeddings[index_atoms[1]]
                p2 = model.ps[index_atoms[1]]

                _, tran12 = ot_fgw(graph1.data, graph2.data, p1, p2,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb1.data, emb2.data)
                reg = fgwd(graph1, emb1, p1, graph2, emb2, p2, tran12)

                reg_total += zeta * reg
                loss_total -= zeta * reg

            counts += 1
            if counts % size_batch == 0 or counts == num_samples:
                if counts % size_batch == 0:
                    num = size_batch
                else:
                    num = counts % size_batch
                loss_epoch += loss_total
                loss_total.backward()
                optimizer.step()

                print('-- {}/{} [{:.1f}%], loss={:.4f}, dgw={:.4f}, reg={:.4f}, time={:.2f}s.'.format(
                    counts, num_samples, counts / num_samples * 100.0,
                                         loss_total / num, d_fgw_total / num, reg_total / num, time.time() - t_start))

                t_start = time.time()
                loss_total = 0
                d_fgw_total = 0
                reg_total = 0
                optimizer.zero_grad()

        if best_loss > loss_epoch.data / num_samples:
            best_model = copy.deepcopy(model)
            best_loss = loss_epoch.data / num_samples

        print('{}: Epoch {}/{}, loss = {:.4f}, best loss = {:.4f}'.format(
            mode, epoch + 1, epochs, loss_epoch / num_samples, best_loss))
        yield best_model


def gwf(C_list, n_atom, atom_size, batch_size=32, epochs=50, ):
    n_graph = len(C_list)
    # ====================================== data conversion ================================
    graph_data = []
    for C in C_list:
        graph_data.append([C, len(C), np.zeros(2), 0])
    ot_method = 'ppa'
    gamma = 1e-1
    gwb_layers = 5
    ot_layers = 50
    lr = 0.25
    weight_decay = 0
    shuffle_data = True
    zeta = None  # the weight of diversity regularizer
    mode = 'fit'
    model = FGWF(num_samples=n_graph, num_classes=n_atom, size_atoms=[atom_size] * n_atom, dim_embedding=n_atom,
                 ot_method=ot_method, gamma=gamma, gwb_layers=gwb_layers, ot_layers=ot_layers, prior=None)
    for model in train_usl(model, graph_data, size_batch=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay,
                           shuffle_data=shuffle_data, zeta=zeta, mode=mode):
        model.eval()
        features = model.weights.cpu().data.numpy()
        embeddings = features.T
        yield embeddings
