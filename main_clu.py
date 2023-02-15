import numpy as np
from numpy import linalg as la
import torch
import networkx as nx
import random
import argparse
import pickle
from copy import deepcopy as dc
import ot
from ot.utils import unif
from datetime import datetime
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score  # python 3.6
from lib.sinkhorn import gromov_loss
from main_io import sbm_balanced_gaussian, sbm_balanced_edge, sbm_unbalanced_edge, sbm_unbalanced_gaussian, imdb_b, \
    deezer, github, imdb_m, mutag, msrc, kki, ohsu, peking
from methods_dict.gdl import gromov_wasserstein_linear_unmixing, gromov_wasserstein_dictionary_learning
from methods_dict.rgdl import unmixing, dictionary_learning
from methods_dict.sc import sc
from methods_dict.gwf import gwf


def evaluate(X: np.ndarray, labels: list, Cembedded_list: list, dataset_clean: list, trans_list: list, ps_list: list,
             pt_list: list, seed: int, n_clu: int):
    print("n_clu={}".format(n_clu))
    kmeans = KMeans(n_clusters=n_clu, init='k-means++', n_init=100, random_state=seed,
                    tol=10 ** (-9), max_iter=10 ** 5)
    pred = kmeans.fit_predict(X)
    # print(X)
    # print(pred)
    score = adjusted_rand_score(labels, pred)
    print("rand_score={}".format(
        score
    ))
    # print("rand_score={}".format(
    #     rand_score(labels, pred)
    # ))
    # total_cost = 0
    # n_graph = len(labels)
    # for i in range(n_graph):
    #     total_cost += gromov_loss(trans_list[i], Cembedded_list[i], dataset_clean[i], ps_list[i], pt_list[i])
    # print("total_cost={}".format(total_cost))
    return score


def evaluate_only_emb(X: np.array, labels: list, seed: int, n_clu: int):
    print("n_clu={}".format(n_clu))
    kmeans = KMeans(n_clusters=n_clu, init='k-means++', n_init=100, random_state=seed,
                    tol=10 ** (-9), max_iter=10 ** 5)
    pred = kmeans.fit_predict(X)
    # print(X)
    # print(pred)
    score = adjusted_rand_score(labels, pred)
    print("rand_score={}".format(
        score
    ))
    return score


def evaluate_only_pred(pred: list, labels: list):
    print("rand_score={}".format(
        adjusted_rand_score(labels, pred)
    ))
    return


def main(args):
    alg = args.alg
    eta_trans = args.et
    lr = args.lr
    dataname = args.dataset
    rtype = args.rtype  # relation type
    diff_time = args.diff_time
    reg = args.reg
    noise = args.noise
    bound = args.bound
    n_atom = args.n_atom
    n_clu = args.n_clu
    atom_size = args.atom_size
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    use_cg = args.use_cg
    use_gd = args.use_gd
    assert use_cg == False or use_gd == False
    ps = args.ps
    if ps:
        projection, ps_name = "nonnegative_symmetric", "ns"
    else:
        projection, ps_name = "symmetric", "s"
    # ======================= loading and splitting the dataset ===================
    if "sbm_gauss" in dataname:
        full_dataname = "{}_{}_{}".format(dataname, noise, rtype)
        dataset, labels, _, dataset_clean = sbm_balanced_gaussian(scale=noise)
    elif "sbm_un_gauss" in dataname:
        full_dataname = "{}_{}_{}".format(dataname, noise, rtype)
        dataset, labels, _, dataset_clean = sbm_unbalanced_gaussian(scale=noise)
    elif "sbm_edge" in dataname:
        full_dataname = "{}_{}_{}".format(dataname, noise, rtype)
        dataset, labels, _, dataset_clean = sbm_balanced_edge(scale=noise)
    elif "sbm_un_edge" in dataname:
        full_dataname = "{}_{}_{}".format(dataname, noise, rtype)
        dataset, labels, _, dataset_clean = sbm_unbalanced_edge(scale=noise)
    elif "imdb_b" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = imdb_b(rtype, diff_time=diff_time)
    elif "imdb_m" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = imdb_m(rtype, diff_time=diff_time)
    elif "mutag" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = mutag(rtype, diff_time=diff_time)
    elif "msrc" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = msrc(rtype, diff_time=diff_time)
    elif "kki" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = kki(rtype, diff_time=diff_time)
    elif "ohsu" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = ohsu(rtype, diff_time=diff_time)
    elif "peking" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = peking(rtype, diff_time=diff_time)
    elif "deezer" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = deezer(rtype, diff_time=diff_time)
    elif "github" in dataname:
        full_dataname = "{}_{}".format(dataname, rtype)
        dataset, labels, _, dataset_clean = github(rtype, diff_time=diff_time)
    else:
        raise NotImplementedError
    n_graph = len(dataset)
    ps_list = [unif(atom_size) for _ in range(n_graph)]
    pt_list = []
    for i in range(n_graph):
        n = dataset[i].shape[0]
        pt_list.append(unif(n))
    print("dataset loaded")

    timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    start = time.time()
    if alg == "gdl":
        if rtype == "heat":
            board_path = os.path.join("results_clu", full_dataname,
                                      timestamp + "_gdl_{}_{}_{}_sd{}".format(diff_time, ps_name, n_atom, seed))
        else:
            board_path = os.path.join("results_clu", full_dataname,
                                      timestamp + "_gdl_{}_{}_{}_sd{}".format(rtype, ps_name, n_atom, seed))
        print(board_path)
        writer = SummaryWriter(log_dir=board_path)
        global_step = 0
        for Cdict in gromov_wasserstein_dictionary_learning(dataset, n_atom, atom_size, reg=reg, projection=projection):
            X = np.zeros([n_graph, n_atom])
            Cembedded_list = []
            trans_list = []
            for i in range(n_graph):
                X[i], Cembedded, trans, _ = gromov_wasserstein_linear_unmixing(dataset[i], Cdict, reg=reg)
                Cembedded_list.append(Cembedded)
                trans_list.append(trans.T)
            score = evaluate(X=X, labels=labels, Cembedded_list=Cembedded_list, dataset_clean=dataset_clean,
                             trans_list=trans_list, ps_list=ps_list, pt_list=pt_list, seed=seed, n_clu=n_clu)
            runtime = time.time() - start
            global_step += 1
            writer.add_scalar(tag="ARI", scalar_value=score, global_step=global_step)
            writer.add_scalar(tag="runtime", scalar_value=runtime, global_step=global_step)
    elif alg == "rgdl":
        if use_cg:
            board_path = os.path.join("results_clu", full_dataname,
                                      timestamp + "_rgdl_{}_cg_{}_sd{}_lr{}_reg{}".format(ps_name, bound, seed, lr,
                                                                                          reg))
        elif use_gd:
            board_path = os.path.join("results_clu", full_dataname,
                                      timestamp + "_rgdl_{}_gd_{}_{}_sd{}_lr{}_{}".format(ps_name, eta_trans, bound,
                                                                                          seed, lr, reg))
        else:
            board_path = os.path.join("results_clu", full_dataname,
                                      timestamp + "_rgdl_{}_md_{}_{}_sd{}_lr{}_{}".format(ps_name, eta_trans, bound,
                                                                                          seed, lr, reg))
        print(board_path)
        writer = SummaryWriter(log_dir=board_path)
        global_step = 0
        for Cdict in dictionary_learning(dataset, n_atom, atom_size, eta_trans=eta_trans, bound=bound, reg=reg,
                                         learning_rate=lr, use_cg=use_cg, use_gd=use_gd, projection=projection):
            X = np.zeros([n_graph, n_atom])
            Cembedded_list = []
            trans_list = []
            for i in range(n_graph):
                X[i], Cembedded, trans, pert, fit_loss = unmixing(dataset[i], Cdict, ps=ps_list[i], pt=pt_list[i],
                                                                  eta_trans=eta_trans, bound=bound, reg=reg,
                                                                  use_cg=use_cg, use_gd=use_gd)
                Cembedded_list.append(Cembedded)
                trans_list.append(trans)
            score = evaluate(X=X, labels=labels, Cembedded_list=Cembedded_list, dataset_clean=dataset_clean,
                             trans_list=trans_list, ps_list=ps_list, pt_list=pt_list, seed=seed, n_clu=n_clu)
            runtime = time.time() - start
            global_step += 1
            writer.add_scalar(tag="ARI", scalar_value=score, global_step=global_step)
            writer.add_scalar(tag="runtime", scalar_value=runtime, global_step=global_step)
    elif alg == "sc":
        pred = sc(dataset, n_clu=n_clu, seed=seed)
        evaluate_only_pred(pred=pred, labels=labels)
        print(time.time() - start)
    elif alg == "gwf":
        board_path = os.path.join("results_clu", full_dataname, timestamp + "_gwf_sd{}".format(seed))
        print(board_path)
        writer = SummaryWriter(log_dir=board_path)
        global_step = 0
        for X in gwf(dataset, n_atom, atom_size):
            score = evaluate_only_emb(X, labels=labels, seed=seed, n_clu=n_clu)
            runtime = time.time() - start
            global_step += 1
            writer.add_scalar(tag="ARI", scalar_value=score, global_step=global_step)
            writer.add_scalar(tag="runtime", scalar_value=runtime, global_step=global_step)
    else:
        raise NotImplementedError
    if alg in ["gdl", "rgdl", "gwf"]:
        emb_path = os.path.join(board_path, "{}_{}.npy".format(dataname, alg))
        label_path = os.path.join(board_path, "{}_label.npy".format(dataname))
        np.save(emb_path, X)
        np.save(label_path, np.array(labels))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="The dataset used", default="sbm_un_gauss")
    parser.add_argument("-n", "--noise", help="noise level", default=0.05, type=float)
    parser.add_argument("--rtype", default="heat")
    parser.add_argument("--diff-time", help="diffusion time", default=10.0, type=float)
    parser.add_argument("--reg", help="regularizer", default=0.001, type=float)
    parser.add_argument("--n-clu", default=3, type=int)
    parser.add_argument("--atom-size", default=6, type=int)
    parser.add_argument("--n-atom", default=3, type=int)
    parser.add_argument("-a", "--alg", help="The method to run", default="gdl")
    parser.add_argument("--et", help="stepsize of transport plan", default=1e-4, type=float)
    parser.add_argument("--lr", help="stepsize of the dictionary", default=1.0, type=float)
    parser.add_argument("--bd", dest="bound", help="uncertainty set size", default=1, type=float)
    parser.add_argument("--ps", default=True, type=bool)
    parser.add_argument("--use-cg", default=False, type=bool)
    parser.add_argument("--use-gd", default=True, type=bool)
    parser.add_argument("--seed", help="Random seed", default=1, type=int)
    args = parser.parse_args()
    main(args)
