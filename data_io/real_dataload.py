import numpy as np
from collections import defaultdict
from tqdm import tqdm
import networkx as nx


def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret


def graph_label_list(path, name, real=False):
    graphs = []
    with open(path + name) as f:
        sections = list(per_section(f))
        k = 1
        for elt in sections[0]:
            if real:
                graphs.append((k, float(elt)))
            else:
                graphs.append((k, int(elt)))
            k = k + 1
    return graphs


def graph_indicator(path, name):
    data_dict = defaultdict(list)
    with open(path + name) as f:
        sections = list(per_section(f))
        k = 1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k = k + 1
    return data_dict


def compute_adjency(path, name):
    adjency = defaultdict(list)
    with open(path + name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def node_labels_dic(path, name):
    node_dic = dict()
    with open(path + name) as f:
        sections = list(per_section(f))
        k = 1
        for elt in sections[0]:
            node_dic[k] = int(elt)
            k = k + 1
    return node_dic


def node_attr_dic(path, name):
    node_dic = dict()
    with open(path + name) as f:
        sections = list(per_section(f))
        k = 1
        for elt in sections[0]:
            node_dic[k] = [float(x) for x in elt.split(',')]
            k = k + 1
    return node_dic


def build_dataset_withoutfeatures(dataset_name, path, use_node_deg=False):
    # assert dataset_name in ['IMDB-MULTI', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']
    graphs = graph_label_list(path, '%s_graph_labels.txt' % dataset_name)
    adjency = compute_adjency(path, '%s_A.txt' % dataset_name)
    data_dict = graph_indicator(path, '%s_graph_indicator.txt' % dataset_name)
    data = []
    for i in tqdm(graphs, desc='loading graphs'):
        g = nx.Graph()
        for node in data_dict[i[0]]:
            g.name = i[0]
            g.add_node(node)
            # g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge(node, node2)
        data.append((g, i[1]))

    return data


def build_dataset_realfeatures(dataset_name, path, type_attr='label', use_node_deg=False):
    assert dataset_name in ['PROTEINS_full', 'PROTEINS', 'ENZYMES', 'BZR', 'COX2']
    if type_attr == 'label':
        node_dic = node_labels_dic(path, '%s_node_labels.txt' % dataset_name)
    if type_attr == 'real':
        node_dic = node_attr_dic(path, '%s_node_attributes.txt' % dataset_name)
    graphs = graph_label_list(path, '%s_graph_labels.txt' % dataset_name)
    adjency = compute_adjency(path, '%s_A.txt' % dataset_name)
    data_dict = graph_indicator(path, '%s_graph_indicator.txt' % dataset_name)
    data = []
    for i in graphs:
        g = nx.Graph()
        for node in data_dict[i[0]]:
            g.name = i[0]
            g.add_node(node)

            nx.set_node_attributes(g, {node: node_dic[node]}, name="att")
            # nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
            # g.add_one_attribute(node, node_dic[node])
            for node2 in adjency[node]:
                g.add_edge(node, node2)
        data.append((g, i[1]))

    return data
