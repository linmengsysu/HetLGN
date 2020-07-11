import json, os
import math, copy, time
import numpy as np
from collections import defaultdict
import pandas as pd
from utils import *
import random
import math
from numpy.linalg import inv
from tqdm import tqdm

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp
from sample import *


class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        # self.edge_list = defaultdict( #target_type
        #                     lambda: defaultdict(  #source_type
        #                         lambda: defaultdict(  #relation_type
        #                             lambda: defaultdict(  #target_id
        #                                 lambda: defaultdict( #source_id(
        #                                     # lambda: int # time
        #                                 )))))
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}
    def add_node(self, node):
        nfl = self.node_forward[node['type']]
        if node['id'] not in nfl:
            self.node_bacward[node['type']] += [node]
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]

    def add_edge(self, source_node, target_node, time = None, relation_type=None, directed=True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
        '''
        # if source_node['id'] == 'STOC':
        # print('edge 1', edge)
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = 1
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = 1
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = 1
        # self.times[time] = True
        # print(self.edge_list[target_node['type']][source_node['type']].keys())
        
    def update_node(self, node):
        nbl = self.node_bacward[node['type']]
        ser = self.add_node(node)
        for k in node:
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())

    def del_edge(self, source_node, target_node, source_type, target_type, relation_type, directed=True):
        # print(target_type, source_type, self.edge_list[target_type][source_type].keys())
        self.edge_list[target_type][source_type][relation_type][target_node].pop(source_node)
        if directed:
            self.edge_list[source_type][target_type]['rev_' + relation_type][source_node].pop(target_node)
        else:
            self.edge_list[source_type][target_type][relation_type][source_node].pop(target_node)




def random_walk_restart(graph, sampled_number=8, sampled_depth = 2, inp=None, feature_extractor=feature_IsoNode):
    subgraph_data = defaultdict( #target type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # sampled_type
                            lambda: defaultdict(  # sampled_id
                                lambda: []  # [ser]
                            ))))

    for _type in inp:
        for _id in inp[_type]:
            subgraph_data[_type][_id] = sample_subgraph_v3(graph, sampled_number, sampled_depth, inp={'paper': [_id]})

    # return subgraph_data
    feature, indxs, texts = feature_extractor(subgraph_data, graph)

     
    return feature, subgraph_data, indxs, texts


def normalize(adj):
    row_sum = np.sum(adj, axis=0)
    d = np.diag(row_sum).inv()
    d[d==np.inf] = 0
    norm = np.dot(adj, d)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).toarray()

def to_torch_iso(feature, subgraph_data, graph, sample_number):
        c = 0.5
        '''
            Transform a sampled sub-graph into pytorch Tensor
            node_dict[type][0] (a number) stores the start index of a specific type of nodes
            node_dict: {target_id: {node_type: <node_number, node_type_ID>} } node_number is used to trace back the nodes in original graph.
            edge_dict: {edge_type: edge_type_ID}
        '''
        node_dict = defaultdict(lambda: defaultdict())
        node_feature = []
        node_type = []
        # node_time    = []
        edge_index = []
        edge_type = []
        # edge_time    = []
        adj = []
        types = graph.get_types()
        for target_id in feature:
            node_num = 0
            for t in types:
                if t in feature[target_id]:
                    node_dict[target_id][t] = [node_num, len(node_dict[target_id])]
                    node_num += len(feature[target_id][t])

        for target_id in feature:
            tmp_type = []
            tmp_node_feature = []
            for t in types:
                if t in feature[target_id]:
                    tmp_node_feature += list(feature[target_id][t])
                    n_feature = feature[target_id][t].shape[1]
                    tmp_type += [node_dict[target_id][t][1] for _ in range(len(feature[target_id][t]))]
           
            node_type.append(tmp_type)
            node_feature.append(tmp_node_feature)

        # edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
        # edge_dict['self'] = len(edge_dict)

        for target_type in subgraph_data:
            for target_id in subgraph_data[target_type]:
                tmp_adj = np.zeros((sample_number, sample_number))
                for t1 in subgraph_data[target_type][target_id]:
                    for s1 in subgraph_data[target_type][target_id][t1]:
                        for t2 in subgraph_data[target_type][target_id]:
                            for s2 in subgraph_data[target_type][target_id][t2]:
                                try:
                                    for r_type in graph.edge_list[t1][t2]:
                                        if s2 in graph.edge_list[t1][t2][r_type][s1]:
                                            sid1 = subgraph_data[target_type][target_id][t1][s1][0]
                                            sid2 = subgraph_data[target_type][target_id][t2][s2][0]
                                            tid, sid = sid1 + node_dict[t1][0], sid2 + node_dict[t2][0]
                                            tmp_adj[tid, sid], tmp_adj[sid, tid] = 1., 1.
                                except:
                                    # print('t1={}, t2={}'.format(t1, t2))
                                    continue
                tmp_adj = normalize_adj(tmp_adj + np.eye(tmp_adj.shape[0]))
                # tmp_adj = c * inv(np.eye(tmp_adj.shape[0]) - (1 - c) * normalize_adj(tmp_adj))
                adj.append(tmp_adj)

        node_type = torch.FloatTensor(node_type).unsqueeze(1)  # [B, 1, n_sample]
        adj = torch.FloatTensor(adj).unsqueeze(1)  # [B, 1, k, k]
        node_feature = torch.FloatTensor(node_feature) #[B, n_sample, in_dim]

        print('node features size={}, node type size={},  normalized adj={}'.format(node_feature.size(), node_type.size(), adj.size()))
        return node_feature, node_type, adj, node_dict
