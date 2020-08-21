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
from torch_geometric.data import Data
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

import dill
from functools import partial
import multiprocessing as mp
from sample import *
from utils import feature_HetLGN


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

            

def random_walk_restart(graph, sampled_number=8, sampled_depth = 2, inp=None, feature_extractor=feature_HetLGN):
    subgraph_data = defaultdict( #target type
                    lambda: defaultdict(  # target_id
                        lambda: defaultdict(  # sampled_type
                            lambda: defaultdict(  # sampled_id
                                lambda: []  # [ser]
                            ))))

    size = defaultdict(lambda:0)
    for _type in inp:
        for _id in inp[_type]:
            subgraph_data[_type][_id], size[_id]  = sample_subgraph_v3(graph, sampled_number, sampled_depth, inp={_type: [_id]})
            # print('subgraph_data', subgraph_data[_type][_id])
    feature = feature_extractor(subgraph_data, graph, sampled_depth)
    # print(feature)
    
    return feature, subgraph_data


def to_torch_mp(feature, subgraph_data, graph, label):
    node_dict = defaultdict(lambda: defaultdict())
    node_feature = defaultdict(lambda:[])
    node_type    = defaultdict(lambda:[])
    
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    types = graph.get_types()
    for target_type in subgraph_data:
        for target_id in subgraph_data[target_type]:
                node_num = 0
                for t in subgraph_data[target_type][target_id]:
                    node_dict[target_id][t] = [node_num, len(node_dict[target_id])]
                    node_num += len(subgraph_data[target_type][target_id][t])

    for target_id in feature:
        tmp_type = []
        tmp_node_feature = []
        for t in types:
            if t in feature[target_id]:
                tmp_node_feature += list(feature[target_id][t])
                try:
                    n_feature = feature[target_id][t].shape[1]
                except:
                    print('in except', feature[target_id][t])
                tmp_type += [node_dict[target_id][t][1] for _ in range(len(feature[target_id][t]))]
        
        node_type[target_id] = tmp_type
        node_feature[target_id] = torch.FloatTensor(tmp_node_feature)
        
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)
    subgraphs = []
    
    for target_type in subgraph_data:
        for target_id in subgraph_data[target_type]:
            tmp_edge_index = []
            tmp_edge_type = []
           
            for t1 in subgraph_data[target_type][target_id]:
                for s1 in subgraph_data[target_type][target_id][t1]:
                    for t2 in subgraph_data[target_type][target_id]:
                        for s2 in subgraph_data[target_type][target_id][t2]:
                            try:
                                for r_type in graph.edge_list[t1][t2]:
                                    if s2 in graph.edge_list[t1][t2][r_type][s1]:
                                        sid1 = subgraph_data[target_type][target_id][t1][s1][0]
                                        sid2 = subgraph_data[target_type][target_id][t2][s2][0]
                                        tid, sid = sid1 + node_dict[target_id][t1][0], sid2 + node_dict[target_id][t2][0]
                                        tmp_edge_index += [[sid, tid]]
                                        tmp_edge_type += [edge_dict[r_type]]
                            except:
                                # print('t1={}, t2={}'.format(t1, t2))
                                continue
            sid1 = subgraph_data[target_type][target_id][target_type][target_id][0]
            tid = sid1  + node_dict[target_id][target_type][0]
            tmp_edge_index += [[tid, tid]]
            tmp_edge_type  += [edge_dict['self']] 
            tmp_node_features = node_feature[target_id]
            # print(tmp_node_features)
            tmp_node_type    = torch.LongTensor(node_type[target_id])
            tmp_edge_type    = torch.LongTensor(tmp_edge_type)
            tmp_edge_index   = torch.LongTensor(tmp_edge_index).t().contiguous()
            g = Data(x=tmp_node_features, edge_index=tmp_edge_index, edge_type=tmp_edge_type, node_type=tmp_node_type, y=label[target_id])
            subgraphs.append(g)
    
    return subgraphs
    
