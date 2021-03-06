'''
    Author: Lin Meng
    Made modification on sample_subgraph in HGT 
    reference: https://github.com/acbull/pyHGT/tree/master/pyHGT
    
    
'''
import dill
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import math, copy, time, os
from data import *
from tqdm import tqdm
from collections import defaultdict, Counter
import networkx as nx
import argparse
from utils import *



'''
    layer balanced sample
    Input:
        sampled_number:  number of nodes per layer
        sampled_depth:   number of search depth
        inp:             target node dict {node_type: node_id}   

    output:
        sampled context graph: {node_type: node_id}
'''
def sample_subgraph_v3(graph, sampled_number = 8, sampled_depth = 3, inp = None):
    layer_data  = defaultdict( #node_type
                        lambda: {} # {target_id: [ser]}
                    )
    budget     = defaultdict( #(node_type, node_id)
                                        lambda: [0.] #[sampled_score]
                            )
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider a fixed number of nodes
    '''
    def add_budget(te, target_id, layer_data, budget):
        for source_type in te:
            tes = te[source_type]
            for relation_type in tes:
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                adl = tes[relation_type][target_id]
                # print('adl', adl.keys())
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
                for source_id in sampled_ids:
                    if source_id in layer_data[source_type]:
                        continue
                    budget[(source_type,source_id)][0] += 1. / len(sampled_ids)

    # np.random.seed(23)
    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id in inp[_type]:
            # print(_type, _id)
            layer_data[_type][_id] = [len(layer_data[_type]), 0]
    for _type in inp:
        te = graph.edge_list[_type]
        for _id in inp[_type]:
            add_budget(te, _id, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
    '''
    count = defaultdict(lambda:0)
    for layer in range(sampled_depth):
        keys = np.array(list(budget.keys()))
        if len(keys) == 0:

            continue
        if sampled_number > len(keys):
            '''
                Directly sample all the nodes
            '''
            sampled_ids = np.arange(len(keys))
        else:
            '''
                Sample based on accumulated degree
            '''
            score = np.array(list(budget.values()))[:,0] ** 2
            # print('scores={}'.format(score))
            score = score / np.sum(score)
            sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
        # print('sampled_ids',sampled_ids)
        sampled_keys = keys[sampled_ids]
        '''
            First adding the sampled nodes then updating budget.
        '''
        for k_type, k_id in sampled_keys:
            k_id = int(k_id)
            # if sum(count.values())+1 >= sampled_depth*sampled_number:
            #     break
            layer_data[k_type][k_id] = [len(layer_data[k_type]), layer] # id of node k in layer data not in orignal graph
            count[k_type] += 1
        for k_type, k_id in sampled_keys:
            k_id = int(k_id)
            te = graph.edge_list[k_type]
            add_budget(te, k_id, layer_data, budget) 
            budget.pop((k_type, k_id))   # already added k into sampled nodes set
    # del budget
    return layer_data, sum(count.values())



# '''
#     type balanced sample
# '''
# def sample_subgraph_v4(graph, sampled_number = 8, sampled_depth = 3, inp = None):
#     # print('type balanced sampling')
#     layer_data  = {}
#     budget    = defaultdict( #source_type
#                             lambda: defaultdict(  #source_id
#                             lambda: [0.] #[sampled_score]
#                     ))
    
#     '''
#         For each node being sampled, we find out all its neighborhood, 
#         adding the degree count of these nodes in the budget.
#         Note that there exist some nodes that have many neighborhoods
#         (such as fields, venues), for those case, we only consider 
#     '''
    
#     def add_budget(te, target_id, layer_data, budget):
#         for source_type in te:
#             tes = te[source_type]
#             for relation_type in tes:
#                 if relation_type == 'self' or target_id not in tes[relation_type]:
#                     continue
#                 adl = tes[relation_type][target_id]
#                 if len(adl) < sampled_number:
#                     sampled_ids = list(adl.keys())
#                 else:
#                     sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
#                 for source_id in sampled_ids:
#                     if source_type in layer_data and source_id in layer_data[source_type]:
#                         continue
#                     budget[source_type][source_id][0] += 1. / len(sampled_ids)

#     '''
#         First adding the sampled nodes then updating budget.
#     '''
#     for _type in inp:
#         for _id in inp[_type]:
#             if _type not in layer_data:
#                 layer_data[_type] = {}
      
#             layer_data[_type][_id] = [len(layer_data[_type]), 0]
#     for _type in inp:
#         te = graph.edge_list[_type]
#         for _id in inp[_type]:
#             add_budget(te, _id, layer_data, budget)
#     '''
#         We recursively expand the sampled graph by sampled_depth.
#         Each time we sample a fixed number of nodes for each budget,
#         based on the accumulated degree.

#         condition
#         1. if all types have enough nodes for a target node, then do nothing
#         2. if there are types don't have enough then mark it. let other types make up.
#         3. choose high degree?? if too much; ok solved
#     '''
#     count = defaultdict(lambda:0)
#     for layer in range(sampled_depth):
#         sts = list(budget.keys())
#         for source_type in sts:
#             te = graph.edge_list[source_type]
#             keys  = np.array(list(budget[source_type].keys())) # number of sampled nodes of one type in budget
#             if sampled_number >= len(keys):
#                 '''
#                     Directly sample all the nodes
#                 '''
#                 sampled_ids = np.arange(len(keys))
#             else:
#                 '''
#                     Sample based on accumulated degree
#                 '''
#                 score = np.array(list(budget[source_type].values()))[:,0] ** 2
#                 score = score / np.sum(score)
#                 sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False) 
#             sampled_keys = keys[sampled_ids]
#             '''
#                 First adding the sampled nodes then updating budget.
#             '''
#             for k in sampled_keys:
#                 if source_type not in layer_data:
#                     layer_data[source_type] = {}

#                 layer_data[source_type][k] = [len(layer_data[source_type]), layer] # id of node k in layer data not in orignal graph
#                 count[source_type] += 1
#             for k in sampled_keys:
#                 add_budget(te, k, layer_data, budget)
#                 budget[source_type].pop(k)   # already added k into sampled nodes set
#     del budget
#     return layer_data, sum(count.values())



# '''
#     accumulate degree ordered sampling
# '''
# def sample_subgraph_v5(graph, sampled_number = 8, sampled_depth = 3, inp = None):
#     layer_data  = defaultdict( #target_type
#                         lambda: {} # {target_id: [ser]}
#                     )
#     budget     = defaultdict( #source_type
#                                         lambda: [0.] #[sampled_score]
#                             )
#     '''
#         For each node being sampled, we find out all its neighborhood, 
#         adding the degree count of these nodes in the budget.
#         Note that there exist some nodes that have many neighborhoods
#         (such as fields, venues), for those case, we only consider 
#     '''
#     def add_budget(te, target_id, layer_data, budget):
#         for source_type in te:
#             tes = te[source_type]
#             for relation_type in tes:
#                 if relation_type == 'self' or target_id not in tes[relation_type]:
#                     continue
#                 adl = tes[relation_type][target_id]
               
#                 if len(adl) < sampled_number:
#                     sampled_ids = list(adl.keys())
#                 else:
#                     sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
               
#                 for source_id in sampled_ids:
#                     if source_id in layer_data[source_type]:
#                         continue
#                     budget[(source_type,source_id)][0] += 1. / len(sampled_ids)
#     '''
#         First adding the sampled nodes then updating budget.
#     '''
#     for _type in inp:
#         for _id in inp[_type]:
#             layer_data[_type][_id] = [len(layer_data[_type]),0]
#     for _type in inp:
#         te = graph.edge_list[_type]
#         for _id in inp[_type]:
#             add_budget(te, _id, layer_data, budget)
#     '''
#         We recursively expand the sampled graph by sampled_depth.
#         Each time we sample a fixed number of nodes for each budget,
#         based on the accumulated degree.
#     '''
#     count = defaultdict(lambda:0)
#     empty_type = []
#     for layer in range(sampled_depth):
#         keys = np.array(list(budget.keys()))
#         if len(keys) == 0:
#             continue
#         '''
#             Directly sample all the nodes, sampled_number is for each type
#         '''
#         if sampled_number > len(keys):
#             '''
#                 Directly sample all the nodes, sampled_number is for each type
#             '''
#             sampled_ids = np.arange(len(keys))
#         else:
#             '''
#                 Sample based on accumulated degree
#             '''
#             score = np.array(list(budget.values()))[:,0] ** 2
#             score = score / np.sum(score)
#             sampled_ids = np.argsort(score)[-sampled_number:]

       
#         sampled_keys = keys[sampled_ids]
#         '''
#             First adding the sampled nodes then updating budget.
#         '''
#         for k_type, k_id in sampled_keys:
#             k_id = int(k_id)
#             layer_data[k_type][k_id] = [len(layer_data[k_type]), layer] # id of node k in layer data not in orignal graph
#             count[k_type] += 1
#         for k_type, k_id in sampled_keys:
#             k_id = int(k_id)
#             te = graph.edge_list[k_type]
#             add_budget(te, k_id, layer_data, budget) 
#             budget.pop((k_type, k_id))   # already added k into sampled nodes set

#     return layer_data, sum(count.values())


# function for data analysis
# def HGT_sample(graph, sampled_number, sampled_depth, inp=None):
#     subgraph_data = defaultdict(  # target type
#         lambda: defaultdict(  # target_id
#             lambda: defaultdict(  # sampled_type
#                 lambda: defaultdict(  # sampled_id
#                     lambda: []  # [ser]
#                 ))))

#     for _type in inp:
#         for _id in inp[_type]:
#             subgraph_data[_type][_id] = sample_subgraph_v4(graph, sampled_number, sampled_depth, inp={'item': [_id]})



     
#     return subgraph_data


# def sample_subgraphxx(graph, sampled_number = 8, sampled_depth=2, inp = None):
#     '''
#         Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
#         We maintain budgets for each node type, indexed by <node_id, time>.
#         Currently sampled nodes are stored in layer_data.
#         After nodes are sampled, we construct the sampled adjacancy matrix.
#     '''
#     layer_data  = defaultdict( #target_type
#                         lambda: {} # {target_id: [ser]}
#                     )
#     budget     = defaultdict( #source_type
#                                     lambda: defaultdict(  #source_id
#                                         lambda: [0.] #[sampled_score]
#                             ))
#     new_layer_adj  = defaultdict( #target_type
#                                     lambda: defaultdict(  #source_type
#                                         lambda: defaultdict(  #relation_type
#                                             lambda: [] #[target_id, source_id]
#                                 )))
#     '''
#         For each node being sampled, we find out all its neighborhood, 
#         adding the degree count of these nodes in the budget.
#         Note that there exist some nodes that have many neighborhoods
#         (such as fields, venues), for those case, we only consider 
#     '''
#     def add_budget(te, target_id, layer_data, budget):
#         for source_type in te:
#             tes = te[source_type]
#             for relation_type in tes:
#                 if relation_type == 'self' or target_id not in tes[relation_type]:
#                     continue
#                 adl = tes[relation_type][target_id]
#                 if len(adl) < sampled_number:
#                     sampled_ids = list(adl.keys())
#                 else:
#                     sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
#                 for source_id in sampled_ids:
#                     if source_id in layer_data[source_type]:
#                         continue
#                     budget[source_type][source_id][0] += 1. / len(sampled_ids)


#     '''
#         First adding the sampled nodes then updating budget.
#     '''
#     for _type in inp:
#         for _id in inp[_type]:
#             layer_data[_type][_id] = [len(layer_data[_type])]
#     for _type in inp:
#         te = graph.edge_list[_type]
#         for _id in inp[_type]:
#             add_budget(te, _id, layer_data, budget)
#     '''
#         We recursively expand the sampled graph by sampled_depth.
#         Each time we sample a fixed number of nodes for each budget,
#         based on the accumulated degree.
#     '''
#     count = 1
#     layer_sample = int(sampled_number/sampled_depth) + 1
#     while not satisfy(layer_data, sampled_number, 4):
#         sts = list(budget.keys())
#         for source_type in sts:
#             te = graph.edge_list[source_type]
#             keys  = np.array(list(budget[source_type].keys())) # number of sampled nodes of one type in budget
#             if layer_sample > len(keys):
#                 '''
#                     Directly sample all the nodes
#                 '''
#                 sampled_ids = np.arange(len(keys))
#             else:
#                 '''
#                     Sample based on accumulated degree
#                 '''
#                 score = np.array(list(budget[source_type].values()))[:,0] ** 2
#                 score = score / np.sum(score)
#                 sampled_ids = np.random.choice(len(score), layer_sample, p = score, replace = False) 
#             sampled_keys = keys[sampled_ids]
#             '''
#                 First adding the sampled nodes then updating budget.
#             '''
#             for k in sampled_keys:
#                 if count >= 4*sampled_number:
#                     break
#                 layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][0]] # id of node k in layer data not in orignal graph
#                 count += 1
#             for k in sampled_keys:
#                 add_budget(te, k, layer_data, budget)
#                 budget[source_type].pop(k)   # already added k into sampled nodes set
    

#     return layer_data
