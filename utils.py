'''
    Author: Lin Meng

    codes except feature_MAG and feature_HetLGN are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/tree/master/pyHGT
'''
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# shuffle mini-batch
def randint():
    return np.random.randint(2**10 - 1)
    # return 8


'''
    extract content features and structure features of sampled context graphs
    Input:
        subgraph data: all context graphs 
        graph: the original graph
        depth: searching depth
    output:
        feature dict for context graphs
'''
def feature_HetLGN(subgraph_data, graph, depth):
    feature = defaultdict( #target_id
                lambda: defaultdict(  #source_type
                            ))
  
    for target_type in subgraph_data:
        for target_id in subgraph_data[target_type]:
            for source_type in subgraph_data[target_type][target_id]:
                if len(subgraph_data[target_type][target_id][source_type]) == 0:
                    continue
                idxs = np.array(list(subgraph_data[target_type][target_id][source_type].keys()))
                struct_label = np.array(list(subgraph_data[target_type][target_id][source_type].values()))[:,1].reshape(-1,1)
                # print(struct_label.tolist())
                struct_feat = np.zeros((len(struct_label), depth))
                for i in range(len(struct_label)):
                    struct_feat[i,struct_label[i]] = 1 
                # print(struct_feat)
                content_feat = np.array(list(graph.node_feature[source_type].loc[idxs, 'emb']))
                feature[target_id][source_type] = np.concatenate([content_feat, struct_feat], axis=-1)
                # print('feature shape', feature[target_id][source_type].shape, content_feat.shape, struct_feat.shape)
                

    return feature

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def normalizex(adj):
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


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
    Feature extractor for MAG dataset
'''
def feature_MAG(subgraph_data, subfeature):
    feature = defaultdict( #target_id
                lambda: defaultdict(  #source_type
                            ))

    # texts   = []
    # indxs = defaultdict( #target_id
    #             lambda: defaultdict(  #source_type
    #                         ))
  
    for target_type in subgraph_data:
        for target_id in subgraph_data[target_type]:
            for source_type in subgraph_data[target_type][target_id]:
                if len(subgraph_data[target_type][target_id][source_type]) == 0:
                    continue
                idxs  = np.array(list(subgraph_data[target_type][target_id][source_type].keys()), dtype = np.int)
              
                feature[target_id][source_type] = subfeature[source_type][idxs]

                # indxs[target_id][source_type] = idxs

    return feature