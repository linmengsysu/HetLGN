import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from collections import defaultdict

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



def feature_OAG(layer_data, graph):
    feature = {}
    # times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        # tims  = np.array(list(layer_data[_type].values()))[:,1]
        
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        # times[_type]   = tims
        indxs[_type]   = idxs
        
        # if _type == 'paper':##? didn't process in preprocess.py
        #     texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    # return feature, times, indxs, texts
    return feature, indxs, texts


def feature_IsoNode(subgraph_data, graph):
    feature = defaultdict( #target_id
                lambda: defaultdict(  #source_type
                            ))

    # times   = {}
    indxs = defaultdict( #target_id
                lambda: defaultdict(  #source_type
                            ))
    texts = []

    for target_type in subgraph_data:
        for target_id in subgraph_data[target_type]:
            for source_type in subgraph_data[target_type][target_id]:
                if len(subgraph_data[target_type][target_id][source_type]) == 0:
                    continue
                idxs = np.array(list(subgraph_data[target_type][target_id][source_type].keys()))

                feature[target_id][source_type] = np.array(list(graph.node_feature[source_type].loc[idxs, 'emb']))
                indxs[target_id][source_type] = idxs

    return feature, indxs, texts

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1