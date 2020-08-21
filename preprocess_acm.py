'''
    Preprocessing dataset
    Author: Lin Meng
    codes for averaging features to neighbors and reconstruct edges are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/blob/master/OAG/preprocess_OAG.py
'''

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
from data import *
from scipy import io as sio

import argparse

parser = argparse.ArgumentParser(description='Preprocess ACM Data')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--domain', type=str, default='acm_v3',
                    help='dataset filename')

args = parser.parse_args()

url = 'dataset/ACM.mat'
data_path = args.data_dir + '/ACM.mat'


conf_ids = [0, 1, 9, 10, 13]
label_ids = [0, 1, 2, 2, 1]

data = sio.loadmat(data_path)
p = data['P']
c = data['C']
a = data['A']
f = data['F']  # institute
l = data['L']  # field

p_vs_l = data['PvsL']  # paper-field
p_vs_a = data['PvsA']  # paper-author
p_vs_t = data['PvsT']  # paper-term, bag of words
p_vs_c = data['PvsC']  # paper-conference, labels come from that
a_vs_f = data['AvsF']  # author-institute
p_vs_p = data['PvsP']  # paper-paper
p_vs_c_filter = p_vs_c[:, conf_ids]
p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0] #paper index

p_vs_l = p_vs_l.tocoo()
p_vs_a = p_vs_a.tocoo()
p_vs_t = p_vs_t.tocoo()
c_vs_p = p_vs_c.transpose()
p_vs_c = p_vs_c.tocoo()
a_vs_f = a_vs_f.tocoo()
p_vs_p = p_vs_p.tocoo()
l_selected = (p_vs_l.sum(0) != 0).A1.nonzero()[0]
# print(type(p_vs_c_filter), p_selected, p_selected.shape, l_selected, p_vs_p.shape)


'''
get labels, labels are assigned according to the conference {label_id:[paper id]}
'''
label_paper_dict = defaultdict(lambda: [])
c_vs_p = c_vs_p[conf_ids].tocoo()
for i, j, v in zip(c_vs_p.row, c_vs_p.col, c_vs_p.data):
    label_paper_dict[label_ids[i]].append(j)

# dill.dump(label_paper_dict, open(args.output_dir + 'labels_%s.pk' % args.domain, 'wb'))


graph = Graph()
'''
    add node
'''
pfl = defaultdict(lambda: {})
count = 0
for i in range(len(p)):
    pi = {'id': i, 'abstract': p[i], 'type': 'paper'}
    pfl[count] = pi
    count += 1
    graph.add_node(pi)
print('pfl size={}'.format(len(pfl)))

count = 0
lfl = defaultdict(lambda: {})
for i in range(len(l)):
    # print('field', l[i])
    li = {'id': l[i][0][0], 'type': 'field'}
    lfl[count] = li
    count += 1
    graph.add_node(li)

count = 0
afl = defaultdict(lambda: {})
for i in range(len(a)):
    ai = {'id': a[i][0][0], 'type': 'author'}
    afl[count] = ai
    count += 1
    graph.add_node(ai)

'''
    add edges: 
'''
count_edge = defaultdict(lambda:0)
tmp = []
for i, j, v in zip(p_vs_l.row, p_vs_l.col, p_vs_l.data):
    tmp.append(i)
    graph.add_edge(lfl[j], pfl[i], time=None, relation_type='PF')
    count_edge['pf'] += 1
print('paper-field', len(set(tmp)))

tmp = []
for i, j, v in zip(p_vs_a.row, p_vs_a.col, p_vs_a.data):
    tmp.append(i)
    graph.add_edge(afl[j], pfl[i], time=None, relation_type='PA')
    count_edge['pa']+=1
print('author-paper', len(set(tmp)))
tmp = []
for i,j,v in zip(p_vs_p.row, p_vs_p.col, p_vs_p.data):
    count_edge['pp'] += 1
    graph.add_edge(pfl[i], pfl[j], time=None, relation_type='PP_cite')

# for i, j, v in zip(p_vs_c.row, p_vs_c.col, p_vs_c.data):
#     graph.add_edge(vfl[j], pfl[i], time=None, relation_type='PV')
#     count_edge['pa'] += 2

'''
    Calculate the total citation information as node attributes. Pass
'''

'''   
add node features (only for paper node)
bag-of-words
'''
feats = p_vs_t.toarray()
for i in range(len(pfl)):
    pfl[i]['emb'] = feats[i].tolist()
print('paper feature size', feats.shape)
print('count_edge', count_edge)
for _type in graph.node_forward:
    print(_type,'has nodes:', len(graph.node_forward[_type]))

'''
    Since only paper have bag-of-words embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['paper'])
graph.node_feature = {'paper': d}
cv = np.array(list(d['emb']))
for _type in graph.node_bacward:
    if _type not in ['paper']:
        d = pd.DataFrame(graph.node_bacward[_type])
        i = []
        for _rel in graph.edge_list[_type]['paper']:
            for t in graph.edge_list[_type]['paper'][_rel]:
                for s in graph.edge_list[_type]['paper'][_rel][t]:
                    i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
                                    shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['paper']))))
        out = m.dot(cv)
        d['emb'] = list(out)
        graph.node_feature[_type] = d


edg = {}
for k1 in graph.edge_list:
    if k1 not in edg:
        edg[k1] = {}
    for k2 in graph.edge_list[k1]:
        if k2 not in edg[k1]:
            edg[k1][k2] = {}
        for k3 in graph.edge_list[k1][k2]:
            if k3 not in edg[k1][k2]:
                edg[k1][k2][k3] = {}
            for e1 in graph.edge_list[k1][k2][k3]:
                if len(graph.edge_list[k1][k2][k3][e1]) == 0:
                    continue
                edg[k1][k2][k3][e1] = {}
                for e2 in graph.edge_list[k1][k2][k3][e1]:
                    edg[k1][k2][k3][e1][e2] = graph.edge_list[k1][k2][k3][e1][e2]
            print(k1, k2, k3, len(edg[k1][k2][k3]))

graph.edge_list = edg

del graph.node_bacward
# dill.dump(graph, open(args.output_dir + 'graph_%s.pk' % args.domain, 'wb'))
print(args.output_dir + 'graph_%s.pk' % args.domain)


