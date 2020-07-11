# from pytorch_transformers import *

from data import *
# import gensim
# from gensim.models import Word2Vec
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook
from collections import defaultdict
import dill
import argparse

parser = argparse.ArgumentParser(description='Preprocess OAG (CS/Med/All) Data')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='citation-acm-v8',
                    help='CS, Medical or All: _CS or _Med or (empty)')
# {1: 5691, 0: 5437, 2: 3789})
args = parser.parse_args()
conf_ids = [22, 16, 779]
label_ids = [0, 1, 2, 2, 4, 5, 6]


paper_info = defaultdict(lambda: defaultdict())
with open(os.path.join(args.data_dir, '%s_paper_info.tsv'% args.domain), 'r') as fin:
    for line in fin:
        pi = line.strip().split('\t')
        paper_info[pi[0]] = {'id': pi[0], 'title':pi[1], 'type': 'paper', 'abstract': pi[2], 'year':pi[3]}
print('paper info finished')

author_info = defaultdict()
with open(os.path.join(args.data_dir, '%s_author.tsv'% args.domain), 'r') as fin:
     for line in fin:
        ai = line.strip().split('\t')
        # print(ai)
        author_info[ai[1]] = {'id':ai[1], 'name':ai[0], 'type': 'author'}
print('author finished')


conf_info = defaultdict()
with open(os.path.join(args.data_dir, '%s_conf.tsv'% args.domain), 'r') as fin:
    for line in fin:
        ci = line.strip().split('\t')
        conf_info[ci[1]] = {'id':ci[1], 'name':ci[0], 'type': 'venue'}
print('conf finished')

graph = Graph()
count = defaultdict(lambda:0)
with open(os.path.join(args.data_dir, '%s_paper_author.tsv'% args.domain), 'r') as fin:
    for line in fin:
        pa = line.strip().split('\t')
        graph.add_edge(author_info[pa[1]], paper_info[pa[0]], relation_type = 'PA')
        count['pa'] += 1

print('paper author finished')

count = defaultdict(lambda:0)
label_paper_dict = defaultdict(lambda:[])
with open(os.path.join(args.data_dir, '%s_paper_conf.tsv'% args.domain), 'r') as fin:
    for line in fin:
        pc = line.strip().split('\t')
        # print(pc[1])
        if int(pc[1]) not in conf_ids:
            graph.add_edge(paper_info[pc[0]], conf_info[pc[1]], relation_type = 'PV')
        else:
            ind = conf_ids.index(int(pc[1]))
            label_paper_dict[label_ids[ind]].append(pc[0])
            count[label_ids[ind]] += 1
print('paper conf finished')
print('count={}'.format(count))
dill.dump(label_paper_dict, open(args.output_dir + 'labels_%s.pk' % args.domain, 'wb'))
# print(args.output_dir + 'graph%s.pk' % args.domain)

with open(os.path.join(args.data_dir, '%s_paper_cite.tsv'% args.domain), 'r') as fin:
    for line in fin:
        pa = line.strip().split('\t')
        for p in pa[1:]:
            if p in paper_info:
                graph.add_edge(paper_info[pa[0]], paper_info[p], relation_type = 'PP_cite')
print('paper cite finished')


     
    
'''
    Calculate the total citation information as node attributes.
'''
    
for idx, pi in enumerate(graph.node_bacward['paper']):
    pi['citation'] = len(graph.edge_list['paper']['paper']['PP_cite'][idx])
for idx, ai in enumerate(graph.node_bacward['author']):
    citation = 0
    for rel in graph.edge_list['author']['paper'].keys():
        for pid in graph.edge_list['author']['paper'][rel][idx]:
            citation += graph.node_bacward['paper'][pid]['citation']
    ai['citation'] = citation
# for idx, fi in enumerate(graph.node_bacward['affiliation']):
#     citation = 0
#     for aid in graph.edge_list['affiliation']['author']['in'][idx]:
#         citation += graph.node_bacward['author'][aid]['citation']
#     fi['citation'] = citation
# for idx, vi in enumerate(graph.node_bacward['venue']):
#     citation = 0
#     for rel in graph.edge_list['venue']['paper'].keys():
#         for pid in graph.edge_list['venue']['paper'][rel][idx]:
#             citation += graph.node_bacward['paper'][pid]['citation']
#     vi['citation'] = citation
# for idx, fi in enumerate(graph.node_bacward['field']):
#     citation = 0
#     for rel in graph.edge_list['field']['paper'].keys():
#         for pid in graph.edge_list['field']['paper'][rel][idx]:
#             citation += graph.node_bacward['paper'][pid]['citation']
#     fi['citation'] = citation
    
    
    

# '''
#     Since only paper have w2v embedding, we simply propagate its
#     feature to other nodes by averaging neighborhoods.
#     Then, we construct the Dataframe for each node type.
# '''
# d = pd.DataFrame(graph.node_bacward['paper'])
# graph.node_feature = {'paper': d}
# cv = np.array(list(d['emb']))
# for _type in graph.node_bacward:
#     if _type not in ['paper', 'affiliation']:
#         d = pd.DataFrame(graph.node_bacward[_type])
#         i = []
#         for _rel in graph.edge_list[_type]['paper']:
#             for t in graph.edge_list[_type]['paper'][_rel]:
#                 for s in graph.edge_list[_type]['paper'][_rel][t]:
#                     if graph.edge_list[_type]['paper'][_rel][t][s] <= test_time_bar:
#                         i += [[t, s]]
#         if len(i) == 0:
#             continue
#         i = np.array(i).T
#         v = np.ones(i.shape[1])
#         m = normalize(sp.coo_matrix((v, i), \
#             shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['paper']))))
#         out = m.dot(cv)
#         d['emb'] = list(out)
#         graph.node_feature[_type] = d
# '''
#     Affiliation is not directly linked with Paper, so we average the author embedding.
# '''
# cv = np.array(list(graph.node_feature['author']['emb']))
# d = pd.DataFrame(graph.node_bacward['affiliation'])
# i = []
# for _rel in graph.edge_list['affiliation']['author']:
#     for j in graph.edge_list['affiliation']['author'][_rel]:
#         for t in graph.edge_list['affiliation']['author'][_rel][j]:
#             i += [[j, t]]
# i = np.array(i).T
# v = np.ones(i.shape[1])
# m = normalize(sp.coo_matrix((v, i), \
#     shape=(len(graph.node_bacward['affiliation']), len(graph.node_bacward['author']))))
# out = m.dot(cv)
# d['emb'] = list(out)
# graph.node_feature['affiliation'] = d           
      
    
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
dill.dump(graph, open(args.output_dir + '/graph_%s.pk' % args.domain, 'wb'))   