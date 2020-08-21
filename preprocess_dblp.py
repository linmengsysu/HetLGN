'''
    Preprocessing dataset
    Author: Lin Meng
    codes for averaging features to neighbors and reconstruct edges are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/blob/master/OAG/preprocess_OAG.py
'''
from data import *
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook
from collections import defaultdict
import dill
import argparse
import collections
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
spw = stopwords.words('english')
parser = argparse.ArgumentParser(description='Preprocess DBLP Data')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='DBLP_four_area',
                    help='dataset filename')

args = parser.parse_args()


conf_info = defaultdict()
with open(os.path.join(args.data_dir, '%s/conf.txt'% args.domain), 'r') as fin:
    for line in fin:
        ci = line.strip().split('\t')
        conf_info[ci[0]] = {'id':ci[0], 'name':ci[1], 'type': 'venue'}
print('conf finished')

author_info = defaultdict()
with open(os.path.join(args.data_dir, '%s/author_label.txt'% args.domain), 'r', encoding='utf-8') as fin:
    for line in fin:
        al = line.strip().split('\t')
        author_info[al[0]] = {}
print('label_target_dict author info finished')



with open(os.path.join(args.data_dir, '%s/author.txt'% args.domain), 'r',encoding='utf-8') as fin:
     for line in fin:
        ai = line.strip().split('\t')
        if ai[0] in author_info:
            author_info[ai[0]] = {'id':ai[0], 'name':ai[1], 'type': 'author'}
print('author finished', len(author_info))

paper_info = defaultdict(lambda: defaultdict())
with open(os.path.join(args.data_dir, '%s/paper.txt'% args.domain), 'r', encoding='latin1') as fin:
    for line in fin:
        pi = line.strip().split('\t')
        paper_info[pi[0]] = {'id': pi[0], 'title':pi[1], 'type': 'paper'}
print('paper info finished')


term_info = defaultdict()
with open(os.path.join(args.data_dir, '%s/term.txt'% args.domain), 'r', encoding='utf-8') as fin:
    for line in fin:
        ti = line.strip().split('\t')
        if ti[1] not in spw:
            term_info[ti[0]] = {'id':ti[0], 'word':ti[1], 'type': 'term'}


graph = Graph()
'''
    add egdes author-paper (AP), paper-venue (PV)
'''
count = defaultdict(lambda:0)
res = []
count_author_papars = defaultdict(lambda:[])
# count = defaultdict(lambda:0)
with open(os.path.join(args.data_dir, '%s/paper_author.txt'% args.domain), 'r', encoding='utf-8') as fin:
    for line in fin:
        pa = line.strip().split('\t')
        if pa[1] in author_info:
            graph.add_edge(author_info[pa[1]], paper_info[pa[0]], relation_type = 'PA')
            count['pa'] += 1
            res.append(pa[0])
            count_author_papars[pa[1]].append(pa[0])

paper_info = {x: paper_info[x] for x in res}        
# print('paper author finished, len paper info', len(paper_info))
# print('len of author={}'.format(len(count_author_papars)))
count_pa=[len(count_author_papars[x]) for x in count_author_papars]
cnt = collections.Counter(count_pa)



res = []
with open(os.path.join(args.data_dir, '%s/paper_conf.txt'% args.domain), 'r', encoding='utf-8') as fin:
    for line in fin:
        pc = line.strip().split('\t')
        if pc[0] in paper_info:
            graph.add_edge(paper_info[pc[0]], conf_info[pc[1]], relation_type = 'PV')
            count['PV'] += 1
            res.append(pc[0])
print('paper conf finished')

res = []
with open(os.path.join(args.data_dir, '%s/paper_term.txt'% args.domain), 'r',encoding='utf-8') as fin:
    for line in fin:
        pt = line.strip().split('\t')
        if pt[0] in paper_info and pt[1] in term_info:
            graph.add_edge(term_info[pt[1]], paper_info[pt[0]], relation_type = 'PT')
            res.append(pt[1])
            count['pt'] += 1

term_info = {x: term_info[x] for x in res}
print('paper term finished', len(term_info))
print('count={}'.format(count))

'''
    paper is directly connected to keywords, do bag-of-words; author feature: bag-of-words 
'''
selected_keyword = []
with open(os.path.join(args.data_dir, '%s/selected_term.txt'% args.domain), 'r',encoding='utf-8') as fin:
    for line in fin:
        w = line.strip()
        selected_keyword.append(w)
print('selected keywords len={}'.format(len(selected_keyword)))



paper_word = defaultdict(lambda:[])
with open(os.path.join(args.data_dir, '%s/paper_term.txt'% args.domain), 'r',encoding='utf-8') as fin:
    for line in fin:
        pt = line.strip().split('\t')
        if pt[0] in paper_info and pt[1] in term_info and term_info[pt[1]]['word'] in selected_keyword:
            paper_word[pt[0]].append(pt[1])

paper2feature = [paper_word[x] for x in paper_word]
        
ohe = MultiLabelBinarizer()
paper_feature = ohe.fit_transform(paper2feature).astype('float')

for i, paper in enumerate(paper_word.keys()):
    paper_info[paper]['emb'] = paper_feature[i]
    # print('features')
    
for paper in paper_info:
    if paper not in paper_word:
        paper_info[paper]['emb'] = np.zeros(334)

# print('count_edge', count_edge)
for _type in graph.node_forward:
    print(_type,'has nodes:', len(graph.node_forward[_type]))

'''
    Since only paper have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['paper'])
graph.node_feature = {'paper': d}
cv = np.array(list(d['emb']))
print('cv shape', cv.shape)
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
        print(type(m),m.shape)
        out = m.dot(cv)
        d['emb'] = list(np.array(out>0, dtype=np.int32))
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
# dill.dump(graph, open(args.output_dir + '/graph_%s.pk' % args.domain, 'wb'))   



label_target_dict = defaultdict(lambda:[])
with open(os.path.join(args.data_dir, '%s/author_label.txt'% args.domain), 'r', encoding='utf-8') as fin:
    for line in fin:
        al = line.strip().split('\t')
        label_target_dict[int(al[1])].append(graph.node_forward['author'][al[0]])
# print('label_target_dict author info finished')
# dill.dump(label_target_dict, open(args.output_dir + 'labels_%s.pk' % args.domain, 'wb'))

