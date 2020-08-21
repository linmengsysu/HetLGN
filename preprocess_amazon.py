'''
    Preprocessing dataset
    Author: Lin Meng
    codes for averaging features to neighbors and reconstruct edges are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/blob/master/OAG/preprocess_OAG.py
'''
import pandas as pd
import gzip
from transformers import *
import numpy as np
import math
from data import *
# import gensim

# from gensim.models import Word2Vec
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Preprocess OAG (CS/Med/All) Data')

'''
    Dataset arguments
'''
parser.add_argument('--input_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=1,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_amazon_sports',
                    help='CS, Medical or All: _CS or _Med or (empty)')
parser.add_argument('--citation_bar', type=int, default=1,
                    help='Only consider papers with citation larger than (2020 - year) * citation_bar')

args = parser.parse_args()

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('./data/meta_Sports_and_Outdoors.json.gz')
print(list(df.columns))
also_buy = defaultdict(lambda: 0)
also_view = defaultdict(lambda: 0)
for i in range(len(df['asin'])):
    # print(df['asin'][i], df['also_buy'][i], df['also_view'][i])
    if isinstance(df['also_buy'][i], list):
        also_buy[df['asin'][i]] += len(df['also_buy'][i])
    elif isinstance(df['also_buy'][i], float):
        continue

    if isinstance(df['also_view'][i], list):
        also_view[df['asin'][i]] += len(df['also_view'][i])
    elif isinstance(df['also_view'][i], float):
        continue
bound = 10
'''
    add nodes: products, categories, tech 1, brand or  (images)
'''
count = defaultdict(lambda:0)
pfl = defaultdict(lambda: {})
for i in range(len(df['asin'])):
    if isinstance(df['title'][i], float):
        continue
    if also_buy[df['asin'][i]] < bound or also_view[df['asin'][i]] < bound:
        continue
    print('id={}, also_buy len={}, also_view len={}'.format(df['asin'][i], also_buy[df['asin'][i]], also_view[df['asin'][i]]))
    # print('id={} title={}'.format(df['asin'][i], df['title'][i]))
    if isinstance(df['title'][i], str) and 'var aPageStart' not in df['title'][i]:
        # s = ''.join(filter(str.isalnum, df['title'][i]))
        pi = {'id':df['asin'][i], 'title': df['title'][i], 'type':'item'}
        pfl[df['asin'][i]] = pi
        count['item'] += 1

bfl = defaultdict(lambda: {})
cfl = defaultdict(lambda: {})
tfl = defaultdict(lambda: {})
count_brand = defaultdict(lambda:0)
for i in range(len(df['brand'])):
    if df['asin'][i] not in pfl:
        continue
    if not isinstance(df['brand'][i], float):
        bi = {'id': df['brand'][i], 'type':'brand', 'name':df['brand'][i]}
        bfl[df['brand'][i]] = bi
        count_brand[df['brand'][i]] += 1
    # count['brand'] += 1
    # print('tech2={}, cate={}, brand={}, main_cate={}'.format(df['tech2'][i], df['category'][i], df['brand'][i], df['main_cat'][i]))
    if isinstance(df['category'][i], list):
        for cate in df['category'][i]:
            ci = {'id': cate, 'type':'category', 'name': cate}
            cfl[cate] = ci
    # count['category'] += 1

count['brand'] = len(bfl)
# count['tech1'] = len(tfl)
count['category'] = len(cfl)
print('total node number={}, detail={}'.format(sum(count.values()), count))
count_brand = {k: v for k, v in sorted(count_brand.items(), key=lambda item: item[1], reverse=True)}
for k,v in count_brand.items():
    print('COUNT BRAND={}={}'.format(k,v))  
'''
    get product node embedding by XLNet
'''

if args.cuda != -1:
    print('XLNET GPU')
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
        



'''
item-brand (choose 3 brand?)
get labels, labels are assigned according to the conference {label_id:[paper id]}
'''
cand_list = ['Nike', 'Columbia', 'SHIMANO']
label_ids = [0,1,2]
label_item_dict = defaultdict(lambda: [])
for i in range(len(df['brand'])):
    if df['brand'][i] in cand_list and df['asin'][i] in pfl:
        label_item_dict[label_ids[cand_list.index(df['brand'][i])]].append(df['asin'][i])

dill.dump(label_item_dict, open(args.output_dir + 'labels%s.pk' % args.domain, 'wb'))
# print(args.output_dir + 'graph%s.pk' % args.domain)



'''
    add edges: also_buy, also_view, cate-product, brand-product, 
'''
graph = Graph()
rem = []
count_edge=defaultdict(lambda:0)
items = set()

for i in tqdm(range(len(df['asin'])), total = len(df['asin'])):
    # if l[0] not in pfl or l[4] != 'en' or 'emb' not in pfl[l[0]] or l[3] not in vfi_ids:
    if df['asin'][i] not in pfl: 
        continue
    if isinstance(df['category'][i], float):
        print('no cate category', df['asin'][i], df['category'][i])
        df['category'][i] = ['Sports & Outdoors']


    for cate in df['category'][i]:
        if cate not in cfl:
            continue 
        graph.add_edge(pfl[df['asin'][i]], cfl[cate], relation_type = 'PC_in')
        count_edge['PC_in']+=1

    if not isinstance(df['brand'][i], str):
        print('no brand'); continue
    if df['brand'][i] in cand_list:
        print('is target'); continue
    graph.add_edge(pfl[df['asin'][i]], bfl[df['brand'][i]], relation_type = 'PB_belong')
    count_edge['PB_belong'] += 1
    print('item', df['asin'][i], df['category'][i], df['brand'][i])


for i in tqdm(range(len(df['asin'])), total = len(df['asin'])):
    # if l[0] not in pfl or l[4] != 'en' or 'emb' not in pfl[l[0]] or l[3] not in vfi_ids:
    if df['asin'][i] not in pfl:
        continue
    # idx = list(np.where(df['asin']==id)[0])[0]
    print('item', df['asin'][i], df['also_buy'][i], df['also_view'][i])

    rem += [df['asin'][i]]
    for cob in df['also_buy'][i]:
        if cob not in pfl:
           continue 
        graph.add_edge(pfl[df['asin'][i]], pfl[cob], relation_type = 'PP_buy')
        items.add(df['asin'][i])
        items.add(cob)
        count_edge['PP_buy'] += 1

    for cov in df['also_view'][i]:
        if cov not in pfl:
           continue 
        graph.add_edge(pfl[df['asin'][i]], pfl[cov], relation_type = 'PP_view')
        items.add(df['asin'][i])
        items.add(cov)
        count_edge['PP_view'] += 1 

pfl = {i: pfl[i] for i in rem}
print('pfl len={}, items={}'.format(len(pfl), len(items)))


# for i in tqdm(range(len(df['also_view'])), total = len(df['also_view'])):
#     # if l[0] not in pfl or l[4] != 'en' or 'emb' not in pfl[l[0]] or l[3] not in vfi_ids:
#     if df['asin'][i] not in pfl:
#         continue
#     # print('papeer', l[0])
#     rem += [df['asin'][i]]
#     for cov in df['also_view'][i]:
#         if cov not in pfl:
#            continue 
#         graph.add_edge(pfl[df['asin'][i]], pfl[cov], relation_type = 'co-view')
#         count_edge['co-view']+=1

# pfl = {i: pfl[i] for i in rem}
# print(len(pfl))


print('total node number={}, detail={}'.format(sum(count.values()), count))
print('total edge number={}, detail={}'.format(sum(count_edge.values()), count_edge))


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased', output_hidden_states=True,
                                    output_attentions=True).to(device)
assert model.config.output_attentions == True
assert model.config.output_hidden_states == True

for id in tqdm(pfl, total = len(pfl)):
    try:
        input_ids = torch.tensor([tokenizer.encode(pfl[id]['title'])]).to(device)[:, :64]
        # if len(input_ids[0]) < 4:
        #     continue
        all_hidden_states, all_attentions = model(input_ids)[-2:]
        rep = (all_hidden_states[-2][0] * all_attentions[-2][0].mean(dim=0).mean(dim=0).view(-1, 1)).sum(dim=0)
        pfl[id]['emb'] = rep.tolist()
    except Exception as e:
        print(e)



'''
    Since only paper have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['item'])
graph.node_feature = {'item': d}
cv = np.array(list(d['emb']))
for _type in graph.node_bacward:
    if _type not in ['item']:
        d = pd.DataFrame(graph.node_bacward[_type])
        i = []
        for _rel in graph.edge_list[_type]['item']:
            for t in graph.edge_list[_type]['item'][_rel]:
                for s in graph.edge_list[_type]['item'][_rel][t]:
                    i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['item']))))
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
dill.dump(graph, open(args.output_dir + '/graph%s.pk' % args.domain, 'wb'))       


