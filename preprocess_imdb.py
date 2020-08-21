'''
    Preprocessing dataset
    Author: Lin Meng
    codes for averaging features to neighbors and reconstruct edges are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/blob/master/OAG/preprocess_OAG.py
'''
# from pytorch_transformers import *

from data import *
# import gensim
# from gensim.models import Word2Vec
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm   # Comment this line if using jupyter notebook
from collections import defaultdict
import dill
import argparse
import collections
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
spw = stopwords.words('english')

parser = argparse.ArgumentParser(description='Preprocess IMDB Data')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address to store the original data directory.')
parser.add_argument('--output_dir', type=str, default='./data/',
                    help='The address to output the preprocessed graph.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='imdb',
                    help='filename')

args = parser.parse_args()


df = pd.read_csv(os.path.join(args.data_dir, '%s/movie_metadata.csv'% args.domain))
print('# records', len(df.index))
'''
    load label action | comedy | Drama
'''
director_info = defaultdict()
movie_info = defaultdict()
actor_info = defaultdict()
all_genres = []
selected_genres = ['action', 'comedy', 'drama']
label_dict = defaultdict(lambda:[])
count  = {lambda:0}
genre_movie = defaultdict(lambda:[])
for ind in df.index:
    gs = df['genres'][ind]
    genre_list = df['genres'][ind].split('|')
    genre_list = [g.strip().lower() for g in genre_list]
    all_genres.extend(genre_list)
    
    di = df.director_name[ind]
    link = df.movie_imdb_link[ind]
    title = df.movie_title[ind]
    plots = df.plot_keywords[ind]
  
    # remove repeated movies and non-feature movies
    if link not in movie_info and isinstance(plots, str):
        movie_info[link] = {'id': link, 'title': title, 'type':'movie'}
        if isinstance(di, str) and di not in director_info:
            # print(di)
            director_info[di] = {'id': len(director_info), 'name':di, 'type':'director'}
        for label in selected_genres:
            if label in genre_list:
                genre_movie[label].append(link)

    
idx = 0
for col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
    for ind in df.index:
        a = df[col][ind]
        mid = df.movie_imdb_link[ind]
        if mid in movie_info:
            if isinstance(a, str) and a not in actor_info:
                actor_info[a] = {'id': len(actor_info), 'name':a, 'type': 'actor'}
cnt = collections.Counter(all_genres)
print('all genres', len(set(all_genres)), cnt)
print('all movies',len(movie_info))
print('all director',len(director_info))
print('all actor_info',len(actor_info))

graph = Graph()

'''
    load edges movie-director & movie-actor
'''
count_edge = defaultdict(lambda:0)
for ind in df.index:
    mid = df.movie_imdb_link[ind]
    di = df.director_name[ind]
    if mid in movie_info and di in director_info:
        count_edge['md'] += 1
        graph.add_edge(movie_info[mid], director_info[di], relation_type = 'MD')
    
    for col in ['actor_1_name', 'actor_2_name', 'actor_3_name']:
        a = df[col][ind]
        if a in actor_info and mid in movie_info:
            count_edge['ma'] += 1
            graph.add_edge(movie_info[mid], actor_info[a], relation_type='MA')

'''
    movie feature bag-of-words on plots
'''
selected_keyword = []
movie_plots = defaultdict(lambda:[])
for ind in df.index:
    plots = df.plot_keywords[ind]
    mid = df.movie_imdb_link[ind]
    if isinstance(plots, str) and mid in movie_info:
        plot_list = plots.split('|')
        plot_list = [p.lower() for p in plot_list]
        mi_plots = []
        # selected_keyword.extend(plot_list)
        for ps in plot_list:
            p_list = ps.split(' ')
            mi_plots.extend([p for p in p_list if p not in spw])
            selected_keyword.extend([p for p in p_list if p not in spw])
        movie_plots[mid] = mi_plots


cnt = collections.Counter(selected_keyword)
keywords = []
for k, v in cnt.items():
    if v > 5:
        keywords.append(k)
    
# print('all keywords len={}, selected len={}'.format(len(selected_keyword), len(keywords)))
for mid in movie_plots:
    movie_plots[mid] = [w for w in movie_plots[mid] if w in keywords]
    

movie2feature = [movie_plots[x] for x in movie_plots]
        
ohe = MultiLabelBinarizer()
movie_feature = ohe.fit_transform(movie2feature).astype('float')

for i, mi in enumerate(movie_plots.keys()):
    movie_info[mi]['emb'] = movie_feature[i]

print('feature size', len(movie_feature[0]))
for mi in movie_info:
    if mi not in movie_plots:
        movie_info[mi]['emb'] = np.zeros(len(movie_feature[0]))
print('count edge', count_edge)
for _type in graph.node_forward:
    print(_type,'has nodes:', len(graph.node_forward[_type]))

'''
    Since only movie have w2v embedding, we simply propagate its
    feature to other nodes by averaging neighborhoods.
    Then, we construct the Dataframe for each node type.
'''
d = pd.DataFrame(graph.node_bacward['movie'])
graph.node_feature = {'movie': d}
cv = np.array(list(d['emb']))
for _type in graph.node_bacward:
    if _type not in ['movie']:
        d = pd.DataFrame(graph.node_bacward[_type])
        i = []
        for _rel in graph.edge_list[_type]['movie']:
            for t in graph.edge_list[_type]['movie'][_rel]:
                for s in graph.edge_list[_type]['movie'][_rel][t]:
                    i += [[t, s]]
        if len(i) == 0:
            continue
        i = np.array(i).T
        v = np.ones(i.shape[1])
        m = normalize(sp.coo_matrix((v, i), \
            shape=(len(graph.node_bacward[_type]), len(graph.node_bacward['movie']))))
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
# dill.dump(graph, open(args.output_dir + '/graph_%s.pk' % args.domain, 'wb'))   



label_target_dict = defaultdict(lambda:[])
count = defaultdict(lambda:0)
for ind in df.index:
    gs = df.genres[ind]
    mid = df.movie_imdb_link[ind]
    s = 0
    if mid in movie_info:
        genre_list = gs.strip().split('|')
        genre_list = [g.lower() for g in genre_list]
        tmp = set(selected_genres).intersection(set(genre_list))
        # print(tmp)
        
        if 'action' in tmp:
            s += 1
            label_target_dict[selected_genres.index('action')].append(graph.node_forward['movie'][mid])
            count['action'] += 1
            continue
        elif 'comedy' in tmp:
            s += 1
            label_target_dict[selected_genres.index('comedy')].append(graph.node_forward['movie'][mid])
            count['comedy'] += 1
            continue
        elif 'drama' in tmp:
            s += 1
            label_target_dict[selected_genres.index('drama')].append(graph.node_forward['movie'][mid])
            count['drama'] += 1

        # '''
        #    result in class imbalance
        # '''
        # if len(tmp) == 1:
        #     g = list(tmp)[0]
        #     label_target_dict[selected_genres.index(g)].append(graph.node_forward['movie'][mid])
        #     count[g] += 1
          
label_target_dict[selected_genres.index('comedy')] = np.random.choice(label_target_dict[selected_genres.index('comedy')], 1200,replace=False)
label_target_dict[selected_genres.index('drama')] = np.random.choice(label_target_dict[selected_genres.index('drama')], 1200,replace=False)

for i in label_target_dict:
    print('label={}, len={}'.format(i, len(label_target_dict[i])))

# print('movie label finished, count={}, total={}'.format(count, sum(count.values())))

# dill.dump(label_target_dict, open(args.output_dir + 'labels_%s.pk' % args.domain, 'wb'))

