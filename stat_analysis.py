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
from data import sample_subgraph
from sample import *
import seaborn as sns

# import randomexit
parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address of preprocessed graph.')

parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_acm_v2',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
parser.add_argument('--sample_number', type=int, default=8,
                    help='How many nodes to be sampled per layer per type')

parser.add_argument('--sample_depth', type=int, default=3,
                    help='How many nodes to be sampled per layer per type')

parser.add_argument('--output_dir', type=str, default='dblp',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
parser.add_argument('--repeat', type=int, default=30,
                    help='How many times to be sampled')

random.seed(time.time())

args = parser.parse_args()
params = {arg: getattr(args, arg) for arg in vars(args)}
print('parameters', params)
     

def random_walk_bidirectional(graph, sampled_number=8, inp=None):
    subgraph_data = defaultdict(  # target type
        lambda: defaultdict(  # target_id
            lambda: defaultdict(  # sampled_type
                lambda: defaultdict(  # sampled_id
                    lambda: []  # [ser]
                ))))

    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id in inp[_type]:
            if _id in subgraph_data[_type]:
                print('repeated target id')
            subgraph_data[_type][_id][_type][_id] = [len(subgraph_data[_type][_id][_type])]
            forward_cur_node = [_type, _id]
            backward_cur_node = [_type, _id]
            count_stop = 0
            neighbor_L = 0
            while neighbor_L < sampled_number:
                # print('neighbor_L', neighbor_L)
                rand_p = random.random()
                if rand_p > 0.15:
                    forward_next_type = random.choice(list(graph.edge_list[forward_cur_node[0]].keys()))
                    for r in graph.edge_list[forward_cur_node[0]][forward_next_type]:
                        if forward_cur_node[1] not in graph.edge_list[forward_cur_node[0]][forward_next_type][r]:
                            forward_cur_node = [_type, _id]
                            # print('cur node type={}, node_id={}, next_type={}'.format(cur_node[0], cur_node[1], next_type))
                            count_stop += 1
                            break
                        forward_next_id = random.choice(list(graph.edge_list[forward_cur_node[0]][forward_next_type][r][forward_cur_node[1]].keys()))
                        # if next_id not in subgraph_data[_type][_id][next_type]:
                        neighbor_L += 1
                        cur_node = [forward_next_type, forward_next_id]
                        subgraph_data[_type][_id][forward_cur_node[0]][forward_cur_node[1]] = [
                            len(subgraph_data[_type][_id][forward_cur_node[0]])]
                     
                else:
                    forward_cur_node = [_type, _id]
                # print('neighbor_L={}'.format(neighbor_L))
                rand_p = random.random()
                if rand_p > 0.15:
                    backward_next_type = random.choice(list(graph.edge_list[backward_cur_node[0]].keys()))
                    for r in graph.edge_list[backward_next_type][backward_cur_node[0]]:
                        if backward_cur_node[1] not in graph.edge_list[backward_next_type][backward_cur_node[0]][r]:
                            backward_cur_node = [_type, _id]
                            # print('cur node type={}, node_id={}, next_type={}'.format(cur_node[0], cur_node[1], next_type))
                            count_stop += 1
                            continue
                        backward_next_id = random.choice(list(graph.edge_list[backward_next_type][backward_cur_node[0]][r][backward_cur_node[1]].keys()))
                        # if backward_next_id not in subgraph_data[_type][_id][backward_next_type]:
                        neighbor_L += 1
                        backward_cur_node = [backward_next_type, backward_next_id]
                        subgraph_data[_type][_id][backward_cur_node[0]][backward_cur_node[1]] = [
                            len(subgraph_data[_type][_id][backward_cur_node[0]])]
                            # if neighbor_L >= sampled_number:
                            #     break
                        # else:
                        #     backward_cur_node = [_type, _id]
                        #     count_stop += 1
                        #     # print('in')
                else:
                    backward_cur_node = [_type, _id]
            # print('neighbor_L={}'.format(neighbor_L))
    return subgraph_data


def random_walk_restart1(graph, sampled_number=8, inp=None):
    subgraph_data = defaultdict(  # target type
        lambda: defaultdict(  # target_id
            lambda: defaultdict(  # sampled_type
                lambda: defaultdict(  # sampled_id
                    lambda: []  # [ser]
                ))))

    '''
        First adding the sampled nodes then updating budget.
    '''
    for _type in inp:
        for _id in inp[_type]:
            if _id in subgraph_data[_type]:
                print('repeated target id')
            subgraph_data[_type][_id][_type][_id] = [len(subgraph_data[_type][_id][_type])]
            cur_node = [_type, _id]
            count_stop = 0
            neighbor_L = 0
            while neighbor_L < sampled_number:
                rand_p = random.random()
                if rand_p > 0.15:
                    next_type = random.choice(list(graph.edge_list[cur_node[0]].keys()))
                    for r in graph.edge_list[cur_node[0]][next_type]:
                        if cur_node[1] in graph.edge_list[cur_node[0]][next_type][r]:
                            cur_node = [_type, _id]
                            # print('cur node type={}, node_id={}, next_type={}'.format(cur_node[0], cur_node[1], next_type))
                            # count_stop += 1
                            # break
                            next_id = random.choice(list(graph.edge_list[cur_node[0]][next_type][r][cur_node[1]].keys()))
                            neighbor_L += 1
                            cur_node = [next_type, next_id]
                            subgraph_data[_type][_id][cur_node[0]][cur_node[1]] = [
                                len(subgraph_data[_type][_id][cur_node[0]])]
                            if neighbor_L >= sampled_number:
                                break
                            else:
                                cur_node = [_type, _id]
                                count_stop += 1
                                # print('in')
                else:
                    cur_node = [_type, _id]
            # print('neighbor_L={}'.format(neighbor_L))
    return subgraph_data


def degree_distribution(graph_nx, nodes, node_type='all'):
    # print(type(nodes), node_type, nodes)
    degree_sequence = sorted([graph_nx.degree(n) for n in nodes], reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    degreeCount = Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    percentage = sorted({deg[i]: cnt[i] / len(degree_sequence) for i in range(len(deg))}.items(), reverse=False)
    # print(percentage)
    cumulative = dict()
    cumulative[percentage[0][0]] = percentage[0][1]
    for i in range(1, len(percentage)):
        cumulative[percentage[i][0]] = percentage[i][1] + cumulative[percentage[i - 1][0]]

    print('{} nodes\' degree count cumulative percentage={}'.format(node_type, cumulative))
    # plt.loglog(deg, cnt, 'b-', marker='o')
    plt.plot(deg, cnt, 'o', color='b')
    plt.title("Degree rank plot")
    plt.ylabel("#degree")
    plt.xlabel("rank")
    plt.savefig('./data/{}/degree_distribution_{}.png'.format(args.output_dir, node_type))
    # plt.show()
    plt.clf()
    max_deg = max(degree_sequence)
    min_deg = min(degree_sequence)
    plt.hist(degree_sequence, bins=[i for i in range(min_deg, max_deg + 1, 10)], density=True)  # cumulative=True,
    plt.title("Degree Histogram")
    # plt.ylabel("")
    plt.xlabel("degrees")
    plt.savefig('./data/{}/degree_histogram_{}.png'.format(args.output_dir, node_type))
    plt.clf()

    percentage = dict(percentage)
    # --- dataset 1: just 4 values for 4 groups:
    df = pd.DataFrame(list(percentage.values()), index=list(percentage.keys()), columns=['x'])

    # make the plot
    fig = df.plot(kind='pie', subplots=True, figsize=(8, 8))[0].get_figure()
    fig.savefig('./data/{}/degree_count_pie_chart_{}.png'.format(args.output_dir, node_type))


def validate_subgraph_size(data, name='all'):
    # calculate all sampled size
    subgraph_size = defaultdict(lambda: [])
    for rwr in data:
        for target_id in rwr:
            subgraph_size[target_id].append(len(rwr[target_id]))

    # expectation of size for a target node
    aver_size = [sum(subgraph_size[tid]) / len(subgraph_size[tid]) for tid in subgraph_size]
    max_aver_size = int(max(aver_size)) + 1
    min_aver_size = int(min(aver_size)) - 1
    bins = [i for i in range(min_aver_size, max_aver_size)]
    cnt = [0] * len(bins)
    for s in aver_size:
        bin = int(s)
        ind = bins.index(bin)
        cnt[ind] += 1
    # averSizeCount = Counter(aver_size)
    # size, cnt = zip(*averSizeCount.items())
    percentage = sorted({bins[i]: cnt[i] / len(aver_size) for i in range(len(bins))}.items(), reverse=False)
    # print(percentage)
    cumulative = dict()
    cumulative[percentage[0][0]] = percentage[0][1]
    for i in range(1, len(percentage)):
        cumulative[percentage[i][0]] = percentage[i][1] + cumulative[percentage[i - 1][0]]
    print('random walk size count cumulative percentage={}'.format(cumulative))

    plt.bar(bins, cnt)
    plt.title("Unique Node Count in Random Walks")
    plt.ylabel("Quantity")
    plt.xlabel("Number of Unique Nodes")
    plt.savefig('./data/{}/aver_size_{}.png'.format(args.output_dir, name))
    # plt.show()
    plt.clf()
    percentage = dict(percentage)
    # --- dataset 1: just 4 values for 4 groups:
    df = pd.DataFrame(list(percentage.values()), index=list(percentage.keys()), columns=['x'])

    # make the plot
    fig = df.plot(kind='pie', subplots=True, figsize=(8, 8))[0].get_figure()
    fig.savefig('./data/{}/aver_size_pie_chart_{}.png'.format(args.output_dir, name))
    plt.clf()


def validate_distance(data, graph_nx, name='all'):
    distances = defaultdict(lambda: [])
    for rwr in data:
        for tid in rwr:
            dist = []
            for sid in rwr[tid]:
                dist.append(nx.shortest_path_length(graph_nx, tid, sid))
            distances[tid] = dist

    aver_distance = []
    for tid in distances:
        aver_distance.append([min(distances[tid]), max(distances[tid]), sum(distances[tid]) / len(distances[tid])])
    aver_distance = sorted(aver_distance, key=lambda dist: dist[2])
    aver_distance = np.array(aver_distance)
    # aver_distance = np.sort(aver_distance, axis=)
    fig, ax = plt.subplots()
    # # Dataset: for threshold???
    a = pd.DataFrame({'group': ['minimum size'] * len(aver_distance), 'value': aver_distance[:, 0]})
    b = pd.DataFrame({'group': ['maximum size'] * len(aver_distance), 'value': aver_distance[:, 1]})
    c = pd.DataFrame({'group': ['average size'] * len(aver_distance), 'value': aver_distance[:, 2]})
    # d = pd.DataFrame({'group': np.repeat('C', 20), 'value': np.random.normal(25, 4, 20)})
    # e = pd.DataFrame({'group': np.repeat('D', 100), 'value': np.random.uniform(12, size=100)})
    # df = a.append(b).append(c).append(d).append(e)
    df = a.append(b).append(c)

    # Start with a basic boxplot
    sns.boxplot(x="group", y="value", data=df)

    # Calculate number of obs per group & median to position labels
    medians = df.groupby(['group'])['value'].median().values
    nobs = df.groupby("group").size().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, ax.get_xticklabels()):
        plt.text(pos[tick], medians[tick] + 0.4, nobs[tick], horizontalalignment='center', size='medium', color='w',
                 weight='semibold')
    # add title
    plt.title("Boxplot with number of observation", loc="left")
    plt.savefig('./data/{}/aver_distance_{}.png'.format(args.output_dir, name))
    # plt.show()


def validate_local_density(data, graph_nx, threshold=True, name='all'):
    density = defaultdict(lambda: [])
    for rwr in data:
        for tid in rwr:
            sub = graph_nx.subgraph(rwr[tid])
            node_num = len(rwr[tid])
            sub_edges = len(list(sub.edges()))
            if node_num == 1:
                den = 0
            else:
                den = sub_edges / (node_num)
            density[tid].append(den)
    aver_density = []
    for tid in density:
        aver_density.append([min(density[tid]), max(density[tid]), sum(density[tid]) / len(density[tid])])
    aver_density = sorted(aver_density, key=lambda dist: dist[2])
    aver_density = np.array(aver_density)
    max_aver_density = int(max(aver_density[:, 2])) + 1
    min_aver_density = int(min(aver_density[:, 2]))
    bins = [i for i in range(min_aver_density, max_aver_density)]
    cnt = [0] * len(bins)
    for s in aver_density:
        bin = int(s[2])
        ind = bins.index(bin)
        cnt[ind] += 1
    percentage = sorted({bins[i]: cnt[i] / len(aver_density) for i in range(len(bins))}.items(), reverse=False)
    # print(percentage)
    cumulative = dict()
    cumulative[percentage[0][0]] = percentage[0][1]
    for i in range(1, len(percentage)):
        cumulative[percentage[i][0]] = percentage[i][1] + cumulative[percentage[i - 1][0]]
    print('random walk density count cumulative percentage={}'.format(cumulative))

    # fig, ax = plt.subplots()
    # # # Dataset: for threshold???
    # a = pd.DataFrame({'group': ['minimum size'] * len(aver_density), 'value': aver_density[:, 0]})
    # b = pd.DataFrame({'group': ['maximum size'] * len(aver_density), 'value': aver_density[:, 1]})
    # c = pd.DataFrame({'group': ['average size'] * len(aver_density), 'value': aver_density[:, 2]})
    # # d = pd.DataFrame({'group': np.repeat('C', 20), 'value': np.random.normal(25, 4, 20)})
    # # e = pd.DataFrame({'group': np.repeat('D', 100), 'value': np.random.uniform(12, size=100)})
    # # df = a.append(b).append(c).append(d).append(e)
    # df = a.append(b).append(c)

    # # Start with a basic boxplot
    # sns.boxplot(x="group", y="value", data=df)

    # # # Calculate number of obs per group & median to position labels
    # # medians = df.groupby(['group'])['value'].median().values
    # # nobs = df.groupby("group").size().values
    # # nobs = [str(x) for x in nobs.tolist()]
    # # nobs = ["n: " + i for i in nobs]
    # #
    # # # Add it to the plot
    # # pos = range(len(nobs))
    # # for tick, label in zip(pos, ax.get_xticklabels()):
    # #     plt.text(pos[tick], medians[tick] + 0.4, nobs[tick], horizontalalignment='center', size='small', color='w',
    # #              weight='semibold')
    # #     # plt.text(pos[tick], medians[tick], horizontalalignment='center', size='small', color='w',
    # #     #                   weight='semibold')

    # # add title
    # plt.title("Density of Graphs Generated by Random Walks", loc="left")
    # plt.savefig('./{}/aver_density_{}.png'.format(args.output_dir, name))
    # plt.clf()
    # percentage = dict(percentage)
    # # --- dataset 1: just 4 values for 4 groups:
    # df = pd.DataFrame(list(percentage.values()), index=list(percentage.keys()), columns=['x'])

    # # make the plot
    # fig = df.plot(kind='pie', subplots=True, figsize=(8, 8))[0].get_figure()
    # fig.savefig('./{}/aver_density_pie_chart_{}.png'.format(args.output_dir, name))
    # plt.clf()


def hop_analysis(data, graph_nx):
    hop_dis = defaultdict(lambda: []) #keyed by target id
    for rwr in data:
        for tid in rwr:
            for sid in rwr[tid]:
                d = nx.shortest_path_length(graph_nx, source=tid, target=sid)
                hop_dis[tid].append(d)
    
    all_hop_dis = np.array([hop_dis[tid] for tid in hop_dis]).reshape(-1)
    n, bins, patches = plt.hist(all_hop_dis, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('hops from target node')
    plt.ylabel('Probability')
    plt.xlim(min(all_hop_dis), max(all_hop_dis))
    plt.grid(True)
    plt.savefig('hist_of_hop_{}'.format('all'))
    plt.clf()


    

def neighborhood(G, node, k):
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=k)
    # return [node for node, length in path_lengths.items()
    #         if length == n]
    return path_lengths


def k_hop_degree_distribution(k_hop_degree, k, name):
    k_hop_degree = sorted(k_hop_degree, reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    max_k_hop_degree = int(max(k_hop_degree)) + 1
    min_k_hop_degree = int(min(k_hop_degree))
    bins = [i for i in range(min_k_hop_degree, max_k_hop_degree, 100)]
    cnt = [0] * len(bins)
    for s in k_hop_degree:
        bin = int(s)
        ind = bins.index(bin)
        cnt[ind] += 1
    # averSizeCount = Counter(aver_size)
    # size, cnt = zip(*averSizeCount.items())
    percentage = sorted({bins[i]: cnt[i] / len(k_hop_degree) for i in range(len(bins))}.items(), reverse=False)
    # print(percentage)
    cumulative = dict()
    cumulative[percentage[0][0]] = percentage[0][1]
    for i in range(1, len(percentage)):
        cumulative[percentage[i][0]] = percentage[i][1] + cumulative[percentage[i - 1][0]]
    print('{}: {}-hop degree count cumulative percentage={}'.format(name, k, cumulative))
    percentage = dict(percentage)
    # --- dataset 1: just 4 values for 4 groups:
    df = pd.DataFrame(list(percentage.values()), index=list(percentage.keys()), columns=['x'])

    # make the plot
    fig = df.plot(kind='pie', subplots=True, figsize=(8, 8))[0].get_figure()
    fig.savefig('./data/acm/{}_hop_degree_pie_chart_{}.png'.format(k, name))
    plt.clf()


def all_k_hop_degree_distribution(graph_nx, nodes, k, name='all'):
    all_k_hop_degree = defaultdict(lambda: [])
    for n in nodes:
        ds = neighborhood(graph_nx, n, k)
        for i in range(k):
            di = len([node for node, length in ds.items() if length == i+1])
            all_k_hop_degree[i+1].append(di)
    for n in all_k_hop_degree:
        k_hop_degree_distribution(all_k_hop_degree[n], n, name)

def type_distribution(sub_graph, type_dis):
    for t_type in sub_graph:
        for tid in sub_graph[t_type]:
            for s_type in sub_graph[t_type][tid]:
                type_dis[s_type][tid].append(len(set(sub_graph[t_type][tid][s_type])))
    return type_dis
            
def plot_type_his(type_dis):
  
    for _type in type_dis:
        # the histogram of the data
        type_avg = [sum(type_dis[_type][tid]) / len(type_dis[_type][tid]) for tid in type_dis[_type]]
        n, bins, patches = plt.hist(type_avg, density=True, facecolor='g', alpha=0.75)
        plt.xlabel('{} size'.format(_type))
        plt.ylabel('Probability')
        # plt.xlim(min(type_avg), max(type_dis[_type]))
        plt.grid(True)
        plt.savefig('hist_of_type_{}'.format(_type))
        plt.clf()

def to_nx_idx(sub_graph, index_dict):
    rwr = defaultdict(  # target id
        lambda: []  # [sample ids]
    )
    for t_type in sub_graph:
        for tid in sub_graph[t_type]:
            for s_type in sub_graph[t_type][tid]:
                # print('type={}, size={}'.format(s_type, len(sub_graph[t_type][tid])))

                for sid in sub_graph[t_type][tid][s_type]:
                    rwr[tid + index_dict[t_type]].append(sid + index_dict[s_type])
            rwr[tid + index_dict[t_type]] = list(set(rwr[tid + index_dict[t_type]]))
            print('total size={}'.format(len(rwr[tid + index_dict[t_type]])))

    return rwr


'''
    graph has nodes that is not in graph_nx due to preprocessed part?
'''
def to_nx(graph, index_dict):
    g = nx.DiGraph()
    for t_type in graph.edge_list:
        for s_type in graph.edge_list[t_type]:
            for r_type in graph.edge_list[t_type][s_type]:
                for tid in graph.edge_list[t_type][s_type][r_type]:
                    for sid in graph.edge_list[t_type][s_type][r_type][tid]:
                        # print('grapj nx', tid, sid)
                        tidx, sidx = index_dict[t_type] + tid, index_dict[s_type] + sid
                        g.add_edge(tidx, sidx)
    return g






graph = dill.load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
label_paper_dict = dill.load(open(os.path.join(args.data_dir, 'labels%s.pk' % args.domain), 'rb'))

node_index_dict = {}
node_num = 0
for _type in graph.node_forward:
    node_index_dict[_type] = node_num
    node_num += len(graph.node_forward[_type])
    print('node num', _type, node_num)



all_pairs = {}
# label_paper_dict = dill.load(open(os.path.join(args.data_dir, 'labels%s.pk' % args.domain), 'rb'))
for label in label_paper_dict:
    # np.random.shuffle(label_paper_dict[label])
    for p in label_paper_dict[label]:
        # if p in graph.node_forward['paper']:
            # node_id = graph.node_forward['paper'][p]
        all_pairs[p] = label
        # print('p', p)

graph_nx = to_nx(graph, node_index_dict)
all_rwr = []
# try:
#     subgraphs = dill.load(open(os.path.join(args.data_dir,
#                                             'rwr{}_sample_{}_repeat_{}.pk'.format(args.domain, args.sample_number,
#                                                                                   args.repeat)), 'rb'))
#     print('loaded from files')
#     for i in range(args.repeat):
#         rwr = to_nx_idx(subgraphs[i], node_index_dict)
#         all_rwr.append(rwr)
# except:
print('random walk on graph')
target_info = list(all_pairs.keys())
subgraphs = []
type_dis = defaultdict(  # type
        lambda:defaultdict(lambda: []  # [sample ids]
    ))
for i in range(args.repeat):
    subgraph_data = HGT_sample(graph, inp={'paper': np.array(target_info)}, \
                                        sampled_number=args.sample_number, sampled_depth=args.sample_depth)
    rwr = to_nx_idx(subgraph_data, node_index_dict)
    type_dis = type_distribution(subgraph_data, type_dis)
    all_rwr.append(rwr)
    subgraphs.append(subgraph_data)

dill.dump(subgraphs, open(args.data_dir+'/hgt_rwr{}_sample_{}_repeat_{}.pk'.format(args.domain, args.sample_number, args.repeat),'wb'))

print('all rwr len={}'.format(len(all_rwr)))
'''
    plot degree distribution over graph/specific type
'''
# degree_distribution(graph_nx, list(graph_nx.nodes()), 'all')
print('type analysis')
plot_type_his(type_dis)
print('hop analysis')
hop_analysis(all_rwr, graph_nx)
print('local density')
validate_local_density(all_rwr, graph_nx)
# start = 0
# for _type in graph.node_forward:
#     node_num = len(graph.node_forward[_type])
#     nodes = list(set([start + i for i in range(node_num)]).intersection(set(graph_nx.nodes())))
#     degree_distribution(graph_nx, nodes, _type)
#     start += node_num
#     if _type == 'paper':
#         all_k_hop_degree_distribution(graph_nx, nodes, 4, _type)
'''
    rwr -> {target: [sampled ids]}
    random walk size in histogram; 
    show the average/min/max distances in #repeat random walks
    cal the percentage under average random walk length 
'''
# validate_subgraph_size(all_rwr)
# validate_distance(all_rwr, graph_nx)
