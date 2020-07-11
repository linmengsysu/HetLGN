import sys
import random
from data import *
from utils import *
# from model import *
from warnings import filterwarnings
from isonode import *
filterwarnings("ignore")
import dgl
import argparse
# import sample
# import randomexit
parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='PV',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=3,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='_acm_v2',
                    help='CS, Medicion or All: _CS or _Med or (empty)')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=64,
                    help='Number of hidden dimension')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=3,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=10,
                    help='How many nodes to be sampled per layer per type')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--data_percentage', type=int, default=1.0,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=80,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--repeat', type=int, default=1,
                    help='How many time to train over a singe batch (reuse data)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=int, default=None,
                    help='Gradient Norm Clipping')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate')
parser.add_argument('--kernel_size', nargs='?', default='[3,3,3,3]',
                    help='Learning Rate')              
parser.add_argument('--out_channel_number', nargs='?', default='[4,4,4,4]',
                    help='Learning Rate')
parser.add_argument('--in_channel_number', nargs='?', default='[1,1,1,1]',
                    help='Learning Rate')
parser.add_argument('--dilations', nargs='?', default='[4,3,2,1]',
                    help='Learning Rate')
parser.add_argument('--name', type=str, default='isonode',
                    help='Learning Rate')
                    


args = parser.parse_args()
params = {arg: getattr(args, arg) for arg in vars(args)}
print('parameters', params)
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
print('loaded graph from {}'.format(os.path.join(args.data_dir, 'graph%s.pk' % args.domain)))

graph = dill.load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.domain), 'rb'))

types = graph.get_types()
'''
    cand_list stores all the Journal, which is the classification domain.
'''
# cand_list = [0, 1, 9, 10, 13, 3, 4]
# cand_list = list(graph.edge_list['venue']['paper']['PV'].keys())
# print('cand_list={}'.format(len(cand_list)))

# '''
# before construct graph, split train:val:test = 7:1:2
# '''
train_pairs = {}
valid_pairs = {}
test_pairs = {}
np.random.seed(23)

cand_list = []
all_pairs = {}
count = 0
label_paper_dict = dill.load(open(os.path.join(args.data_dir, 'labels%s.pk' % args.domain), 'rb'))
for label in label_paper_dict:
    cand_list.append(label)
    np.random.shuffle(label_paper_dict[label])
    if label == 1:
        label_paper_dict[label] = np.sort(np.random.choice(label_paper_dict[label],994,replace=False))
    train_len = int(len(label_paper_dict[label]) * 0.8)
    val_len   = int(len(label_paper_dict[label]) * 0.9)
    # train_len = 200
    # val_len = 300
    for p in label_paper_dict[label][:train_len]:
        train_pairs[p] = label
    for p in label_paper_dict[label][train_len:val_len]:
        valid_pairs[p] = label
    for p in label_paper_dict[label][val_len:]:
        test_pairs[p] = label

# edge_hid_from_graph = {'paper': {'venue': hide_pairs}} #{target type: {source type: {target id, source_id} } }
print('train size = {}, val size = {}, test size = {}'.format(len(train_pairs), len(valid_pairs),
                                                                              len(test_pairs)))

def node_classification_sample(seed, pairs, batch_size):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        _ = pairs[target_id]
        target_info += [target_id]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, subgraph_data, _, _ = random_walk_restart(graph, \
                                               inp={'paper': np.array(target_info)}, \
                                               sampled_number=args.sample_width, sampled_depth=args.sample_depth)

    '''
        (3) Transform the subgraph into torch Tensor (edge_index is in format of pytorch_geometric)
    '''
    node_feature, node_type, adj, node_dict = to_torch_iso(feature, subgraph_data, graph, sample_number=args.sample_width*args.sample_depth*4)
    '''
        (4) Prepare the labels for each output target node (paper), and their index in sampled graph.
            (node_dict[type][0] stores the start index of a specific type of nodes)
    '''
    ylabel = torch.zeros(batch_size, dtype=torch.long)  # size (B+?, )
    for x_id, target_id in enumerate(target_ids):
        ylabel[x_id] = int(pairs[target_id])
    # print('node_feature={}'.format(node_feature.size()))
    return node_feature, node_type, adj, ylabel


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_pairs, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), valid_pairs, args.batch_size))
    jobs.append(p)
    return jobs

'''
    mini-batch on test
'''
def next_batch(pairs, target_list, start):
    end = start + args.batch_size
    if end < len(target_list):
        next_batch_ids = target_list[start: end]
        next_batch_pairs = {k: pairs[k] for k in next_batch_ids}
        return next_batch_pairs, end
    else:
        next_batch_ids = target_list[start:]
        next_batch_pairs = {k: pairs[k] for k in next_batch_ids}
        return next_batch_pairs, end


def test():
    with torch.no_grad():
        test_res = []
        y_true = []
        y_pred = []
        for _ in range(1):
            start = 0
            test_id_list = list(test_pairs.keys())
            while start < len(test_pairs):
                next_batch_pairs, start = next_batch(test_pairs, test_id_list, start)
                node_feature, node_type, adj, ylabel = \
                    node_classification_sample(randint(), next_batch_pairs, len(next_batch_pairs))
                paper_rep = gnn.forward(adj.to(device), node_type.to(device), \
                                        node_feature.to(device))
                res = classifier.forward(paper_rep)
                # print(res.size())
                y_pred.append(res)
                y_true.append(ylabel)
                for ai, bi in zip(ylabel, res.argsort(descending=True)):
                    test_res += [(bi == ai).int().tolist()]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        test_mrr = mean_reciprocal_rank(test_res)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        test_acc, test_f1_micro, test_f1_macro = score(y_pred, y_true)
        print('Last Test Acc:{} | Test Mirco F1: {} | Test Macro F1:{} | Test NDCG:{} | Test MRR: {}'.format(
            test_acc, test_f1_micro, test_f1_macro, np.average(test_ndcg), np.average(test_mrr)))

'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.
'''
criterion = nn.NLLLoss()

'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = HeteroIsoNode(name=args.name, in_channels=eval(args.in_channel_number), out_channels=eval(args.out_channel_number), kernel_sizes=eval(args.kernel_size), in_dim=len(graph.node_feature['paper']['emb'].values[0]), out_dim=args.n_hid, num_types=4, context_size=args.sample_width, device=device, dilation=eval(args.dilations)).to(device)
# classifier = Classifier((len(eval(args.in_channel_number)))*args.n_hid, len(cand_list)).to(device)
classifier = Classifier(4*64, len(cand_list)).to(device)
model = nn.Sequential(gnn, classifier)

if args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)

stats = []
res = []
best_val = 0
train_step = 1500

# pool = mp.Pool(args.n_pool)
st = time.time()
# jobs = prepare_data(pool)

for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    # print(jobs)
    train_data = node_classification_sample(randint(), train_pairs, args.batch_size)
    valid_data = node_classification_sample(randint(), valid_pairs, args.batch_size)
    # print('train_data', len(train_data))
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    # pool = mp.Pool(args.n_pool)
    # jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))


    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    for _ in range(args.repeat):
        node_feature, node_type, adj, ylabel = train_data
        node_rep = gnn.forward(adj.to(device), node_type.to(device), node_feature.to(device))
        res = classifier.forward(node_rep)
        loss = criterion(res, ylabel.to(device))

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        train_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            train_res += [(bi == ai).int().tolist()]
        train_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in train_res])
        train_mrr = mean_reciprocal_rank(train_res)
        train_acc, train_f1_micro, train_f1_macro = score(res, ylabel)
        print(("Epoch: {} {}s  LR: {} Train Loss: {}  Last train Acc:{} | train Mirco F1: {} | train Macro F1:{} | train NDCG:{} | train MRR: {}").format(\
                epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                train_acc, train_f1_micro, train_f1_macro, train_ndcg, np.average(train_mrr)))
        del res, loss
    # print('end of training')

    model.eval()
    with torch.no_grad():
        node_feature, node_type, adj, ylabel = valid_data
        node_rep = gnn.forward(adj.to(device), node_type.to(device), node_feature.to(device))
        res = classifier.forward(node_rep)
        loss = criterion(res, ylabel.to(device))

        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            valid_res += [(bi == ai).int().tolist()]
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
        valid_mrr = mean_reciprocal_rank(valid_res)
        valid_acc, valid_f1_micro, valid_f1_macro = score(res, ylabel)

        test()
       
        if valid_acc > best_val:
            best_val = valid_acc
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            print('UPDATE!!!')

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg))
        print(
            'Last valid Acc:{} | valid Mirco F1: {} | valid Macro F1:{} | valid NDCG:{} | valid MRR: {}'.format(
                valid_acc, valid_f1_micro, valid_f1_macro, valid_ndcg, np.average(valid_mrr)))

        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
    # print('end of validation')
    del train_data, valid_data


best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    y_true = []
    y_pred = []
    for _ in range(10):
        start = 0
        test_id_list = list(test_pairs.keys())
        while start < len(test_pairs):
            next_batch_pairs, start = next_batch(test_pairs, test_id_list, start)
            node_feature, node_type, adj, ylabel = \
                node_classification_sample(randint(), next_batch_pairs, len(next_batch_pairs))
            paper_rep = \
            gnn.forward(adj.to(device), node_type.to(device), node_feature.to(device))
            res = classifier.forward(paper_rep)
        
            y_pred.append(res)
            y_true.append(ylabel)
            for ai, bi in zip(ylabel, res.argsort(descending=True)):
                test_res += [(bi == ai).int().tolist()]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    test_mrr = mean_reciprocal_rank(test_res)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    test_acc, test_f1_micro, test_f1_macro = score(y_pred, y_true)
    print('Best Test Acc:{} | Test Mirco F1: {} | Test Macro F1:{} | Test NDCG:{} | Test MRR: {}'.format(test_acc,
                                                                                                         test_f1_micro,
                                                                                                         test_f1_macro,
                                                                                                         np.average(
                                                                                                             test_ndcg),
                                                                                                         np.average(
                                                                                                             test_mrr)))


