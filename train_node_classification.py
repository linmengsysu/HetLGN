'''
    Author: Lin Meng
    codes for preparing_data are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/tree/master/pyHGT
'''
import sys
import random
from torch_geometric.data import DataLoader
from data import *
from utils import *
# from model import *
from warnings import filterwarnings
from model import *
filterwarnings("ignore")
import argparse
# import sample
# import randomexit
parser = argparse.ArgumentParser(description='Training GNN on ACM classification task')
import datetime
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--cuda', type=int, default=3,
                    help='Avaiable GPU ID')
parser.add_argument('--dataset', type=str, default='_acm_v3',
                    help='dataset')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='gcn',
                    help='The name of GNN filter')
parser.add_argument('--n_hid', type=int, default=8,
                    help='Number of hidden dimension')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=int, default=0.5,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=3,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=6,
                    help='How many nodes to be sampled per layer')
parser.add_argument('--target_type', type=str, default='paper',
                    help='How many nodes to be sampled per layer')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--n_epoch', type=int, default=50,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=int, default=None,
                    help='Gradient Norm Clipping')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate')


                    


args = parser.parse_args()
params = {arg: getattr(args, arg) for arg in vars(args)}
print('parameters', params)
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
print('loaded graph from {}'.format(os.path.join(args.data_dir, 'graph%s.pk' % args.dataset)))

graph = dill.load(open(os.path.join(args.data_dir, 'graph%s.pk' % args.dataset), 'rb'))

types = graph.get_types()
'''
    cand_list stores labels, which is the classification domain.
'''
# '''
# before construct graph, split train:val:test = 8:1:1
# '''
train_pairs = {}
valid_pairs = {}
test_pairs = {}
np.random.seed(23)

cand_list = []
all_pairs = {}
count = 0
label_paper_dict = dill.load(open(os.path.join(args.data_dir, 'labels%s.pk' % args.dataset), 'rb'))
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

print('train size = {}, val size = {}, test size = {}'.format(len(train_pairs), len(valid_pairs),
                                                                              len(test_pairs)))

def node_classification_sample(seed, pairs, batch_size):
    '''
        (1) Sample batch_size number of output nodes (papers).
    '''
    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), batch_size, replace=False)
    target_info = []
    labels = defaultdict()
    for target_id in target_ids:
        target_info += [target_id]
        labels[target_id] = pairs[target_id]

    '''
        (2) Based on the seed nodes, sample a subgraph with 'sampled_depth' and 'sampled_number'
    '''
    feature, subgraph_data = random_walk_restart(graph, \
                                               inp={args.target_type: np.array(target_info)}, \
                                               sampled_number=args.sample_width, sampled_depth=args.sample_depth)

    '''
        (3) Transform the subgraph into torch_geometric graphs (edge_index is in format of pytorch_geometric)
    '''
    graph_data = to_torch_mp(feature, subgraph_data, graph, label=labels)

    return graph_data


def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_pairs, args.batch_size))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), valid_pairs, len(valid_pairs)))
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



'''
Use CrossEntropy (log-softmax + NLL) here, since each paper can be associated with one venue.
'''
criterion = nn.NLLLoss()
types = graph.get_types()
'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(n_layer=args.n_layers, depth=args.sample_depth, dropout=args.dropout, in_dim=len(graph.node_feature[args.target_type]['emb'].values[0])+args.sample_depth, out_dim=args.n_hid, num_types=len(types), device=device).to(device)
classifier = Classifier(args.n_layers*args.n_hid, len(cand_list)).to(device)
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
test_score_under_best_valid = []
pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)
log_dict = defaultdict(lambda:[])
for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    train_data = [g for sub in train_data for g in sub]
    # print(train_data)
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))

    single_epoch = defaultdict(lambda: [])
    model.train()
    train_losses = []
    torch.cuda.empty_cache()

    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    for batch in loader:
        # print(batch)
        batch = batch.to(device)
        node_rep = gnn.forward(batch, batch.node_type, batch.x)
        # print('node_rep.size()', node_rep.size())
        res = classifier.forward(node_rep)
        loss = criterion(res, batch.y)
        single_epoch['train_loss'].append(loss.cpu().detach())
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        train_res = []
        for ai, bi in zip(batch.y, res.argsort(descending=True)):
            train_res += [(bi == ai).int().tolist()]
        train_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in train_res])
        train_mrr = mean_reciprocal_rank(train_res)
        train_acc, train_f1_micro, train_f1_macro = score(res, batch.y)
        single_epoch['train_f1_micro'].append(train_f1_micro)
        single_epoch['train_f1_macro'].append(train_f1_macro)
        del res, loss
    print(("Epoch: {} {}s  LR: {} Train Loss: {}  Last train Acc:{} | train Mirco F1: {} | train Macro F1:{} | train NDCG:{} | train MRR: {}").format(\
                epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                train_acc, train_f1_micro, train_f1_macro, train_ndcg, np.average(train_mrr)))
    print('end of training')

    model.eval()
    with torch.no_grad():
        loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=True)
        for batch in loader:
            batch = batch.to(device)
            node_rep = gnn.forward(batch, None, None)
            res = classifier.forward(node_rep)
            loss = criterion(res, batch.y)

           
            valid_res = []
            for ai, bi in zip(batch.y, res.argsort(descending=True)):
                valid_res += [(bi == ai).int().tolist()]
            valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
            valid_mrr = mean_reciprocal_rank(valid_res)
            valid_acc, valid_f1_micro, valid_f1_macro = score(res, batch.y)
            single_epoch['val_loss'].append(loss.cpu().detach())
            single_epoch['valid_f1_micro'].append(valid_f1_micro)
            single_epoch['valid_f1_macro'].append(valid_f1_macro)
            

    with torch.no_grad():
        test_res = []
        y_true = []
        y_pred = []
        for _ in range(1):
            start = 0
            test_id_list = list(test_pairs.keys())
            while start < len(test_pairs):
                next_batch_pairs, start = next_batch(test_pairs, test_id_list, start)
                test_data = \
                    node_classification_sample(randint(), next_batch_pairs, len(next_batch_pairs))
                loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
                for batch in loader:
                    batch = batch.to(device)
                    paper_rep = gnn.forward(batch, None, None)
                    res = classifier.forward(paper_rep)
                    loss = criterion(res, batch.y)
                    single_epoch['test_loss'].append(loss.cpu().detach())
                    y_pred.append(res)
                    y_true.append(batch.y)
                    for ai, bi in zip(batch.y, res.argsort(descending=True)):
                        test_res += [(bi == ai).int().tolist()]
                del test_data
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        test_mrr = mean_reciprocal_rank(test_res)
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        test_acc, test_f1_micro, test_f1_macro = score(y_pred, y_true)
        print('Last Test Acc:{} | Test Mirco F1: {} | Test Macro F1:{} | Test NDCG:{} | Test MRR: {}'.format(
            test_acc, test_f1_micro, test_f1_macro, np.average(test_ndcg), np.average(test_mrr)))
        single_epoch['test_f1_micro'].append(test_f1_micro)
        single_epoch['test_f1_macro'].append(test_f1_macro)
       
        if valid_acc > best_val:
            best_val = valid_acc
            torch.save(model, os.path.join(args.model_dir, args.dataset + '_' + args.conv_name))
            print('Best score at epoch {} --> UPDATE!!!'.format(epoch))
            test_score_under_best_valid = [test_f1_micro, test_f1_macro]

        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
              (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
               loss.cpu().detach().tolist(), valid_ndcg))
        print(
            'Last valid Acc:{} | valid Mirco F1: {} | valid Macro F1:{} | valid NDCG:{} | valid MRR: {}'.format(
                valid_acc, valid_f1_micro, valid_f1_macro, valid_ndcg, np.average(valid_mrr)))

        stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
        del res, loss
   
    for _metric in single_epoch:
        tmp = sum(single_epoch[_metric])/len(single_epoch[_metric])
        log_dict[_metric].append(tmp)
    
    del train_data, valid_data

print('\nUnder model with best validation: Test F1-micro = {} | Test F1-macro={}\n'.format(test_score_under_best_valid[0], test_score_under_best_valid[1]))

print('---- Variance reduction under model selected by best validation score (run 10 times) -----')
best_model = torch.load(os.path.join(args.model_dir, args.dataset + '_' + args.conv_name))
best_model.train()
params = count_parameters(best_model)
print('number of parameters:{}'.format(params))
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
            test_data = \
                node_classification_sample(randint(), next_batch_pairs, len(next_batch_pairs))
            loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
            for batch in loader:
                batch = batch.to(device)
                paper_rep = \
                gnn.forward(batch, None, None)
                res = classifier.forward(paper_rep)

                y_pred.append(res)
                y_true.append(batch.y)
                for ai, bi in zip(batch.y, res.argsort(descending=True)):
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

'''
    save log dict
'''
dt = datetime.datetime.now()
suffix = '{}_{:02d}-{:02d}'.format(
    dt.date(), dt.hour, dt.minute)
dill.dump(log_dict, open('acm_log_dict_'+suffix, 'wb'))


'''
    store represenations and labels for all data points
'''
pairs = {**train_pairs, **valid_pairs, **test_pairs}
results = defaultdict(lambda:[])
gnn, classifier = best_model
with torch.no_grad():
    test_res = []
    y_true = []
    y_pred = []
    embeddings = []
    for _ in range(1):
        start = 0
        test_id_list = list(pairs.keys())
        while start < len(pairs):
            next_batch_pairs, start = next_batch(pairs, test_id_list, start)
            test_data = \
                node_classification_sample(randint(), next_batch_pairs, len(next_batch_pairs))
            loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
            for batch in loader:
                batch = batch.to(device)
                paper_rep = gnn.forward(batch, None, None)
                res = classifier.forward(paper_rep)
                embeddings.append(paper_rep.detach().cpu())
                y_pred.append(res)
                y_true.append(batch.y)
                for ai, bi in zip(batch.y, res.argsort(descending=True)):
                    test_res += [(bi == ai).int().tolist()]
    test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    test_mrr = mean_reciprocal_rank(test_res)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    results['embedding'] = torch.cat(embeddings, dim=0)
    results['y_pred'] = y_pred
    results['y_true'] = y_true

dill.dump(results, open('acm_result_dict.pk','wb'))