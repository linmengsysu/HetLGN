'''
    The script cannot run due to the large quantity of training data
    Author: Lin Meng
    
    codes for Graph class are from HGT resiptory 
    reference: https://github.com/acbull/pyHGT/tree/master/pyHGT
    
    
'''


import sys
import random
from torch_geometric.data import DataLoader
from data import *
from utils import *
# from model import *
from warnings import filterwarnings
from isonode import *
from model import *
filterwarnings("ignore")
import argparse
# import sample
# import randomexit
parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')
import datetime


import torch
import torch.nn.functional as F


from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./datasets/OGB_MAG.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./model_save/hgt_4layer',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='MAG_PV',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--domain', type=str, default='',
                    help='CS, Medicion or All: _CS or _Med or (empty)')

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
parser.add_argument('--sample_depth', type=int, default=2,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=5,
                    help='How many nodes to be sampled per layer')

'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--n_epoch', type=int, default=100,
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

'''
    IsoNN arguments
'''
parser.add_argument('--kernel_size', nargs='?', default='[3,3,3]',
                    help='Kernel Size')              
parser.add_argument('--out_channel_number', nargs='?', default='[4,4,4]',
                    help='Out Channel Number')
parser.add_argument('--in_channel_number', nargs='?', default='[1,1,1]',
                    help='In Channel Number')
parser.add_argument('--dilations', nargs='?', default='[3,2,1]',
                    help='Dilations')

parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')


args = parser.parse_args()
params = {arg: getattr(args, arg) for arg in vars(args)}
print('parameters', params)
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")
# print('loaded graph from {}'.format(os.path.join(args.data_dir, '%s.pk' % args.domain)))

def ogbn_sample(seed, samp_nodes):
    np.random.seed(seed)
    ylabel      = graph.y[samp_nodes]
    ylabel      = {node: graph.y[node] for node in samp_nodes}
    # ======================================
    feature, subgraph_data, _, _ = random_walk_restart(graph, \
                                               inp={'paper': np.array(samp_nodes)}, \
                                               sampled_number=args.sample_width, sampled_depth=args.sample_depth, feature_extractor=feature_MAG)
    graph_data = to_torch_mp(feature, subgraph_data, graph, label=ylabel)
    # ======================================
    return graph_data
    
def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), \
                            np.random.choice(graph.train_paper, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'validation':
        for i in np.arange(n_batch):
            target_papers = graph.valid_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)

    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'variance_reduce':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs

graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
device = torch.device("cuda:%d" % args.cuda)
target_nodes = np.arange(len(graph.node_feature['paper']))
types = graph.get_types()
gnn = GNN(name=args.name, dropout=args.dropout, in_dim=len(graph.node_feature['paper'][0]), out_dim=args.n_hid, num_types=len(types), device=device)
classifier = Classifier(args.n_layers*args.n_hid, graph.y.max()+1)



model = nn.Sequential(gnn, classifier).to(device)
print('Model #Params: %d' % count_parameters(model))
criterion = nn.NLLLoss()


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
    ]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)

stats = []
res   = []
best_val   = 0
train_step = 0



log_dict = defaultdict(lambda:[])
for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    pool = mp.Pool(args.n_pool)
    st = time.time()
    jobs = prepare_data(pool)
    datas = [job.get() for job in jobs]
    train_data = [g for sub in datas for g in sub]
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    single_epoch = defaultdict(lambda: [])
    model.train()
    train_losses = []
    torch.cuda.empty_cache()
    # print('train_data={}'.format(train_data.shape))
    y_pred = []
    ylabel = []
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    for batch in loader:
        # print('batch.num_graphs',batch.num_graphs)
        batch = batch.to(device)
        node_rep = gnn.forward(batch, batch.node_type, batch.x)
        res = classifier.forward(node_rep)
        # print('res size={}, batch.y size={}'.format(res.size(), len(batch.y)))
        batch.y =  torch.LongTensor(batch.y).to(device)
        loss = criterion(res, batch.y)
        single_epoch['train_loss'].append(loss.cpu().detach())
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        y_pred.append(res.detach().cpu().argmax(dim=1))
        ylabel.append(batch.y.cpu())
        del res, loss
    # print('ylabel', torch.cat(ylabel,dim=-1).size())
    # print('y_pred', torch.cat(y_pred,dim=-1).size())
    train_acc  = evaluator.eval({'y_true': torch.cat(ylabel,dim=-1).view(-1,1), 'y_pred': torch.cat(y_pred,dim=-1).view(-1,1)})['acc']
    del loader, train_data
    print('end of training')

   
    print(("Epoch: {} {}s  LR: {} Train Loss: {}  Last train Acc:{}").format(\
                epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
                train_acc))
model.eval()
with torch.no_grad():
    y_pred = []
    ylabel = []
    pool.close()
    pool.join()
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool, task_type = 'sequential', s_idx = 0, n_batch = args.n_batch, batch_size=args.batch_size)
    with tqdm(np.arange(len(graph.test_paper) / args.n_batch // args.batch_size), desc='eval') as monitor:
        for s_idx in monitor:
            test_data = [job.get() for job in jobs]
            test_data = [g for sub in test_data for g in sub]
            pool.close()
            pool.join()
            pool = mp.Pool(args.n_pool)
            jobs = prepare_data(pool, task_type = 'sequential', s_idx = int(s_idx * args.n_batch), batch_size=args.batch_size)

            loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
            for batch in loader:
                # print('batch.num_graphs',batch.num_graphs)
                batch = batch.to(device)
                node_rep = gnn.forward(batch, batch.node_type, batch.x)
                res = classifier.forward(node_rep)
                # print('res size={}, batch.y size={}'.format(res.size(), len(batch.y)))
                batch.y =  torch.LongTensor(batch.y).to(device)
                loss = criterion(res, batch.y)
                single_epoch['test_loss'].append(loss.cpu().detach())
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                scheduler.step(train_step)
                y_pred.append(res.detach().cpu().argmax(dim=1))
                ylabel.append(batch.y.cpu())
                del res, loss
                
            test_acc = evaluator.eval({
                            'y_true': torch.cat(ylabel,dim=-1).view(-1,1),
                            'y_pred': torch.cat(y_pred,dim=-1).view(-1,1)
                        })['acc']
            print('test acc', test_acc)
            monitor.set_postfix(accuracy = test_acc)
            del batch

'''
    save log dict
'''
# dt = datetime.datetime.now()
# suffix = '{}_{:02d}-{:02d}'.format(
#     dt.date(), dt.hour, dt.minute)
# dill.dump(log_dict, open('mag_log_dict_'+suffix, 'wb'))

