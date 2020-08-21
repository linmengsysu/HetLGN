import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import dill
metrics=['recall@20','recall@40', 'recall@60', 'recall@80', 'recall@100', 'prec@20', 'prec@40', 'prec@60', 'prec@80', 'prec@100', 'hit@20', 'hit@40', 'hit@60', 'hit@80', 'hit@100', 'ndcg@20', 'ndcg@40', 'ndcg@60', 'ndcg@80', 'ndcg@100']
symbols=['o-', 'v-', 's-', 'p-', 'P-', '*-', 'H-', 'x-', 'X-','D-', 'd-', '1-', '2-', '3-', '4-', '8-', '^-', '<-', '>-', '.-', ',-', '+-', 'o--', 'o-.','o:']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'xkcd:sky blue', 'xkcd:coral', 'xkcd:gold', 'xkcd:indigo', 'xkcd:fuchsia', 'xkcd:crimson',  'xkcd:brown',  'xkcd:azure', 'xkcd:khaki',  'xkcd:navy',  'xkcd:tan',  'xkcd:salmon',  'xkcd:sienna',  'xkcd:yellowgreen',  'xkcd:teal']



def plot_loss_acc():
    data_dir = 'acm_log_dict_2020-08-07_02-47'
    data = dill.load(open(data_dir, 'rb'))
    # y = np.array(result)
    x = np.arange(1,len(data['train_loss'])+1)
    print('x.shape', x.shape)
    # print('y shape', y.shape)

    plt.plot()
    i = 0
    for _metric in data:
        if 'loss' in _metric:
            plt.plot(x, data[_metric], symbols[i]+'-', linewidth=1.0, color=colors[i], label=_metric)
            i += 1
    plt.xlim(1, len(x)+1)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.savefig('./plots/'+data_dir+'_loss.png')

    plt.clf()
    i = 0
    for _metric in data:
        if 'loss' not in _metric:
            plt.plot(x, data[_metric], symbols[i]+'-', linewidth=1.0, color=colors[i], label=_metric)
            i += 1
    plt.xlim(1, len(x)+1)
    plt.xlabel('Epoch', fontsize=14)
    plt.legend()
    plt.savefig('./plots/'+data_dir+'_score.png')

if 1:
    plot_loss_acc()