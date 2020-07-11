import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import numpy as np
from itertools import permutations
from numpy import linalg as LA
from scipy.optimize import linear_sum_assignment
import math

'''
input [B, in, H, W] type [B, in, H] features [B, H, in_dim]
weight [out, in, H, W], type [out, in, K] 
'''
class dilated_iso_layer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, in_dim, out_dim, device, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(dilated_iso_layer, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device
        self.fc = nn.Linear(2*in_dim*kernel_size, out_dim)
        self.kernel, self.bias = self._init_weights()
        self.maxpoolk = nn.MaxPool3d((1, len(self.perms), 1), stride=(1, len(self.perms), 1))
        self.maxpoolf = nn.MaxPool2d((self.out_channel*self.in_channel,1), stride=(self.out_channel*self.in_channel,1))
        self.dropout = nn.Dropout(p=0.5)
        self.in_dim = in_dim

    def _init_weights(self):
        kernel = dict()
        bias = dict()
        if torch.cuda.is_available():
            kernel['struct'] = nn.Parameter(torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)).to(self.device))
            kernel['row_type'] = nn.Parameter(torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)).to(self.device))
            kernel['col_type'] = nn.Parameter(torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)).to(self.device))
            bias['struct'] = nn.Parameter(torch.zeros(self.out_channel*self.in_channel).to(self.device))
            bias['row_type'] = nn.Parameter(torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)).to(self.device))
            bias['col_type'] = nn.Parameter(torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)).to(self.device))
            self.perms = nn.Parameter(self._permutation(self.kernel_size).to(self.device), requires_grad=False)
            # self.fc.weight = self.fc.weight.to(self.device)
        else:
            kernel['struct'] = nn.Parameter(
                torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)))
            kernel['row_type'] = nn.Parameter(
                torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)))
            kernel['col_type'] = nn.Parameter(
                torch.zeros(size=(self.out_channel, self.in_channel, self.kernel_size, 1)))
            self.perms = nn.Parameter(self._permutation(self.kernel_size), requires_grad=False)

        nn.init.xavier_uniform_(kernel['col_type'].data, gain=1.414)
        nn.init.xavier_uniform_(kernel['struct'].data, gain=1.414)
        nn.init.xavier_uniform_(kernel['row_type'].data, gain=1.414)

        # nn.init.xavier_uniform_(self.fc.weight.data, gain=1.414)
        return kernel, bias


    def _permutation(self, k):
        n_P = np.math.factorial(k)
        perms = np.zeros([n_P, k, k])
        perm_list = permutations(range(k), k)

        count = 0
        for p in perm_list:
            for i in range(len(p)):
                perms[count, i, p[i]] = 1
            count += 1
        perms = torch.FloatTensor(perms).requires_grad_(False)
        # print('perms size={}'.format(perms.size()))
        return perms

    def _prepare_index(self, graph, types, features):
        pass

    def _compute_p(self, subgraphs, kernel):
        c, k, k = kernel.size()
        N, k, k = subgraphs.size() # N = B * n_subgraph
        VGs, UGs = LA.eig(subgraphs.detach().numpy())
        VHs, UHs = LA.eig(kernel.detach().numpy())

        bar_UGs = np.absolute(UGs).reshape(-1, 1, k, k)
        bar_UHs = np.absolute(UHs)

        P = np.matmul(bar_UGs, np.transpose(bar_UHs,(0,2,1)))
        P_star = torch.from_numpy(np.array(P)).requires_grad_(False)
        P_star = P_star.type(torch.FloatTensor)
        return P_star


    def _prepare(self, graph, types, features, d):
        g = graph
        _, _, n_H_prev, n_W_prev = g.size()
      
        subgraphs = []
        row_type_seq = []
        row_features = []
        col_type_seq = []
        col_features = []
        new_k = (self.kernel_size-1)*d + 1
        n_H = n_H_prev - new_k + 1
        n_W = n_W_prev - new_k + 1
        # print('n_h', n_H)
        for h in range(n_H):
            for w in range(n_W):
                h_idx = [h+i*d for i in range(self.kernel_size)]
                w_idx = [w+i*d for i in range(self.kernel_size)]

                x_slice = g[:, :, h_idx, :][:, :, :, w_idx]
                row_t_slice = types[:, :, h_idx]
                col_t_slice = types[:, :, w_idx]
                row_f_slice = features[:, h_idx, :]
                col_f_slice = features[:, w_idx, :]
                subgraphs.append(x_slice)
                row_type_seq.append(row_t_slice)
                col_type_seq.append(col_t_slice)
                row_features.append(torch.sum(row_f_slice, dim=-2))
                col_features.append(torch.sum(col_f_slice, dim=-2))

        subgraphs = torch.stack(subgraphs, dim=1) #[B, sub, in, k, k]
        row_type_seq = torch.stack(row_type_seq, dim=1) #[B, sub, in,  k]
        row_features = torch.stack(row_features, dim=1) #[B, sub, in_dim]
        col_type_seq = torch.stack(col_type_seq, dim=1)  # [B, sub, in,  k]
        col_features = torch.stack(col_features, dim=1)  # [B, sub, in_dim]
        # print('subgraphs={}, row_type_seq={}, row_features={}, col_type_seq={}, col_features={}'.format(subgraphs.size(), \
        #                                                         row_type_seq.size(), row_features.size(), col_type_seq.size(), col_features.size()))
        return subgraphs, row_type_seq, row_features, col_type_seq, col_features


    def _sub_feature(self, features, types):
        for sub_g in types:
            sub_g = sub_g.reshape(-1)
            for node_i in sub_g:
                continue
        pass

    def compute_p(self, subgraphs, kernel):
        c, k, k = kernel.size()
        N, k, k = subgraphs.size() # N = B * n_subgraph
        VGs, UGs = LA.eig(subgraphs.detach().numpy()) 
        VHs, UHs = LA.eig(kernel.detach().numpy())

        bar_UGs = np.absolute(UGs).reshape(-1, 1, k, k)
        bar_UHs = np.absolute(UHs)

        P = np.matmul(bar_UGs, np.transpose(bar_UHs,(0,2,1)))
        P_star = torch.from_numpy(np.array(P)).requires_grad_(False)
        P_star = P_star.type(torch.FloatTensor)
        return P_star
    

    def _fast_iso_with_type_constraint(self, sub_graphs, kernel, row_types, col_types):
        _, n_subgraph, _ , k, k = sub_graphs.size()
        # _, _, _, _ = sub_types.size()
        sub_graphs = sub_graphs.view(-1, 1, self.in_channel, self.kernel_size, self.kernel_size)
        P = self.compute_p(x, kernel)

    def _iso_layer_with_type_constraint(self, sub_graphs, kernel, row_types, col_types):
        # print('IsoLayer')
        _, n_subgraph, _ , k, k = sub_graphs.size()
        # _, _, _, _ = sub_types.size()
        sub_graphs = sub_graphs.view(-1, 1, self.in_channel, self.kernel_size, self.kernel_size)
        #[k!, k,k]*[out, in, 1,k,k] =
        # print('tmp size={}'.format(torch.matmul(torch.matmul(self.perms, kernel['struct'].unsqueeze(2)),
        #                    torch.transpose(self.perms, 2, 1)).size()))
        struct_mapping = torch.matmul(torch.matmul(self.perms, kernel['struct'].unsqueeze(2)),
                           torch.transpose(self.perms, 2, 1)).transpose(2, 1).view(-1, self.in_channel, self.kernel_size,
                                                               self.kernel_size) - sub_graphs  # [B*n_subgraph, 1, in, k, k] - [out*k!, in, k, k]
        struct_sim = torch.norm(struct_mapping, p='fro', dim=(-2, -1)) ** 2  # [B*sub, out*k!,in,k,k] = [B*sub, out*k!,in]
        #[k!,k,k]*[out,in, 1,  k, 1] = [out,in,k!, k,1]
        row_type_mapping = torch.matmul(self.perms,
                                        kernel['row_type'].unsqueeze(2)).transpose(2, 1).view(-1, self.in_channel, self.kernel_size,1) - row_types.view(-1,1, self.in_channel,
                                                                                 self.kernel_size,
                                                                                1)  # [B*subgraph, out*k!, in, k,1]
        col_type_mapping = torch.matmul(self.perms,
                                        kernel['col_type'].unsqueeze(2)).transpose(2, 1).view(-1, self.in_channel, self.kernel_size,1)  - col_types.view(-1, 1, self.in_channel,
                                                                                 self.kernel_size,
                                                                                 1)  # [B*subgraph, k!, in, k,1]
        # print('col_type_mapping size', col_type_mapping.size(), row_type_mapping.size())
        # print('2 struct_sim size', struct_sim.size())
        row_type_sim = torch.norm(row_type_mapping, p='fro', dim=(-2, -1)) ** 2          #[B*sub, out*k!, in]
        col_type_sim = torch.norm(col_type_mapping, p='fro', dim=(-2, -1)) ** 2  # [B*sub, out*k!, in]
        type_sim = torch.add(row_type_sim, col_type_sim)
        # # print('type_sim size', type_sim.size())
        sim = torch.add(struct_sim, type_sim)
        # print('1 structure size', sim.size())
        sim = -1 * sim.view(-1,n_subgraph, self.out_channel, self.perms.size()[0], self.in_channel) #[B, sub, out, k!,in]
        # print('2 sim size', sim.size())
        similarities = self.maxpoolk(sim).view(-1, n_subgraph, self.out_channel*self.in_channel) #+ self.bias['struct']#[B, sub, in*out] 
        # print('3 sim size', sim.size())
        similarities = (-1) * similarities.transpose(-2,-1) #[B, in*out, sub]
        # print('4 sim size', sim.size())
        return 1-F.softmax(similarities, dim=-1) # [B, in*out, sub]

    def forward(self, graph, types, features):
        sub_graphs, row_type_seq, row_features, col_type_seq, col_features = self._prepare(graph, types, features, self.dilation)
        features = row_features + col_features
        # sub_features = torch.cat([row_features, col_features], dim=-1) # concat on feature dim
        # print('sub features={}'.format(sub_features.size()))
        # features = self.fc(sub_features) #[B, sub, out_dim]

        sim = self._iso_layer_with_type_constraint(sub_graphs, self.kernel, row_type_seq, col_type_seq) # [B, in*out, sub]
        sim = sim/sim.size()[-1]
        # print('fetures size={}, sim size={}'.format(features.size(), sim.size()))
        # sim = self.maxpoolf(sim) #[B, 1, sub] keepdim
        # sim = torch.mean(sim, dim=1)[:, :64]
        # sim = torch.where(sim>0.8, sim, -9e15*torch.ones_like(sim))
        # sim = self.dropout(sim)
        print('mean fetures size={}, only sim size={}'.format(features.size(), sim.size()))
        # features = torch.matmul(sim, features) # [B, in*out, out_dim] sum respect to subgraphs
        # sorted_sim, indices = torch.sort(sim, dim=-1, descending=True)
        # print(' sorted sim', features.size(), indices.squeeze().size(), sorted_sim.size())
        # for b in range(features.size()[0]):
        #     features[b] = features[b][indices[b][0]]
        # selected_sorted_sim = sorted_sim[:, :64]
        h_mean = torch.mean(features[:,:,:], dim=-2) # [B,  in*out, out_dim] -> [B, out_dim]
        # h_max = self.maxpoolf(features)
        # print('features size={}, h_mean size={}, filter <0.95 sim={}'.format(features.size(), h_mean.size(), selected_sorted_sim.size()))
        return h_mean, sim

class gcn_layer(nn.Module):
    def __init__(self, in_hid, out_hid, device, dropout=0, bias=True):
        super(gcn_layer, self).__init__()
        self.in_hid = in_hid
        self.out_hid = out_hid
        self.dropout = dropout
        self.device = device
        
        self.is_bias = bias
        # glorot(self.weight[0])
        # glorot(self.weight[1])
        self.reset_parameters()


    def reset_parameters(self):
        self.weight = dict()
        self.bias = dict()
        
        if torch.cuda.is_available():
            for i in range(3):
                self.weight[i] = nn.Parameter(torch.FloatTensor(self.in_hid, self.out_hid).to(self.device))
                stdv = 1. / math.sqrt(self.weight[i].size(1))
                self.weight[i].data.uniform_(-stdv, stdv)
                if self.is_bias:
                    self.bias[i] = nn.Parameter(torch.FloatTensor(self.out_hid).to(self.device))
                    self.bias[i].data.uniform_(-stdv, stdv)
                else:
                    self.register_parameter('bias', None)


    '''
        adj [B, 1, n, n]
        input [B, n, nfeat]

    '''
    def forward(self, input, adj):
        adj = adj.squeeze() #[B, n, n]
        output = input
        layer = []
        print('in feature size={}'.format(input.size()))
        for i in range(3):
            support = torch.matmul(output, self.weight[i])
            output = torch.matmul(adj, support)
            if self.bias is not None:
                output = output + self.bias[i]
            layer.append(torch.sum(output, dim=-2))
        layer = torch.cat(layer, dim=-1)
        # output =  # [B, nfeat]
        print('3 gcn layer, mean node embeddings (SIZE={}) concatenate 3 layers'.format(layer.size()))
        return layer

    

        
# class GeneralConv(nn.Module):
#     def __init__(self, conv_name, in_hid, out_hid, num_types, num_relations, n_heads, dropout):
#         super(GeneralConv, self).__init__()
#         self.conv_name = conv_name
       
#         if self.conv_name == 'gcn':
#             self.base_conv = gcn_layer(in_hid, out_hid)
#         elif self.conv_name == 'gat':
#             self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

#     def forward(self, X, ):
      
#         if self.conv_name == 'gcn':
#             return self.base_conv(features, adj)
             
#         elif self.conv_name == 'gat':
#             return self.base_conv(meta_xs, edge_index)


