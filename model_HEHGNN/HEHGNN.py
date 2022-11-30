import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from .util_funcs import cos_sim, normalize
import scipy.sparse as sp
import tqdm


class HEHGNN_lp(nn.Module):
    """
    Decode neighbors of input graph.
    """

    def __init__(self,
                 num_ntype,
                 type_ind,
                 types,
                 nums_nodes,
                 dims_ori,
                 method,
                 conv_method,
                 dev,
                 cf):
        super(HEHGNN_lp, self).__init__()
        # # ! Init variables
        self.dev = dev
        self.method = method
        self.num_ntype, self.types, self.nums_nodes, self.num_sample, self.t = num_ntype, types, nums_nodes, cf['num_sample'], cf['t']
        self.t_id = type_ind
        self.non_linear = nn.ReLU()
        MD = nn.ModuleDict
        self.probs_agg, self.fgg_sim = MD({}), MD({})
        self.probs_agg = TypeAttentionLayer(sum(self.nums_nodes), cf['num_heads'], cf['attn_vec_dim'])
        # self.fgg_sim_all = GraphGenerator(cf['com_feat_dim'], 2, cf['sim_th'], self.dev)  # 2 head
        for t in types:
            self.fgg_sim[t] = GraphGenerator(cf['com_feat_dim'], 2, cf['sim_th'], self.dev)  # 2 head
            # self.fgg_sim[t] = GraphGenerator(dims_ori[0], 2, cf['sim_th'], self.dev)  # 2 head
        # Feature Encoder
        self.encoder = MD(dict(zip(types, [nn.Linear(dims_ori[i], cf['com_feat_dim']) for i in range(num_ntype)])))

        # ! Graph Convolution
        if conv_method == 'gcn':
            self.GNN = GCN(cf['com_feat_dim'], cf['gnn_emb_dim'], cf['gnn_fin_dim'], cf['dropout_rate'])
            # self.GNN = GCN(dims_ori[0], cf['gnn_emb_dim'], cf['gnn_fin_dim'], cf['dropout_rate'])
        elif conv_method == 'gat':
            # self.GNN = GAT(cf['com_feat_dim'], cf['gnn_emb_dim'], cf['gnn_fin_dim'], cf['dropout_rate'], 0.2, cf['num_heads'])
            self.GNN = GAT(dims_ori[0], cf['gnn_emb_dim'], cf['gnn_fin_dim'], cf['dropout_rate'], 0.2,
                           cf['num_heads'])

    def sample_gumbel(self, shape):
        """Sample from Gumbel(0, 1)"""
        U = np.random.uniform(size=shape)
        eps = np.finfo(U.dtype).eps
        r = -np.log(U + eps)
        r = -np.log(r + eps)
        return r

    # gumbel-softmax() sampling
    # n: time of sampling
    def gumbel_softmax_sample(self, prob, n, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        val = 0
        for i in tqdm.tqdm(range(n)):
            r = torch.from_numpy(self.sample_gumbel(prob.shape)).to(self.dev)
            # values = (torch.where(prob > 0, (torch.log(prob + eps) + r).type(torch.float32), prob))
            val += F.softmax(((prob + r) / temperature), dim=1)
        val = val / n
        val = val + val.T.multiply(val.T > val) - val.multiply(val.T > val)
        rel = val.cpu().detach().numpy()
        val = F.normalize(val + torch.eye(val.shape[0]).to(self.dev), dim=0, p=1).type(torch.float32)
        return val, rel

    # choice() sampling
    # n: num of samples
    def sampler(self, prob, n):
        print('Begin sampling !')
        adj = np.zeros((prob.shape[0], prob.shape[0]), dtype=numpy.float32)
        for i in tqdm.tqdm(range(prob.shape[0])):
            num = prob.shape[1]
            p = prob[i].cpu().detach().numpy()
            sampled_idx = np.sort(np.random.choice(num, n, replace=True, p=p))
            adj[i][sampled_idx] = 1

        print("Finish sampling !")
        # build symmetric adjacency matrix
        adj = adj + np.multiply(adj.T, (adj.T > adj)) - np.multiply(adj, (adj.T > adj))
        rel = adj
        adj_norm = normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.Tensor(adj_norm).to(self.dev)
        return adj, rel

    def agg(self, probs):
        new_prob = torch.cat(probs)
        new_prob_temp = new_prob.clone()
        new_prob = new_prob_temp - torch.diag_embed(torch.diag(new_prob))
        # new_prob = F.softmax(new_prob, dim=1)
        new_prob = F.log_softmax(new_prob, dim=1)
        return new_prob

    def forward(self, adj, features_list, pos_rel_batch, neg_rel_batch):
        def get_glo_rel(mat, t):
            return mat[self.t_id[t][0]:self.t_id[t][1], :]

        # ! Heterogeneous Feature Mapping
        com_feat_mat = torch.cat([   # self.non_linear(
            self.encoder[t](features_list[i]) for i,t in enumerate(self.types)])
        features = torch.cat(features_list)

        # Probability Matrix Generation
        ################################################
        if self.method == 'hehgnn':
            probs = []
            # sim = self.fgg_sim_all(com_feat_mat, com_feat_mat)
            # sim = F.softmax(sim, dim=0)
            for i, t in enumerate(self.types):
                mat_t = com_feat_mat[self.t_id[i][0]: self.t_id[i][1], :]
                sim_t = self.fgg_sim[t](mat_t, mat_t)
                probs.append(torch.spmm(sim_t, get_glo_rel(adj, i)))  # t类点与所有点为同质邻居的概率矩阵 (nt * n)  # 799
                # probs.append(torch.spmm(sim_t, F.normalize(get_glo_rel(adj, i), p=1, dim=0)))  # 74
                # probs.append(get_glo_rel(adj, i) + torch.spmm(sim_t, get_glo_rel(adj, i)))  # 790

            # Type-specific Attention
            new_prob = self.probs_agg(probs, self.nums_nodes, self.t_id).to(self.dev)
            # new_prob = self.agg(probs).to(self.dev)

            # Sampling
            new_adj, rel = self.gumbel_softmax_sample(new_prob, self.num_sample, self.t)
            # new_adj, rel = self.sampler(new_prob, self.num_sample)
        ############################################################

        # GNN baseline
        ######################################################
        else:
            adj = np.asarray(adj.cpu())
            adj_norm = normalize(adj + sp.eye(adj.shape[0]))
            rel = adj_norm
            new_adj = torch.Tensor(adj_norm).to(self.dev)
        ############################################################

        # GNN
        h = self.GNN(com_feat_mat, new_adj)
        # h = self.GNN(features, new_adj)

        # output
        pos_0_id, pos_1_id = list(pos_rel_batch[i][0] for i in range(len(pos_rel_batch))), \
                             list(pos_rel_batch[i][1] for i in range(len(pos_rel_batch)))
        neg_0_id, neg_1_id = list(neg_rel_batch[i][0] for i in range(len(neg_rel_batch))), \
                             list(neg_rel_batch[i][1] for i in range(len(neg_rel_batch)))
        return [h[pos_0_id, :], h[pos_1_id, :]], [h[neg_0_id, :], h[neg_1_id, :]], rel, new_adj, com_feat_mat


class TypeAttentionLayer(nn.Module):
    def __init__(self,
                 out_dim,
                 num_heads,
                 attn_vec_dim):
        super(TypeAttentionLayer, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_vec_dim = attn_vec_dim

        self.fc1 = nn.Linear(out_dim * num_heads, attn_vec_dim, bias=True)
        self.fc2 = nn.Linear(attn_vec_dim, 1, bias=False)

        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def my_normalize(self, a):
        temp = (torch.sum(a, dim=1) + 1e-8)
        a_d = torch.diag_embed(torch.pow(temp, -1))
        return torch.mm(a_d, a)

    def forward(self, probs, nums_nodes, t_id):
        alpha = []
        new_prob = torch.zeros((sum(nums_nodes), sum(nums_nodes)))
        for prob in probs:
            prob = prob.repeat(1, self.num_heads)
            fc1 = torch.tanh(self.fc1(prob))
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            alpha.append(fc2)
        alpha = torch.cat(alpha, dim=0)
        alpha = F.softmax(alpha, dim=0)
        # alpha = torch.unsqueeze(alpha, dim=-1)
        # alpha = torch.unsqueeze(alpha, dim=-1)
        # probs_l = torch.stack(probs)
        # new_prob = torch.sum(alpha * probs_l, dim=0)

        # renormalization for probs fusion
        bu, bs, bl, \
        ub, sb, lb = alpha[0]/(alpha[0]+alpha[1]), alpha[0]/(alpha[0]+alpha[2]), alpha[0]/(alpha[0]+alpha[3]), \
                     alpha[1]/(alpha[0]+alpha[1]), alpha[2]/(alpha[0]+alpha[2]), alpha[3]/(alpha[0]+alpha[3])
        new_prob[:t_id[0][1], t_id[1][0]:t_id[1][1]] = bu * probs[0][:, t_id[1][0]:t_id[1][1]] + torch.t(
            ub * probs[1][:, :t_id[0][1]])
        new_prob[:t_id[0][1], t_id[2][0]:t_id[2][1]] = bs * probs[0][:, t_id[2][0]:t_id[2][1]] + torch.t(
            sb * probs[2][:, :t_id[0][1]])
        new_prob[:t_id[0][1], t_id[3][0]:t_id[3][1]] = bs * probs[0][:, t_id[3][0]:t_id[3][1]] + torch.t(
            sb * probs[3][:, :t_id[0][1]])


        new_prob = new_prob + new_prob.t()
        # new_prob[:nums_nodes[0], :nums_nodes[0]] = probs[0][:, nums_nodes[0]]
        new_prob = new_prob - torch.diag_embed(torch.diag(new_prob))
        # new_prob = F.softmax(new_prob, dim=1)
        new_prob = self.my_normalize(new_prob)
        new_prob = torch.log(new_prob + 1e-8)
        # new_prob = F.log_softmax(new_prob, dim=1)
        return new_prob


class MetricCalcLayer(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, nhid))    # 1*16
        nn.init.xavier_uniform_(self.weight)   # 初始化

    def forward(self, h):
        return h * self.weight


class GraphGenerator(nn.Module):
    """
    Generate graph using similarity.
    """

    def __init__(self, dim, num_head=2, threshold=0.1, dev=None):
        super(GraphGenerator, self).__init__()
        self.threshold = threshold
        self.metric_layer = nn.ModuleList()
        for i in range(num_head):
            self.metric_layer.append(MetricCalcLayer(dim))
        self.num_head = num_head
        self.dev = dev

    def forward(self, left_h, right_h):
        """

        Args:
            left_h: left_node_num * hidden_dim/feat_dim
            right_h: right_node_num * hidden_dim/feat_dim
        Returns:

        """
        if torch.sum(left_h) == 0 or torch.sum(right_h) == 0:
            return torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        s = torch.zeros((left_h.shape[0], right_h.shape[0])).to(self.dev)
        zero_lines = torch.nonzero(torch.sum(left_h, 1) == 0)
        # The ReLU function will generate zero lines, which lead to the nan (devided by zero) problem.
        if len(zero_lines) > 0:
            left_h[zero_lines, :] += 1e-8
        for i in range(self.num_head):
            weighted_left_h = self.metric_layer[i](left_h)
            weighted_right_h = self.metric_layer[i](right_h)
            s += cos_sim(weighted_left_h, weighted_right_h)
        s /= self.num_head
        s = torch.where(s < self.threshold, torch.zeros_like(s), s)
        return s


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x


class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  # AHW
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GATLayer(nn.Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = nn.Parameter(torch.empty(size=(2 * output_feature, 1)))
        self.w = nn.Parameter(torch.empty(size=(input_feature, output_feature)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.w)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return F.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)


class GAT(nn.Module):
    """GAT模型"""

    def __init__(self, input_size, hidden_size, output_size, dropout, alpha, nheads, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention = [GATLayer(input_size, hidden_size, dropout=dropout, alpha=alpha, concat=True) for _ in
                          range(nheads)]
        for i, attention in enumerate(self.attention):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_size * nheads, output_size, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attention], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        # x = F.elu(self.out_att(x, adj))

        # return F.log_softmax(x, dim=1)
        return x