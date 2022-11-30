import pickle
import time
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.pytorchtools import EarlyStopping
from utils.tools import index_generator, cal_loss, write_nested_dict, seed_init, evaluate
from model_HEHGNN.HEHGNN import HEHGNN_lp
from model_HEHGNN import data

# yelp Params
num_ntype = 4
dropout_rate = 0.5
lr = 0.005
weight_decay = 0.001
nums_nodes = [2614, 1286, 4, 9]
type_ind = [[0, 2614], [2614, 3900], [3900, 3904], [3904, 3913]]
types = ['b', 'u', 's', 'l']
nums_edges = [21586, 3084, 6168]
true_num = nums_edges[2]

# method, gnn = 'gcn', 'gcn'
# method, gnn = 'gat', 'gat'
method, gnn = 'hehgnn', 'gcn'



def run(dataset, feats_type, alpha, sim_th, com_feat_dim, gnn_emb_dim, gnn_fin_dim, num_heads, attn_vec_dim,
                      dropout_rate, num_sample, t, num_epochs, early_stop, repeat, seed):
    if seed != -1:
        seed_init(seed)
    if method == 'hehgnn':
        res_file = f'results/{dataset}/hehgnn/sth{sim_th}_nsam{num_sample}_t{t}_{com_feat_dim}_{gnn_emb_dim}_{gnn_fin_dim}.txt '
    elif method == 'gcn':
        res_file = f'results/{dataset}/gcn/do{dropout_rate}_{com_feat_dim}_{gnn_emb_dim}_{gnn_fin_dim}.txt'
    else:
        res_file = f'results/{dataset}/gat/do{dropout_rate}_{com_feat_dim}_{gnn_emb_dim}_{gnn_fin_dim}.txt'

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    adj, features_list, dims_ori, type_mask, train_val_test_pos_rel, train_val_test_neg_rel = \
        data.load_data(dataset, feats_type, num_ntype, type_ind, device)

    adj_2 = torch.mm(adj,adj)
    adj_2 = adj_2 - torch.diag_embed(torch.diag(adj_2))
    L_adj = torch.diag_embed(torch.sum(adj_2, 0)) - adj_2

    train_pos_rel = train_val_test_pos_rel['train_pos']
    val_pos_rel = train_val_test_pos_rel['val_pos']
    test_pos_rel = train_val_test_pos_rel['test_pos']
    train_neg_rel = train_val_test_neg_rel['train_neg']
    val_neg_rel = train_val_test_neg_rel['val_neg']
    test_neg_rel = train_val_test_neg_rel['test_neg']

    y_true_train = np.array([1] * len(train_pos_rel) + [0] * len(train_pos_rel))
    y_true_val = np.array([1] * len(val_pos_rel) + [0] * len(val_neg_rel))
    y_true_test = np.array([1] * len(test_pos_rel) + [0] * len(test_neg_rel))

    auc_list = []
    ap_list = []
    cf = {'alpha': alpha, 'sim_th': sim_th, 'com_feat_dim': com_feat_dim, 'gnn_emb_dim': gnn_emb_dim,
          'gnn_fin_dim': gnn_fin_dim, 'num_heads': num_heads, 'attn_vec_dim': attn_vec_dim,
          'dropout_rate': dropout_rate, 'num_sample': num_sample, 't': t}
    for _ in range(repeat):
        net = HEHGNN_lp(num_ntype, type_ind, types, nums_nodes, dims_ori, method, gnn, device, cf)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=early_stop, verbose=True, save_path='checkpoint/checkpoint_{}.pt'
                                       .format(dataset))
        dur1 = []
        dur2 = []

        # 生成打乱或未打乱的train/val编号序列，编号对应train/val_pos_rel,即正确的边
        train_pos_idx_generator = index_generator(batch_size=nums_edges[0],
                                                  num_data=len(train_pos_rel))  # shuffle the train_id
        val_idx_generator = index_generator(batch_size=nums_edges[1], num_data=len(val_pos_rel),
                                            shuffle=False)  # not shuffle the val_id
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            pos_proba_list = []
            neg_proba_list = []
            # forward
            train_pos_idx = train_pos_idx_generator.next()
            train_pos_rel_ = train_pos_rel[train_pos_idx].tolist()
            train_neg_idx = np.random.choice(len(train_neg_rel), len(train_pos_idx))
            train_neg_rel_ = train_neg_rel[train_neg_idx].tolist()
            # train_neg_rel_ = train_neg_rel.tolist()

            t0 = time.time()

            [pos_embedding_start, pos_embedding_end], [neg_embedding_start, neg_embedding_end], rel, new_adj, com_feat_mat = net(
                adj, features_list, train_pos_rel_, train_neg_rel_)

            if method == 'hehgnn':
                l_reg = torch.linalg.matrix_rank(torch.mm(torch.mm(com_feat_mat.T, L_adj), com_feat_mat))
            else:
                l_reg = 0
            train_loss = cal_loss(pos_embedding_start, pos_embedding_end, neg_embedding_start,
                                  neg_embedding_end) + alpha * l_reg

            pos_embedding_start = pos_embedding_start.view(-1, 1, pos_embedding_start.shape[1])
            pos_embedding_end = pos_embedding_end.view(-1, pos_embedding_end.shape[1], 1)
            neg_embedding_start = neg_embedding_start.view(-1, 1, neg_embedding_start.shape[1])
            neg_embedding_end = neg_embedding_end.view(-1, neg_embedding_end.shape[1], 1)

            pos_out = torch.bmm(pos_embedding_start, pos_embedding_end).flatten()
            neg_out = torch.bmm(neg_embedding_start, neg_embedding_end).flatten()
            pos_proba_list.append(torch.sigmoid(pos_out))
            neg_proba_list.append(torch.sigmoid(neg_out))

            t1 = time.time()
            dur1.append(t1 - t0)  # forward time

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            t2 = time.time()
            dur2.append(t2 - t1)  # backward time

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().detach().numpy()
            train_auc = roc_auc_score(y_true_train, y_proba_test)
            # train_ap = average_precision_score(y_true_test, y_proba_test)

            # print training info
            print(
                'Epoch {:05d} | Train_Loss {:.4f} | train_AUC {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f}'.format(
                    epoch, train_loss.item(), train_auc, np.mean(dur1), np.mean(dur2)))

            # validation
            net.eval()
            pos_proba_list = []
            neg_proba_list = []
            val_loss = []
            with torch.no_grad():
                # forward
                val_idx = val_idx_generator.next()
                val_pos_rel_ = val_pos_rel[val_idx].tolist()
                val_neg_rel_ = val_neg_rel[val_idx].tolist()

                [pos_embedding_start, pos_embedding_end], [neg_embedding_start, neg_embedding_end], _, new_adj, com_feat_mat= net(
                    adj, features_list, val_pos_rel_, val_neg_rel_)

                if method == 'hehgnn':
                    l_reg = torch.linalg.matrix_rank(torch.mm(torch.mm(com_feat_mat.T, L_adj), com_feat_mat))
                else:
                    l_reg = 0
                v_loss = cal_loss(pos_embedding_start, pos_embedding_end, neg_embedding_start,
                                  neg_embedding_end) + alpha * l_reg
                val_loss.append(v_loss)

                pos_embedding_start = pos_embedding_start.view(-1, 1, pos_embedding_start.shape[1])
                pos_embedding_end = pos_embedding_end.view(-1, pos_embedding_end.shape[1], 1)
                neg_embedding_start = neg_embedding_start.view(-1, 1, neg_embedding_start.shape[1])
                neg_embedding_end = neg_embedding_end.view(-1, neg_embedding_end.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_start, pos_embedding_end).flatten()
                neg_out = torch.bmm(neg_embedding_start, neg_embedding_end).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

                y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
                y_proba_test = y_proba_test.cpu().numpy()
                val_auc = roc_auc_score(y_true_val, y_proba_test)
                val_loss = torch.mean(torch.tensor(val_loss))

            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Val_AUC {:.4f} | Epoch Time(s) {:.4f}'.format(
                epoch, val_loss.item(), val_auc, t_end - t_start))
            print('##########################################################################')
            # early stopping
            early_stopping(val_auc, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # test
        test_idx_generator = index_generator(batch_size=nums_edges[2], num_data=len(test_pos_rel), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(dataset)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            # forward
            test_idx = test_idx_generator.next()
            test_pos_rel_ = test_pos_rel[test_idx].tolist()
            test_neg_rel_ = test_neg_rel[test_idx].tolist()

            [pos_embedding_start, pos_embedding_end], [neg_embedding_start, neg_embedding_end], rel, new_adj, com_feat_mat = net(
                adj, features_list, test_pos_rel_, test_neg_rel_)

            pos_embedding_start = pos_embedding_start.view(-1, 1, pos_embedding_start.shape[1])
            pos_embedding_end = pos_embedding_end.view(-1, pos_embedding_end.shape[1], 1)
            neg_embedding_start = neg_embedding_start.view(-1, 1, neg_embedding_start.shape[1])
            neg_embedding_end = neg_embedding_end.view(-1, neg_embedding_end.shape[1], 1)

            pos_out = torch.bmm(pos_embedding_start, pos_embedding_end).flatten()
            neg_out = torch.bmm(neg_embedding_start, neg_embedding_end).flatten()
            pos_proba_list.append(torch.sigmoid(pos_out))
            neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()

        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)

        # adj_save = rel
        # print(adj_save.sum())
        # with open(f"data/adj_test_homophily/{dataset}_adj_{sim_th}.pkl", "wb") as tf:
        #     pickle.dump(adj_save, tf)
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))
    cf.update({'AUC_mean': np.mean(auc_list), 'AUC_std': np.std(auc_list),
               'AP_mean': np.mean(ap_list), 'AP_std': np.std(ap_list)})
    write_nested_dict(cf, res_file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='HE-HGNN testing for the Link Prediction')
    ap.add_argument('--feats-type', type=int, default=1,
                    help='Type of the node features used. ' +
                         '0 - generate features; ' +
                         '1 - load features.')
    ap.add_argument('--patience', type=int, default=50, help='the judge time of early-stop. Default is 5.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--dataset', type=str, default='yelp', help='LastFM/acm/dblp/imdb/yelp')

    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--seed', type=int, default=0, help='-1 means random')
    ap.add_argument('--com_feat_dim', type=int, default=32, help='dim of the first step (MLP) output')
    ap.add_argument('--gnn_emb_dim', type=int, default=16, help='gnn hidden layer dim')
    ap.add_argument('--gnn_fin_dim', type=int, default=8, help='gnn output dim')

    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--dropout_rate', type=float, default=0.5, help='drop-out rate')
    ap.add_argument('--attn-vec-dim', type=int, default=32, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--alpha', type=float, default=1, help='balance of l_reg')
    ap.add_argument('--sim_th', type=float, default=0.99, help='threshold of similarity graphs')
    ap.add_argument('--num_sample', type=int, default=10, help='')
    ap.add_argument('--t', type=float, default=0.4, help='temperature of gumbel-softmax()')
    args = ap.parse_args()

    run(args.dataset, args.feats_type, args.alpha, args.sim_th, args.com_feat_dim, args.gnn_emb_dim, args.gnn_fin_dim,
        args.num_heads, args.attn_vec_dim, args.dropout_rate, args.num_sample, args.t, args.epoch, args.patience,
        args.repeat, args.seed)