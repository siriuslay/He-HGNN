import pickle
import numpy as np
import scipy
import torch
import torch.nn.functional as F


def load_data(dataset, feats_type, num_ntype, type_ind, dev):
    prefix = 'data/preprocessed/{}_processed'.format(dataset)
    # in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    # adjlist00 = [line.strip() for line in in_file]
    # adjlist00 = adjlist00
    # in_file.close()
    # in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    # adjlist01 = [line.strip() for line in in_file]
    # adjlist01 = adjlist01
    # in_file.close()
    # in_file = open(prefix + '/0/0-0.adjlist', 'r')
    # adjlist02 = [line.strip() for line in in_file]
    # adjlist02 = adjlist02
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-1.adjlist', 'r')
    # adjlist10 = [line.strip() for line in in_file]
    # adjlist10 = adjlist10
    # in_file.close()
    # in_file = open(prefix + '/1/1-2-1.adjlist', 'r')
    # adjlist11 = [line.strip() for line in in_file]
    # adjlist11 = adjlist11
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-0-1.adjlist', 'r')
    # adjlist12 = [line.strip() for line in in_file]
    # adjlist12 = adjlist12
    # in_file.close()
    #
    # in_file = open(prefix + '/0/0-1-0_idx.pickle', 'rb')
    # idx00 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/0/0-1-2-1-0_idx.pickle', 'rb')
    # idx01 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/0/0-0_idx.pickle', 'rb')
    # idx02 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-1_idx.pickle', 'rb')
    # idx10 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-2-1_idx.pickle', 'rb')
    # idx11 = pickle.load(in_file)
    # in_file.close()
    # in_file = open(prefix + '/1/1-0-0-1_idx.pickle', 'rb')
    # idx12 = pickle.load(in_file)
    # in_file.close()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')

    adj = adjM.todense()
    # print(adj.sum())
    adj = np.int64(adj > 0)
    adj = torch.from_numpy(adj).type(torch.FloatTensor).to(dev)
    # print(adj.sum())
    # adj = np.asarray(adj)
    type_mask = np.load(prefix + '/node_types.npy')

    features_list = []
    dims_ori = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            dims_ori.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(dev))
    elif feats_type == 1:
        with open(prefix + '/node_features.pkl', 'rb') as f:
            features = pickle.load(f)
        f.close()
        for i in range(num_ntype):
            tmp = torch.from_numpy(features[type_ind[i][0]:type_ind[i][1], :])
            tmp = tmp.type(torch.float32)
            # tmp = F.normalize(tmp)
            dim = tmp.shape[1]
            dims_ori.append(dim)
            features_list.append(tmp.to(dev))

    with open(f"data/preprocessed/{dataset}_processed/train_val_test_pos_rel.pkl", "rb") as tf:
        train_val_test_pos_rel = pickle.load(tf)
        tf.close()
    with open(f"data/preprocessed/{dataset}_processed/train_val_test_neg_rel.pkl", "rb") as tf:
        train_val_test_neg_rel = pickle.load(tf)
        tf.close()
    # train_val_test_pos_user_artist = np.load(prefix + '/train_val_test_pos_rel.pkl')
    # train_val_test_neg_user_artist = np.load(prefix + '/train_val_test_neg_rel.pkl')

    return adj, features_list, dims_ori, type_mask, train_val_test_pos_rel, train_val_test_neg_rel
    # adj, feature_list, type_mask, train_val_test_pos_rel, train_val_test_neg_rel
