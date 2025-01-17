# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

import torch
import torch as t
from torch import nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
import multiprocessing as mp
import random
from models.BaseModel import GeneralModel

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class FuckGF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'selfloop_flag', 'batch', 'seed', 'latdim', 'gcn_layer', 'gt_layer', 'pnn_layer', ]
    # extra_log_args = [
    #     'emb_size', 'n_layers', 'selfloop_flag', 'ext', 'gtw', 'sub', 'ctra', 'b2', 'anchor_set_num', 'batch', 'seed', 'tstBat',
    #     'reg', 'ssl_reg', 'decay', 'save_path', 'latdim', 'head', 'gcn_layer', 'gt_layer', 'pnn_layer', 'load_model', 'data', 'tstEpoch', 'seedNum', 'maskDepth', 'fixSteps',
    #     'keepRate', 'keepRate2', 'reRate', 'addRate', 'addNoise', 'eps', 'approximate'
    # ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of GFormer layers.')
        parser.add_argument('--selfloop_flag', type=bool, default=False, help='Whether to add self-loop in adjacency matrix.')
        #parser.add_argument('--ext', type=float, default=0.5, help='Extension rate for decoder adjacency matrix.')
        #parser.add_argument('--gtw', type=float, default=0.1, help='Graph transformer weight.')
        #parser.add_argument('--sub', type=float, default=0.1, help='Sub matrix rate.')
        #parser.add_argument('--ctra', type=float, default=0.001, help='Contrastive regularizer.')
        #parser.add_argument('--b2', type=float, default=1, help='Learning rate for the second optimizer.')
        #parser.add_argument('--anchor_set_num', type=int, default=32, help='Number of anchor sets.')
        parser.add_argument('--batch', type=int, default=4096, help='Batch size.')
        parser.add_argument('--seed', type=int, default=500, help='Random seed.')
        #parser.add_argument('--tstBat', type=int, default=256, help='Number of users in a testing batch.')
        #parser.add_argument('--reg', type=float, default=1e-4, help='Weight decay regularizer.')
        #parser.add_argument('--ssl_reg', type=float, default=1, help='Contrastive regularizer.')
        #parser.add_argument('--decay', type=float, default=0.96, help='Weight decay rate.')
        #arser.add_argument('--save_path', type=str, default='tem', help='File name to save model and training record.')
        parser.add_argument('--latdim', type=int, default=64, help='Embedding size.')
        #parser.add_argument('--head', type=int, default=4, help='Number of heads in attention.')
        parser.add_argument('--gcn_layer', type=int, default=2, help='Number of GCN layers.')
        parser.add_argument('--gt_layer', type=int, default=1, help='Number of graph transformer layers.')
        parser.add_argument('--pnn_layer', type=int, default=1, help='Number of PNN layers.')
        #parser.add_argument('--load_model', type=str, default=None, help='Model name to load.')
        #parser.add_argument('--data', type=str, default='lastfm', help='Name of dataset.')
        #parser.add_argument('--tstEpoch', type=int, default=3, help='Number of epochs to test while training.')
        #parser.add_argument('--seedNum', type=int, default=9000, help='Number of seeds in patch masking.')
        #parser.add_argument('--maskDepth', type=int, default=2, help='Depth to mask.')
        #parser.add_argument('--fixSteps', type=int, default=10, help='Steps to train on the same sampled graph.')
        #parser.add_argument('--keepRate', type=float, default=0.9, help='Ratio of nodes to keep.')
        #parser.add_argument('--keepRate2', type=float, default=0.7, help='Ratio of nodes to keep.')
        #parser.add_argument('--reRate', type=float, default=0.8, help='Ratio of nodes to keep.')
        #parser.add_argument('--addRate', type=float, default=0.01, help='Ratio of nodes to keep.')
        #parser.add_argument('--addNoise', type=float, default=0.0, help='Ratio of nodes to keep.')
        #parser.add_argument('--eps', type=float, default=0.1, help='Scaled weight as reward.')
        #parser.add_argument('--approximate', type=int, default=-1, help='K-hop shortest path distance. -1 means exact shortest path.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.latdim = args.latdim
        self.norm_adj = self.build_adjmat(self.user_num, self.item_num, corpus.train_clicked_set, args.selfloop_flag)
        self.norm_adj_tensor = torch.tensor(self.norm_adj).float().to(self.device)  # 转换为适合的设备的张量
        self._define_params(args)
        self.apply(self.init_weights)

    def _define_params(self, args):
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(torch.empty(self.user_num, self.emb_size)),
            'item_emb': nn.Parameter(torch.empty(self.item_num, self.emb_size)),
        })
        nn.init.xavier_uniform_(self.embedding_dict['user_emb'])
        nn.init.xavier_uniform_(self.embedding_dict['item_emb'])
        self.layers = [self.emb_size] * self.n_layers

        self.gcn_layers = nn.Sequential(*[GCNLayer(self.emb_size, self.emb_size) for _ in range(args.gcn_layer)])
        self.gt_layers = nn.Sequential(*[GTLayer(self.emb_size) for _ in range(args.gt_layer)])
        self.pnn_layers = nn.Sequential(*[PNNLayer(self.emb_size, self.emb_size) for _ in range(args.pnn_layer)])

    def build_adjmat(self, user_count, item_count, train_mat, selfloop_flag=False):
        R = np.zeros((user_count, item_count), dtype=np.float64)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0

        adj_mat = np.zeros((user_count + item_count, user_count + item_count), dtype=np.float64)
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T

        if selfloop_flag:
            np.fill_diagonal(adj_mat, 1)

        rowsum = np.array(adj_mat.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)

        norm_adj_mat = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        return norm_adj_mat  # 返回稠密矩阵

    def forward(self, feed_dict, is_test=False):
        user, items = feed_dict['user_id'], feed_dict['item_id']
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        # 遍历每一层
        for gcn in self.gcn_layers:
            ego_embeddings = gcn(self.norm_adj_tensor, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        for gt in self.gt_layers:
            ego_embeddings = gt(self.norm_adj_tensor, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        for pnn in self.pnn_layers:
            ego_embeddings = pnn(ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # 确保 all_embeddings 中的所有元素都是张量
        all_embeddings = [x for x in all_embeddings if isinstance(x, torch.Tensor)] + \
                         [x[0] for x in all_embeddings if isinstance(x, tuple)]

        # 堆叠张量
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_embeddings = all_embeddings[:self.user_num, :][user, :]
        item_embeddings = all_embeddings[self.user_num:, :][items, :]

        prediction = (user_embeddings[:, None, :] * item_embeddings).sum(dim=-1)  # [batch_size, -1]
        u_v = user_embeddings.unsqueeze(1).expand(-1, items.shape[1], -1)  # 避免调用 repeat 的性能损耗

        if not is_test:
            # 构造正负样本
            pos_user_embeddings = self.data_augmentation(user_embeddings)
            neg_user_embeddings = self.random_sampling(user_embeddings.size(0), user_embeddings.size(1))

            # 计算对比损失
            contrastive_loss = self.contrastive_loss(user_embeddings, pos_user_embeddings, neg_user_embeddings)

            return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v': item_embeddings,
                    'contrastive_loss': contrastive_loss}
        else:
            return {'prediction': prediction.view(feed_dict['batch_size'], -1), 'u_v': u_v, 'i_v': item_embeddings}

    def data_augmentation(self, embeddings):
        # 示例：节点 dropout
        dropout_rate = 0.1
        mask = torch.rand_like(embeddings) > dropout_rate
        return embeddings * mask

    def random_sampling(self, num_samples, embedding_dim):
        # 随机采样负样本
        return torch.randn(num_samples, embedding_dim).to(self.device)

    def contrastive_loss(self, anchor, positive, negative):
        # 计算 InfoNCE 损失
        tau = 0.1
        sim_pos = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1) / tau
        sim_neg = torch.nn.functional.cosine_similarity(anchor.unsqueeze(1), negative, dim=-1) / tau
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.xavier_uniform_(m)

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, adj, embeds):
        support = torch.mm(embeds, self.weight)
        output = torch.spmm(adj, support)
        return output

class GTLayer(nn.Module):
    def __init__(self, latdim):
        super(GTLayer, self).__init__()
        self.latdim = latdim
        self.qTrans = nn.Parameter(init(t.empty(self.latdim, self.latdim)))
        self.kTrans = nn.Parameter(init(t.empty(self.latdim, self.latdim)))
        self.vTrans = nn.Parameter(init(t.empty(self.latdim, self.latdim)))

    def makeNoise(self, scores):
        noise = t.rand(scores.shape).to(scores.device)
        noise = -t.log(-t.log(noise))
        return scores + 0.01 * noise

    def forward(self, adj, embeds, flag=False):
        # 计算 Q, K, V
        qEmbeds = (embeds @ self.qTrans).view(embeds.shape[0], -1, self.latdim)
        kEmbeds = (embeds @ self.kTrans).view(embeds.shape[0], -1, self.latdim)
        vEmbeds = (embeds @ self.vTrans).view(embeds.shape[0], -1, self.latdim)

        # 计算注意力分数
        att = t.einsum('bhd, bhd -> bh', qEmbeds, kEmbeds)
        att = t.clamp(att, -10.0, 10.0)
        expAtt = t.exp(att)
        attNorm = expAtt / (expAtt.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算加权的 V
        resEmbeds = t.einsum('bh, bhd -> bd', attNorm, vEmbeds).view(-1, self.latdim)
        resEmbeds = torch.spmm(adj, resEmbeds)

        return resEmbeds, attNorm

class PNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PNNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, embeds):
        if isinstance(embeds, tuple):
            embeds = embeds[0]
        return self.linear(embeds)