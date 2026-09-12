#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : hgls
# @Software: PyCharm

import torch
import torch.nn as nn
from rrgcn import RecurrentRGCN
from src.FuzzyLogic import FuzzyLogicModule
from src.SAD import SeConvTransE, SeConvTransR
from src.hrgnn import HRGNN
from rgcn.utils import build_sub_graph
import torch.nn.functional as F
from decoder import ConvTransE, ConvTransR
from rrgcn import RGCNCell
from src.hrgnn import GNN
import math

class HGLS(nn.Module):
    # def __init__(self,args, graph, decoder,num_nodes, num_rels,mu, h_dim, task, relation_prediction, device,short=True, long=True, fuse='con',
    #              r_fuse='re',history_rate = 0.3, hidden_size=None,  ent_emb=None, rel_emb=None, static_graph=None,short_con=None, long_con=None):
    def __init__(self, args, graph, decoder, num_nodes, num_rels, h_dim, task, relation_prediction, device,
                 short=True, long=True, fuse='con',
                 r_fuse='re', history_rate=0.3, hidden_size=None, ent_emb=None, rel_emb=None,
                 static_graph=None, short_con=None, long_con=None):
        super(HGLS, self).__init__()

    # def __init__(self, args, decoder, num_nodes, num_rels, h_dim, task, relation_prediction, device,
    #              short=True, long=True, fuse='con',
    #              r_fuse='re', history_rate=0.3, hidden_size=None, ent_emb=None, rel_emb=None, static_graph=None,
    #              short_con=None, long_con=None):
    #     super(HGLS, self).__init__()
        self.g = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.h_dim = h_dim
        self.static_graph = static_graph
        self.sequence_len = short_con['sequence_len']
        self.task = task
        self.relation_prediction = relation_prediction
        self.short = short
        self.long = long
        self.fuse = fuse
        self.r_fuse = r_fuse
        self.history_rate = history_rate
        self.device = device
        self.use_static = short_con['use_static']
        # self.static_weight = short_con['static_weight']
        self.layer_norm = args['layer_norm']
        self.static_weight = args['weight']
        self.discount = short_con['discount']
        self.angle = short_con['angle']
        self.en_embedding = nn.Embedding(self.num_nodes, self.h_dim)
        self.rel_embedding = nn.Embedding(self.num_rels * 2+1, self.h_dim)
        torch.nn.init.normal_(self.en_embedding.weight)
        torch.nn.init.xavier_normal_(self.rel_embedding.weight)
        self.en_embedding_init = nn.Parameter(ent_emb.clone().detach().requires_grad_(True))
        self.rel_embedding_init = nn.Parameter(rel_emb.clone().detach().requires_grad_(True))
        # # xizneng0112
        # self.mu = nn.Embedding(mu, self.h_dim)
        # torch.nn.init.normal_(self.mu.weight)
        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.fuzzy = FuzzyLogicModule(feature_dim=self.h_dim)
        self.gnn = long_con['encoder']
        # GNN 初始化
        if self.gnn == 'regcn':
            self.rgcn = RGCNCell(num_nodes,
                                 h_dim,
                                 h_dim,
                                 num_rels * 2,
                                 short_con['num_bases'],
                                 short_con['num_basis'],
                                 long_con['a_layer_num'],
                                 short_con['dropout'],
                                 short_con['self_loop'],
                                 short_con['skip_connect'],
                                 short_con['encoder'],
                                 short_con['opn'])
        elif self.gnn == 'rgat':
            self.rgcn = GNN(self.h_dim, self.h_dim, layer_num=long_con['a_layer_num'], gnn=self.gnn, attn_drop=0.0, feat_drop=0.2)
        if self.short:
            self.model_r = RecurrentRGCN(num_ents=num_nodes, num_rels=num_rels, gnn=self.gnn,hidden_size=hidden_size,**short_con)
            self.model_r.rgcn = self.rgcn
            # self.model_r.dynamic_emb = self.en_embedding.weight
            # self.model_r.emb_rel = self.rel_embedding.weight
            self.model_r.dynamic_emb = self.en_embedding_init
            self.model_r.emb_rel = self.rel_embedding_init

        if self.long:
            self.model_t = HRGNN(graph=graph, num_nodes=num_nodes, num_rels=num_rels,hidden_size=hidden_size, **long_con)
            self.model_t.aggregator = self.rgcn
            self.model_t.en_embedding = self.en_embedding
            self.model_t.rel_embedding = self.rel_embedding
        if self.short and self.long:
            if self.fuse == 'con':
                self.linear_fuse = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.fuse == 'att':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
            elif self.fuse == 'att1':
                self.linear_l = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f = nn.Linear(self.h_dim, 1, bias=True)
            elif self.fuse == 'gate':
                self.gate = GatingMechanism(self.num_nodes, self.h_dim)
            else:
                print('no fuse function')
            if self.r_fuse == 'con':
                self.linear_fuse_r = nn.Linear(self.h_dim * 2, self.h_dim, bias=False)
            elif self.r_fuse == 'att1':
                self.linear_l_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.linear_s_r = nn.Linear(self.h_dim, self.h_dim, bias=True)
                self.fuse_f_r = nn.Linear(self.h_dim, 1, bias=True)
            elif self.r_fuse == 'gate':
                self.gate_r = GatingMechanism(self.num_rels *2 , self.h_dim)
            else:
                print('no fuse_r function')
        # self.loss_r = torch.nn.CrossEntropyLoss()
        # self.loss_e = torch.nn.CrossEntropyLoss()

        if decoder == "seconvtranse":
            self.decoder_ob1 = SeConvTransE(num_nodes, h_dim, hidden_size, short_con['input_dropout'], short_con['hidden_dropout'],
                                         short_con['feat_dropout'])
            self.decoder_ob2 = SeConvTransE(num_nodes, h_dim, hidden_size, short_con['input_dropout'], short_con['hidden_dropout'],
                                         short_con['feat_dropout'])
            self.rdecoder_re1 = SeConvTransR(num_rels, h_dim, hidden_size, short_con['input_dropout'], short_con['hidden_dropout'],
                                         short_con['feat_dropout'])
            self.rdecoder_re2 = SeConvTransR(num_rels, h_dim, hidden_size, short_con['input_dropout'], short_con['hidden_dropout'],
                                         short_con['feat_dropout'])
        else:
            self.decoder_ob = ConvTransE(num_nodes, h_dim, short_con['input_dropout'], short_con['hidden_dropout'],
                                         short_con['feat_dropout'])
            self.rdecoder = ConvTransR(num_rels, h_dim, short_con['input_dropout'], short_con['hidden_dropout'],
                                       short_con['feat_dropout'])

    def forward(self, total_list, data_list, node_id_new=None, time_gap=None, device=None, mode='test'):
        # RE-GCN的更新
        t = data_list['t'][0].to(device)
        # all_triples = data_list['triple'].to(device)
        #output = total_list[t]
        if self.short:
            if t - self.sequence_len < 0:
                input_list = total_list[0:t]
            else:
                input_list = total_list[t-self.sequence_len: t]
            history_glist = [build_sub_graph(self.num_nodes, self.num_rels, snap, device) for snap in input_list]
            evolve_embs, static_emb, r_emb, _, _ = self.model_r(history_glist, self.static_graph,device=device)
            pre_emb = F.normalize(evolve_embs[-1])
        if self.long:
            new_embedding = F.normalize(self.model_t(data_list, node_id_new, time_gap, pre_emb,r_emb,device, mode))
            new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels*2]

        if self.long and self.short:
            # entity embedding fusion
            if self.fuse == 'con':
                pre_emb = self.linear_fuse(torch.cat((pre_emb, new_embedding), 1))
            elif self.fuse == 'att':
                pre_emb, e_cof = self.fuse_attention(pre_emb, new_embedding, self.en_embedding.weight)
            elif self.fuse == 'att1':
                pre_emb, e_cof = self.fuse_attention1(pre_emb, new_embedding)
            elif self.fuse == 'gate':
                pre_emb, e_cof = self.gate(pre_emb, new_embedding)
            # relation embedding fusion
            if self.r_fuse == 'short':
                r_emb = r_emb
            elif self.r_fuse == 'long':
                r_emb = new_r_embedding
            elif self.r_fuse == 'con':
                r_emb = self.linear_fuse_r(torch.cat((r_emb, new_r_embedding), 1))
            elif self.r_fuse == 'att1':
                r_emb, r_cof = self.fuse_attention_r(r_emb, new_r_embedding)
            elif self.r_fuse == 'gate':
                r_emb, r_cof = self.gate_r(r_emb, new_r_embedding)
        elif self.long and not self.short:
            pre_emb = new_embedding
            r_emb = new_r_embedding
        return pre_emb, static_emb,r_emb,evolve_embs
    def forward_train(self, total_list, data_list, node_id_new=None, time_gap=None, device=None, mode='test'):
        # RE-GCN的更新
        t = data_list['t'][0].to(device)
        # all_triples = data_list['triple'].to(device)
        #output = total_list[t]
        if self.short:
            if t - self.sequence_len < 0:
                input_list = total_list[0:t]
            else:
                input_list = total_list[t-self.sequence_len: t]
            history_glist = [build_sub_graph(self.num_nodes, self.num_rels, snap, device) for snap in input_list]
            evolve_embs, static_emb, r_emb, _, _ = self.model_r.forward_train(history_glist, self.static_graph,device=device)
            pre_emb = F.normalize(evolve_embs[-1])
        if self.long:
            new_embedding = F.normalize(self.model_t.forward_train(data_list, node_id_new, time_gap, pre_emb,r_emb,device, mode))
            new_r_embedding = self.model_t.rel_embedding.weight[0:self.num_rels*2]

        if self.long and self.short:
            # entity embedding fusion
            if self.fuse == 'con':
                pre_emb = self.linear_fuse(torch.cat((pre_emb, new_embedding), 1))
            elif self.fuse == 'att':
                pre_emb, e_cof = self.fuse_attention(pre_emb, new_embedding, self.en_embedding.weight)
            elif self.fuse == 'att1':
                pre_emb, e_cof = self.fuse_attention1(pre_emb, new_embedding)
            elif self.fuse == 'gate':
                pre_emb, e_cof = self.gate(pre_emb, new_embedding)
            # relation embedding fusion
            if self.r_fuse == 'short':
                r_emb = r_emb
            elif self.r_fuse == 'long':
                r_emb = new_r_embedding
            elif self.r_fuse == 'con':
                r_emb = self.linear_fuse_r(torch.cat((r_emb, new_r_embedding), 1))
            elif self.r_fuse == 'att1':
                r_emb, r_cof = self.fuse_attention_r(r_emb, new_r_embedding)
            elif self.r_fuse == 'gate':
                r_emb, r_cof = self.gate_r(r_emb, new_r_embedding)
        elif self.long and not self.short:
            pre_emb = new_embedding
            r_emb = new_r_embedding
        return pre_emb, static_emb,r_emb,evolve_embs


    def get_loss(self, total_list,data_list,all_triplets,  e_e_his_emb=None, e_r_his_emb=None, e_r_his_id=None, e_e_his_id=None,node_id_new=None, time_gap=None, device=None):
        loss_ent = torch.zeros(1, device=device)
        loss_rel = torch.zeros(1, device=device)
        loss_static = torch.zeros(1, device=device)

        e_r_his_emb = e_r_his_emb.to(device)
        e_e_his_emb = e_e_his_emb.to(device)

        pre_emb, static_emb, r_emb ,evolve_embs= self.forward(total_list, data_list, node_id_new,
                                        time_gap, device=device)
        # pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]


        # Time embeddings
        time_embs = self.get_init_time(data_list)


        # Convert IDs to matrices
        e_e_his_matrix, e_r_his_matrix = self.id2matrix(e_r_his_id, e_e_his_id)

        # raw scores
        score_r = F.softmax(self.decoder_ob1(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb), dim=1)
        score_rel_r = F.softmax(self.rdecoder_re1(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb), dim=1)

        # historical scores
        score_h = F.softmax(self.decoder_ob2(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb, e_r_his_matrix),
                            dim=1)
        score_rel_h = F.softmax(self.rdecoder_re2(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb, e_e_his_matrix),
                                dim=1)

        # final scores
        score_ent = self.history_rate * score_h + (1 - self.history_rate) * score_r
        scores_ent = torch.log(score_ent)
        all_triplets =  all_triplets.to(device)
        loss_ent += F.nll_loss(scores_ent, all_triplets[:, 2])

        score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
        scores_rel = torch.log(score_rel)
        loss_rel += F.nll_loss(scores_rel, all_triplets[:, 1])

        if self.use_static:
            for time_step, evolve_emb in enumerate(evolve_embs):
                if self.discount == 1:
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                elif self.discount == 0:
                    step = (self.angle * math.pi / 180)
                if self.layer_norm:
                    sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                else:
                    sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                    c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                    sim_matrix = sim_matrix / c

                mask = (math.cos(step) - sim_matrix) > 0
                loss_static += self.static_weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))

        return loss_ent, loss_rel, loss_static

    def predict(self, total_list,data_list, all_triplets, e_e_his_emb=None, e_r_his_emb=None, e_r_his_id=None, e_e_his_id=None,node_id_new=None, time_gap=None, device=None):
        """
        Predict entity and relation embeddings.

        Args:
            test_graph: m historical subgraphs.
            all_triplets: current triplets.
            e_e_his_emb: Embedding of historical facts related to the relation to be predicted at the current moment
            e_r_his_emb: Embedding of historical facts related to the entity to be predicted at the current moment
            e_r_his_id: Entity-Relation hisrotical correlation IDs.
            e_e_his_id: Subject-Obejct hisrotical correlation IDs.

        Returns:
            score_ent: Entity scores.
            score_rel: Relation scores.
        """

        with torch.no_grad():
            e_e_his_emb = e_e_his_emb.to(self.device)
            e_r_his_emb = e_r_his_emb.to(self.device)

            # evolve_embs, _, r_emb = self.forward(test_graph)
            pre_emb, _, r_emb, _ = self.forward(total_list, data_list, node_id_new,
                                                                   time_gap, device=device)
            # pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # Time embeddings
            time_embs = self.get_init_time(data_list)

            # Convert IDs to matrices
            e_e_his_matrix, e_r_his_matrix = self.id2matrix(e_r_his_id, e_e_his_id)

            # raw scores
            score_r = F.softmax(self.decoder_ob1(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb), dim=1)
            score_rel_r = F.softmax(self.rdecoder_re1(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb), dim=1)

            # historical scores
            score_h = F.softmax(self.decoder_ob2(pre_emb, r_emb, time_embs, all_triplets, e_r_his_emb, e_r_his_matrix),
                                dim=1)
            score_rel_h = F.softmax(
                self.rdecoder_re2(pre_emb, r_emb, time_embs, all_triplets, e_e_his_emb, e_e_his_matrix), dim=1)

            # final scores
            score_ent = self.history_rate * score_h + (1 - self.history_rate) * score_r
            score_ent = torch.log(score_ent)

            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)

            return score_ent, score_rel


    def fuse_attention(self, s_embedding, l_embedding, o_embedding):
        w1 = (o_embedding * torch.tanh(self.linear_s(s_embedding))).sum(1)
        w2 = (o_embedding * torch.tanh(self.linear_l(l_embedding))).sum(1)
        aff = F.softmax(torch.cat((w1.unsqueeze(1),w2.unsqueeze(1)),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention1(self, s_embedding, l_embedding):
        w1 = self.fuse_f(torch.tanh(self.linear_s(s_embedding)))
        w2 = self.fuse_f(torch.tanh(self.linear_l(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def fuse_attention_r(self, s_embedding, l_embedding):
        w1 = self.fuse_f_r(torch.tanh(self.linear_s_r(s_embedding)))
        w2 = self.fuse_f_r(torch.tanh(self.linear_l_r(l_embedding)))
        aff = F.softmax(torch.cat((w1,w2),1), 1)
        en_embedding = aff[:,0].unsqueeze(1) * s_embedding + aff[:, 1].unsqueeze(1) * l_embedding
        return en_embedding, aff

    def get_init_time(self, data):
        """
        Generate time embeddings based on quadruples.
        """
        T_idx = data['t'].to(self.device)
        batch_size = data['triple'].size(0)  # 获取batch大小

        # 将T_idx扩展为(batch_size,)形状，每个元素相同
        T_idx = T_idx.expand(batch_size)
        # T_idx = quadrupleList[:, 3] // self.time_interval

        T_idx = T_idx.unsqueeze(1).float()
        # T_idx = T_idx.float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def id2matrix(self, e_r_his_id, e_e_his_id, local_obj=None):
        """
        Convert IDs to matrices.
        """

        e_r_his_matrix = torch.zeros((len(e_r_his_id), self.num_nodes), device=self.device).float()
        e_e_his_matrix = torch.zeros((len(e_e_his_id), self.num_rels * 2), device=self.device).float()
        for i, e_id in enumerate(e_r_his_id):
            e_r_his_matrix[i, e_id] = 1
        for i, r_id in enumerate(e_e_his_id):
            e_e_his_matrix[i, r_id] = 1
        if local_obj:
            for i, obj_id in enumerate(local_obj):
                obj_id = list(obj_id)
                e_r_his_matrix[i, obj_id] = 1
        # e_r_his_matrix.cpu()
        # e_e_his_matrix.cpu()
        return e_e_his_matrix, e_r_his_matrix


class GatingMechanism(nn.Module):
    def __init__(self, entity_num, hidden_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.empty(entity_num, hidden_dim))
        nn.init.xavier_uniform_(self.gate_theta)
        # self.dropout = nn.Dropout(self.params.dropout)

    def forward(self, X: torch.FloatTensor, Y: torch.FloatTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.gate_theta)
        output = torch.mul(gate, X) + torch.mul(-gate + 1, Y)
        return output, gate