#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : main
# @Software: PyCharm
from pathlib import Path

import torch
import os

from tqdm import tqdm

from TKG.rgcn.knowledge_graph import _read_triplets_as_list
from TKG.rgcn.utils import build_sub_graph
from src.DSE import get_historical_embeddings
from TKG.utils import  myFloder, Collate, Logger, mkdir_if_not_exist
from rgcn import utils
from torch.utils.data import DataLoader
from TKG.load_data import load_data
from src.hgls import HGLS
import argparse
import yaml
from yaml import SafeLoader
import datetime
import numpy as np
import dgl
import sys

from TKG.utils_new import myFloder_new, collate_new
import warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
THIS_DIR = Path(__file__).parent.resolve()
warnings.filterwarnings('ignore')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, total_data, test_dataset, all_ans_list_test,all_ans_r_list, node_id_new, s_t, test_sid,model_name,mode="test"):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    test_losses = []
    if mode == "test":
        checkpoint = torch.load(model_name, map_location=device)
        print(f"Loaded model: {model_name}. Best epoch: {checkpoint['epoch']}")  # use best stat checkpoint
        print("\n" + "-"*10 + " Start Testing " + "-"*10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for test_data_list in tqdm(test_dataset):
        with torch.no_grad():
            # final_score, final_score_r, test_loss = \
            #     model(total_data, test_data_list, node_id_new[:, test_data_list['t'][0]].to(device),
            #           (test_data_list['t'][0] - s_t[:, test_data_list['t'][0]]).to(device), device=device)
            final_score, final_r_score = model.predict(total_data, test_data_list,test_data_list['triple'], ent_ent_his_emb[test_data_list['t'][0].item()], ent_rel_his_emb[test_data_list['t'][0].item()], ent_rel_his_triplets_id[test_data_list['t'][0].item()],ent_ent_his_triplets_id[test_data_list['t'][0].item()],
                                                  node_id_new[:, test_data_list['t'][0]].to(device),
                                                  (test_data_list['t'][0] - s_t[:, test_data_list['t'][0]]).to(device), device=device)

            rank_raw, rank_filter = utils.get_total_rank(test_data_list['triple'].to(device),
                                                                                    final_score,
                                                                                    all_ans_list_test[
                                                                                        test_data_list['t'][
                                                                                            0] - test_sid],
                                                                                    eval_bz=1000, rel_predict=0)
            rank_raw_r, rank_filter_r = utils.get_total_rank(test_data_list['triple'].to(device),
                                                                                    final_r_score,
                                                                                    all_ans_r_list[
                                                                                        test_data_list['t'][
                                                                                            0] - test_sid],
                                                                                    eval_bz=1000, rel_predict=1)
            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            ranks_raw_r.append(rank_raw_r)
            ranks_filter_r.append(rank_filter_r)
            # test_losses.append(test_loss.item())
    # mrr_raw, h1_raw, h3_raw, h10_raw = utils.stat_ranks(ranks_raw)
    # mrr_filter, h1_f, h3_f, h10_f = utils.stat_ranks(ranks_filter)
    mrr_raw, hit_result_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter, hit_result_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r, hit_result_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r, hit_result_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    if mode == "test" and args['write_output']:
        hits = [1, 3, 10]
        os.makedirs(f"{THIS_DIR.parent}/outputs", exist_ok=True)
        with open(f"{THIS_DIR.parent}/outputs/{args['dataset']}-outputs1.txt", 'a') as f:
            f.write(f"best epoch: {checkpoint['epoch']}\n")
            f.write(f"model_name: {model_name}\n")
            f.write("args: {}\n".format(args))
            f.write(f"mrr_raw: {mrr_raw}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_raw[hit_i]}\n")
            f.write(f"mrr_filter: {mrr_filter}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_filter[hit_i]}\n")
            f.write(f"mrr_raw_r: {mrr_raw_r}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_raw_r[hit_i]}\n")
            f.write(f"mrr_filter_r: {mrr_filter_r}\n")
            for hit_i, hit in enumerate(hits):
                f.write(f"hits@{hit}: {hit_result_filter_r[hit_i]}\n")
            f.write("\n")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGLS')
    parser.add_argument("--gpu", type=int,default='0', help="gpu")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")

    # Diachronic Semantic Encoder Configuration
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-k', type=int, default=5, help="Number of neighbors for DSEP")
    parser.add_argument('--model-type', type=str, required=True, help="Type of pre-trained model (e.g., bert, t5)")
    parser.add_argument('--plm', type=str, required=True, help="Pre-trained language model for DSEP")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    parser.add_argument("--task", type=float, default=0.7, help="weight of entity prediction task")
    parser.add_argument("--short", action='store_true', default=False, help="short-term")
    parser.add_argument("--long", action='store_true', default=False, help="long-term")
    parser.add_argument('--gnn', default='regcn')
    parser.add_argument('--fuse', default='con', help='entity fusion')
    parser.add_argument('--r_fuse', default='re', help='relation fusion')
    # parser.add_argument("--r_p", action='store_true', default=True, help="tkg")     # 关系预测
    parser.add_argument("--record", action='store_true', default=False, help="save log file")
    parser.add_argument("--model_record", action='store_true', default=False, help="save model file")
    parser.add_argument("--write-output", action='store_true', default=False, help="Write output to file")
    # configuration for optimal parameters
    parser.add_argument('--config', type=str, default='long_config.yaml')
    parser.add_argument("--history-rate", type=float, default=0.3, help="History rate")
    parser.add_argument("--decoder", type=str,default="seconvtranse")
    parser.add_argument("--evaluate-every", type=int, default=1, help="Perform evaluation every n epochs")
    parser.add_argument("--weight", type=float, default=0.5, help="weight of static constraint")

    args = parser.parse_args().__dict__                                          # REGCN 的参数
    short_con = yaml.load(open('short_config.yaml'), Loader=SafeLoader)[args['dataset']]
    long_con = yaml.load(open('long_config.yaml'), Loader=SafeLoader)[args['dataset']]
    log_file = f'{args["dataset"]}_short_{args["short"]}_long_{args["long"]}_' \
               f'f_{args["fuse"]}_fr_{args["r_fuse"]}_ta_{args["task"]}' \
               f'_gnn1_{long_con["encoder"]}_{long_con["a_layer_num"]}_gnn2_{long_con["decoder"]}_{long_con["d_layer_num"]}' \
               f'_seq_{short_con["sequence"]}_{short_con["sequence_len"]}_max_length_{long_con["max_length"]}_fil_{long_con["filter"]}_ori_{long_con["ori"]}' \
               f'last_{long_con["last"]}'
    if args['record']:
        log_file_path = f'results/g_{args["gpu"]}_' + log_file
        mkdir_if_not_exist(log_file_path)
        sys.stdout = Logger(log_file_path)
        print(f'Logging to {log_file_path}')

    #选择环境
    device = torch.device('cuda:1')
    # use_cuda = args['gpu']>= 0 and torch.cuda.is_available()
    # device = torch.device(f'cuda:{args['gpu']}' if use_cuda else 'cpu')
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu']
    ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids  = get_historical_embeddings(
        args['dataset'], args['num_k'], plm=args['plm'], model_type=args['model_type'], batch_size=args['batch_size'], gpu=args['gpu'], save=True
        )
    embedding_dim = ent_embs.shape[1]

    num_nodes, num_rels, train_list, valid_list, test_list, total_data, all_ans_list_test, all_ans_list_r_test, \
    all_ans_list_valid, all_ans_list_r_valid, graph, node_id_new, s_t, s_f, s_l, train_sid, valid_sid, test_sid, \
    total_times, time_idx = load_data(args['dataset'])

    model_name = f"{args['plm']}_gl_rate_{args['history_rate']}-{args['dataset']}-{args['gnn']}-{short_con['decoder']}-ly{args['n_layers']}-his{short_con['sequence_len'] }-num-k{args['num_k']}"
    os.makedirs(f'{THIS_DIR.parent}/models', exist_ok=True)
    model_state_file = os.path.join(THIS_DIR.parent, 'models/', model_name)
    print(f"Model save path: {model_state_file}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load static graph
    if short_con['use_static']:
        static_triples = np.array(_read_triplets_as_list(f"{THIS_DIR.parent}/data/{args['dataset']}/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] +=  num_nodes  # Adjust node IDs
        static_node_id = torch.from_numpy(np.arange(num_words + num_nodes)).view(-1, 1).long().to(device)
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, "True", args['gpu'])
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    short_con['num_static_rels'] = num_static_rels
    short_con['num_words'] = num_words
    # short_con['static_triples'] = static_triples
    # short_con['static_graph'] = static_graph
    # HGLS的参数补充
    long_con['time_length'] = len(total_data)
    long_con['time_idx'] = time_idx
    print(args)
    print(short_con)
    print(long_con)

    model = HGLS(args,graph.to(device),args['decoder'], num_nodes, num_rels,args['n_hidden'], args['task'], args['relation_prediction'],device,
                 args['short'], args['long'], args['fuse'], args['r_fuse'], args['history_rate'],embedding_dim, ent_embs, rel_embs, static_graph,short_con, long_con).to(device)
    # model = HGLS(args,args['decoder'], num_nodes, num_rels, args['n_hidden'], args['task'], args['relation_prediction'],device,
    #              args['short'], args['long'], args['fuse'], args['r_fuse'], args['history_rate'],embedding_dim, ent_embs, rel_embs, static_graph,short_con, long_con).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    model.apply(inplace_relu)
    if args['dataset'] in ['ICEWS05-15', 'ICEWS18', 'GDELT']:
        print('load data from folder')
        train_path =f"{THIS_DIR.parent}/TKG/data/" + args['dataset'] + '/train/'
        valid_path = f"{THIS_DIR.parent}/TKG/data/" + args['dataset'] + '/val/'
        test_path = f"{THIS_DIR.parent}/TKG/data/"  + args['dataset'] + '/test/'
        train_set = myFloder_new(train_path, dgl.load_graphs)
        val_set = myFloder_new(valid_path, dgl.load_graphs)
        test_set = myFloder_new(test_path, dgl.load_graphs)
        # train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=collate_new, shuffle=True, pin_memory=True, num_workers=8)
        # val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True, num_workers=3)
        # test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True, num_workers=3)
        train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=collate_new, shuffle=True,
                                   pin_memory=True, num_workers=0)
        val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True,
                                 num_workers=0)
        test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=collate_new, shuffle=False,
                                  pin_memory=True, num_workers=0)
    else:
        print('load data online')
        train_set = myFloder(train_list, max_batch=100, start_id=train_sid, no_batch=True, mode='train')
        val_set = myFloder(valid_list, max_batch=100, start_id=valid_sid, no_batch=True, mode='test')
        test_set = myFloder(test_list, max_batch=100, start_id=test_sid, no_batch=True, mode='test')
        co = Collate(num_nodes, num_rels, s_f, s_t, len(total_data), args['dataset'], long_con['encoder'], long_con['decoder'], max_length=long_con['max_length'], all=False, graph=graph, k=2)
        # train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=co.collate_rel, shuffle=True, pin_memory=True, num_workers=8)
        # val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=co.collate_rel, shuffle=False, pin_memory=True, num_workers=4)
        # test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=co.collate_rel, shuffle=False, pin_memory=True, num_workers=4)
        train_dataset = DataLoader(dataset=train_set, batch_size=1, collate_fn=collate_new, shuffle=True,
                                   pin_memory=True, num_workers=0)
        val_dataset = DataLoader(dataset=val_set, batch_size=1, collate_fn=collate_new, shuffle=False, pin_memory=True,
                                 num_workers=0)
        test_dataset = DataLoader(dataset=test_set, batch_size=1, collate_fn=collate_new, shuffle=False,
                                  pin_memory=True, num_workers=0)

    best_mrr = 0
    best_epoch = 0
    for epoch in range(args['n_epochs']):
        print('Epoch {}'.format(epoch), '_', 'Start training: ', datetime.datetime.now(),
              '=============================================')
        model.train()
        stop = True
        losses = []
        losses_e = []
        losses_r = []
        losses_static = []
        for train_data_list in tqdm(train_dataset):
            # loss_e, loss_r, loss = model(total_data, train_data_list, node_id_new[:, train_data_list['t'][0]].to(device),
            #                             (train_data_list['t'][0] - s_t[:, train_data_list['t'][0]]).to(device), device=device, mode='train')
            # a=train_data_list['t'][0].item()
            loss_e, loss_r, loss_static = model.get_loss(total_data, train_data_list,train_data_list['triple'], ent_ent_his_emb[train_data_list['t'][0].item()], ent_rel_his_emb[train_data_list['t'][0].item()], ent_rel_his_triplets_id[train_data_list['t'][0].item()],ent_ent_his_triplets_id[train_data_list['t'][0].item()],
                                                  node_id_new[:, train_data_list['t'][0].item()].to(device),
                                                  (train_data_list['t'][0] - s_t[:, train_data_list['t'][0]]).to(device), device=device)
            loss = args['task'] * loss_e + (1 - args['task']) * loss_r + loss_static
            losses.append(loss.item())
            # loss_es.append(loss_e.item())
            losses_e.append(loss_e.item())
            losses_r.append(loss_r.item())
            losses_static.append(loss_static.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm'])  # clip gradients
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch {}, loss {:.4f} Best MRR: {:.4f} Best Epoch: {:04d}'.format(epoch, np.mean(losses),best_mrr,best_epoch), datetime.datetime.now())

        # validation
        if epoch % args['evaluate_every'] == 0:
            print('\tStart validating: ', datetime.datetime.now())
            val_result = test(model, total_data, val_dataset, all_ans_list_valid, all_ans_list_r_valid,node_id_new, s_t, valid_sid,model_state_file,mode="train")
        # print('\ttrain_loss:%.4f\tval_loss:%.4f\tval_Mrr_raw:%.4f\tval_Hits(raw)@1:%.4f\tval_Hits(raw)@3:%.4f\tval_Hits(raw)@10:%.4f'
        #       '\tval_Mrr_filter:%.4f\tval_Hits(filter)@1:%.4f\tval_Hits(filter)@3:%.4f\tval_Hits(filter)@10:%.4f' %
        #       (np.mean(losses), val_result[0], val_result[1][0], val_result[1][1], val_result[1][2], val_result[1][3],
        #        val_result[2][0], val_result[2][1], val_result[2][2], val_result[2][3]))
            if not args['relation_prediction']:  # entity prediction evalution
                if val_result[0] > best_mrr:
                    best_mrr = val_result[0]
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                if val_result[2] > best_mrr:
                    best_mrr = val_result[2]
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        if epoch >= args['n_epochs']:
            break
        print('\tStart testing: ', datetime.datetime.now())
        test_result = test(model, total_data, test_dataset, all_ans_list_test,all_ans_list_r_test, node_id_new, s_t, test_sid,model_state_file,mode="test")
        # print('\tval_loss:%.4f\tval_Mrr_raw:%.4f\tval_Hits(raw)@1:%.4f\tval_Hits(raw)@3:%.4f\tval_Hits(raw)@10:%.4f'
        #       '\tval_Mrr_filter:%.4f\tval_Hits(filter)@1:%.4f\tval_Hits(filter)@3:%.4f\tval_Hits(filter)@10:%.4f' %
        #       (test_result[0], test_result[1][0], test_result[1][1], test_result[1][2], test_result[1][3],
        #        test_result[2][0], test_result[2][1], test_result[2][2], test_result[2][3]))






