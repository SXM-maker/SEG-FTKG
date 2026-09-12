#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/14 4:28
# @Author : ZM7
# @File : main
# @Software: PyCharm
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from tqdm import tqdm

from TKG.rgcn.knowledge_graph import _read_triplets_as_list
from TKG.rgcn.utils import build_sub_graph
from src.DSE import get_historical_embeddings
from TKG.utils import myFloder, Collate, Logger, mkdir_if_not_exist
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

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
THIS_DIR = Path(__file__).parent.resolve()
warnings.filterwarnings('ignore')


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def test(model, total_data, test_dataset, all_ans_list_test, all_ans_r_list, node_id_new, s_t, test_sid, model_name,
         device, rank, world_size,  ent_ent_his_emb=None, ent_rel_his_emb=None,
         ent_rel_his_triplets_id=None, ent_ent_his_triplets_id=None,mode="test"):
    """分布式测试函数"""
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    if mode == "test":
        # 只在rank 0加载模型，然后广播到其他进程
        if rank == 0:
            checkpoint = torch.load(model_name, map_location=device)
            print(f"Loaded model: {model_name}. Best epoch: {checkpoint['epoch']}")
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = {'state_dict': None, 'epoch': 0}

        # 广播检查点信息
        checkpoint = broadcast_checkpoint(checkpoint, rank, device)
        model.load_state_dict(checkpoint['state_dict'])

        if rank == 0:
            print("\n" + "-" * 10 + " Start Testing " + "-" * 10 + "\n")

    model.eval()

    # 为测试数据创建分布式采样器
    test_sampler = DistributedSampler(
        test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # 重新创建测试数据加载器
    test_loader = DataLoader(
        test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset,
        batch_size=1,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn if hasattr(test_dataset,
                                                      'collate_fn') else test_dataset.dataset.collate_fn if hasattr(
            test_dataset.dataset, 'collate_fn') else None,
        num_workers=0,
        pin_memory=True
    )

    with torch.no_grad():
        for test_data_list in tqdm(test_loader, disable=rank != 0):
            final_score, final_r_score = model.predict(
                total_data, test_data_list, test_data_list['triple'],
                ent_ent_his_emb[test_data_list['t'][0].item()],
                ent_rel_his_emb[test_data_list['t'][0].item()],
                ent_rel_his_triplets_id[test_data_list['t'][0].item()],
                ent_ent_his_triplets_id[test_data_list['t'][0].item()],
                node_id_new[:, test_data_list['t'][0]].to(device),
                (test_data_list['t'][0] - s_t[:, test_data_list['t'][0]]).to(device),
                device=device
            )

            rank_raw, rank_filter = utils.get_total_rank(
                test_data_list['triple'].to(device),
                final_score,
                all_ans_list_test[test_data_list['t'][0] - test_sid],
                eval_bz=1000,
                rel_predict=0
            )

            rank_raw_r, rank_filter_r = utils.get_total_rank(
                test_data_list['triple'].to(device),
                final_r_score,
                all_ans_r_list[test_data_list['t'][0] - test_sid],
                eval_bz=1000,
                rel_predict=1
            )

            ranks_raw.append(rank_raw)
            ranks_filter.append(rank_filter)
            ranks_raw_r.append(rank_raw_r)
            ranks_filter_r.append(rank_filter_r)

    # 收集所有进程的结果
    all_ranks_raw = [None for _ in range(world_size)]
    all_ranks_filter = [None for _ in range(world_size)]
    all_ranks_raw_r = [None for _ in range(world_size)]
    all_ranks_filter_r = [None for _ in range(world_size)]

    dist.all_gather_object(all_ranks_raw, ranks_raw)
    dist.all_gather_object(all_ranks_filter, ranks_filter)
    dist.all_gather_object(all_ranks_raw_r, ranks_raw_r)
    dist.all_gather_object(all_ranks_filter_r, ranks_filter_r)

    # 只在主进程计算指标
    if rank == 0:
        # 展平所有列表
        all_ranks_raw_flat = []
        all_ranks_filter_flat = []
        all_ranks_raw_r_flat = []
        all_ranks_filter_r_flat = []

        for r in all_ranks_raw:
            all_ranks_raw_flat.extend(r)
        for r in all_ranks_filter:
            all_ranks_filter_flat.extend(r)
        for r in all_ranks_raw_r:
            all_ranks_raw_r_flat.extend(r)
        for r in all_ranks_filter_r:
            all_ranks_filter_r_flat.extend(r)

        mrr_raw, hit_result_raw = utils.stat_ranks(all_ranks_raw_flat, "raw_ent")
        mrr_filter, hit_result_filter = utils.stat_ranks(all_ranks_filter_flat, "filter_ent")
        mrr_raw_r, hit_result_raw_r = utils.stat_ranks(all_ranks_raw_r_flat, "raw_rel")
        mrr_filter_r, hit_result_filter_r = utils.stat_ranks(all_ranks_filter_r_flat, "filter_rel")

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
    else:
        return None, None, None, None, None, None, None, None


def broadcast_checkpoint(checkpoint, rank, device):
    """广播检查点到所有进程"""
    # 广播epoch
    epoch_tensor = torch.tensor(checkpoint['epoch']).to(device)
    dist.broadcast(epoch_tensor, src=0)
    checkpoint['epoch'] = epoch_tensor.item()

    # 广播状态字典
    if rank == 0:
        state_dict = checkpoint['state_dict']
        # 将状态字典转换为列表以便广播
        keys = list(state_dict.keys())
        keys_tensor = torch.tensor([hash(key) for key in keys]).to(device)
    else:
        keys_tensor = torch.zeros(1000, dtype=torch.int64).to(device)  # 假设最多1000个key

    dist.broadcast(keys_tensor, src=0)

    if rank != 0:
        # 重建状态字典结构
        state_dict = {}
        for key_hash in keys_tensor:
            if key_hash == 0:
                break
            # 这里需要根据实际情况调整，或者使用更复杂的广播机制
            pass

    # 简化的广播：只在rank 0加载，其他进程从rank 0接收
    if rank == 0:
        for key, tensor in state_dict.items():
            dist.broadcast(tensor, src=0)

    checkpoint['state_dict'] = state_dict
    return checkpoint


def main(rank, world_size, args):
    """主训练函数（分布式版本）"""
    # 设置分布式环境
    setup(rank, world_size)

    # 设备设置
    device = torch.device(f'cuda:{rank}')

    # 只在主进程加载历史嵌入
    if rank == 0:
        print(f"Loading historical embeddings...")
        ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb, ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids = get_historical_embeddings(
            args['dataset'], args['num_k'], plm=args['plm'], model_type=args['model_type'],
            batch_size=args['batch_size'], gpu=args['gpu'], save=True
        )
        embedding_dim = ent_embs.shape[1]

        # 加载数据
        print(f"Loading data...")
        (num_nodes, num_rels, train_list, valid_list, test_list, total_data,
         all_ans_list_test, all_ans_list_r_test, all_ans_list_valid, all_ans_list_r_valid,
         graph, node_id_new, s_t, s_f, s_l, train_sid, valid_sid, test_sid,
         total_times, time_idx) = load_data(args['dataset'])
    else:
        # 为其他进程创建空变量
        ent_embs, rel_embs, ent_ent_his_emb, ent_rel_his_emb = None, None, None, None
        ent_ent_his_triplets_id, ent_rel_his_triplets_id, new_entity_ids = None, None, None
        embedding_dim = None
        num_nodes, num_rels, train_list, valid_list, test_list = None, None, None, None, None
        total_data, all_ans_list_test, all_ans_list_r_test = None, None, None
        all_ans_list_valid, all_ans_list_r_valid = None, None
        graph, node_id_new, s_t, s_f, s_l = None, None, None, None, None
        train_sid, valid_sid, test_sid, total_times, time_idx = None, None, None, None, None

    # 广播必要的数据
    if world_size > 1:
        # 广播基本参数
        if rank == 0:
            metadata = {
                'embedding_dim': embedding_dim,
                'num_nodes': num_nodes,
                'num_rels': num_rels,
                'train_sid': train_sid,
                'valid_sid': valid_sid,
                'test_sid': test_sid
            }
        else:
            metadata = {}

        metadata = broadcast_object(metadata, rank, device)

        if rank != 0:
            embedding_dim = metadata['embedding_dim']
            num_nodes = metadata['num_nodes']
            num_rels = metadata['num_rels']
            train_sid = metadata['train_sid']
            valid_sid = metadata['valid_sid']
            test_sid = metadata['test_sid']

    # 模型名称
    model_name = f"{args['plm']}_gl_rate_{args['history_rate']}-{args['dataset']}-{args['gnn']}-{short_con['decoder']}-ly{args['n_layers']}-his{short_con['sequence_len']}-num-k{args['num_k']}"
    os.makedirs(f'{THIS_DIR.parent}/models', exist_ok=True)
    model_state_file = os.path.join(THIS_DIR.parent, '../models/', model_name)

    if rank == 0:
        print(f"Model save path: {model_state_file}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"World size: {world_size}")
        print(f"Current rank: {rank}")

    # 加载静态图（只在主进程）
    if rank == 0 and short_con['use_static']:
        static_triples = np.array(
            _read_triplets_as_list(f"{THIS_DIR.parent}/data/{args['dataset']}/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] += num_nodes  # Adjust node IDs
        static_node_id = torch.from_numpy(np.arange(num_words + num_nodes)).view(-1, 1).long().to(device)
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, "True", args['gpu'])
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # 广播静态图参数
    if world_size > 1:
        static_params = {
            'num_static_rels': num_static_rels,
            'num_words': num_words,
            'use_static': short_con['use_static']
        }
        static_params = broadcast_object(static_params, rank, device)

        if rank != 0:
            num_static_rels = static_params['num_static_rels']
            num_words = static_params['num_words']

    short_con['num_static_rels'] = num_static_rels
    short_con['num_words'] = num_words
    long_con['time_length'] = total_times if rank == 0 else 0
    long_con['time_idx'] = time_idx if rank == 0 else []

    if rank == 0:
        print(args)
        print(short_con)
        print(long_con)

    # 创建模型
    model = HGLS(
        args, graph.to(device) if rank == 0 else None, args['decoder'],
        num_nodes, num_rels, args['n_hidden'], args['task'],
        args['relation_prediction'], device, args['short'], args['long'],
        args['fuse'], args['r_fuse'], args['history_rate'],
        embedding_dim, ent_embs.to(device) if rank == 0 else None,
        rel_embs.to(device) if rank == 0 else None,
        static_graph, short_con, long_con
    ).to(device)

    # 使用DDP包装模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    model.apply(inplace_relu)

    # 创建数据集
    if rank == 0:
        if args['dataset'] in ['ICEWS05-15', 'ICEWS18', 'GDELT']:
            print('load data from folder')
            train_path = f"{THIS_DIR.parent}/TKG/data/" + args['dataset'] + '/train/'
            valid_path = f"{THIS_DIR.parent}/TKG/data/" + args['dataset'] + '/val/'
            test_path = f"{THIS_DIR.parent}/TKG/data/" + args['dataset'] + '/test/'
            train_set = myFloder_new(train_path, dgl.load_graphs)
            val_set = myFloder_new(valid_path, dgl.load_graphs)
            test_set = myFloder_new(test_path, dgl.load_graphs)
        else:
            print('load data online')
            train_set = myFloder(train_list, max_batch=100, start_id=train_sid, no_batch=True, mode='train')
            val_set = myFloder(valid_list, max_batch=100, start_id=valid_sid, no_batch=True, mode='test')
            test_set = myFloder(test_list, max_batch=100, start_id=test_sid, no_batch=True, mode='test')

    # 等待主进程完成数据加载
    dist.barrier()

    # 为数据集创建分布式采样器
    if rank == 0:
        train_dataset = DataLoader(
            dataset=train_set,
            batch_size=1,
            collate_fn=collate_new,
            shuffle=False,  # 使用分布式采样器控制shuffle
            pin_memory=True,
            num_workers=0
        )
        val_dataset = DataLoader(
            dataset=val_set,
            batch_size=1,
            collate_fn=collate_new,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )
        test_dataset = DataLoader(
            dataset=test_set,
            batch_size=1,
            collate_fn=collate_new,
            shuffle=False,
            pin_memory=True,
            num_workers=0
        )

        # 为训练数据创建分布式采样器
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=collate_new,
            pin_memory=True,
            num_workers=0
        )
    else:
        # 其他进程使用占位符
        train_loader = None
        val_dataset = None
        test_dataset = None

    # 广播数据加载器信息
    dist.barrier()

    best_mrr = 0
    best_epoch = 0

    # 训练循环
    for epoch in range(args['n_epochs']):
        if rank == 0:
            print('Epoch {}'.format(epoch), '_', 'Start training: ', datetime.datetime.now(),
                  '=============================================')

        model.train()

        # 设置epoch（用于分布式采样器的随机性）
        if rank == 0:
            train_sampler.set_epoch(epoch)

        losses = []
        losses_e = []
        losses_r = []
        losses_static = []

        # 训练批次
        if rank == 0:
            train_iter = tqdm(train_loader, disable=False)
        else:
            # 其他进程也需要迭代，但不显示进度条
            train_iter = train_loader if train_loader is not None else []

        for train_data_list in train_iter:
            # 获取训练数据
            if train_data_list is not None:
                loss_e, loss_r, loss_static = model.module.get_loss(
                    total_data, train_data_list, train_data_list['triple'],
                    ent_ent_his_emb[train_data_list['t'][0].item()] if rank == 0 else None,
                    ent_rel_his_emb[train_data_list['t'][0].item()] if rank == 0 else None,
                    ent_rel_his_triplets_id[train_data_list['t'][0].item()] if rank == 0 else None,
                    ent_ent_his_triplets_id[train_data_list['t'][0].item()] if rank == 0 else None,
                    node_id_new[:, train_data_list['t'][0].item()].to(device) if rank == 0 else None,
                    (train_data_list['t'][0] - s_t[:, train_data_list['t'][0]]).to(device) if rank == 0 else None,
                    device=device
                )

                loss = args['task'] * loss_e + (1 - args['task']) * loss_r + loss_static
                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm'])
                optimizer.step()
                optimizer.zero_grad()

        # 计算平均损失（在所有进程中同步）
        if len(losses) > 0:
            avg_loss = torch.tensor(np.mean(losses)).to(device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size
        else:
            avg_loss = 0

        if rank == 0:
            print('Epoch {}, loss {:.4f} Best MRR: {:.4f} Best Epoch: {:04d}'.format(
                epoch, avg_loss, best_mrr, best_epoch), datetime.datetime.now())

        # 验证
        if epoch % args['evaluate_every'] == 0 and rank == 0:
            print('\tStart validating: ', datetime.datetime.now())
            val_result = test(model.module, total_data, val_dataset, all_ans_list_valid,
                              all_ans_list_r_valid, node_id_new, s_t, valid_sid,
                              model_state_file, device, rank, world_size,ent_ent_his_emb,ent_rel_his_emb,ent_rel_his_triplets_id,ent_ent_his_triplets_id, mode="train")

            if not args['relation_prediction']:  # entity prediction evaluation
                if val_result[0] > best_mrr:
                    best_mrr = val_result[0]
                    best_epoch = epoch
                    torch.save({'state_dict': model.module.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                if val_result[2] > best_mrr:
                    best_mrr = val_result[2]
                    best_epoch = epoch
                    torch.save({'state_dict': model.module.state_dict(), 'epoch': epoch}, model_state_file)

        # 测试
        if epoch == args['n_epochs'] - 1 and rank == 0:  # 只在最后epoch测试
            print('\tStart testing: ', datetime.datetime.now())
            test_result = test(model.module, total_data, test_dataset, all_ans_list_test,
                               all_ans_list_r_test, node_id_new, s_t, test_sid,
                               model_state_file, device, rank, world_size,ent_ent_his_emb,ent_rel_his_emb,ent_rel_his_triplets_id,ent_ent_his_triplets_id, mode="test")

    # 清理
    cleanup()


def broadcast_object(obj, rank, device):
    """广播Python对象到所有进程"""
    if world_size == 1:
        return obj

    if rank == 0:
        # 序列化对象
        import pickle
        data = pickle.dumps(obj)
        data_tensor = torch.ByteTensor(list(data)).to(device)
        data_size = torch.tensor(len(data)).to(device)
    else:
        data_tensor = torch.ByteTensor(1000000).to(device)  # 假设最大1MB
        data_size = torch.tensor(0).to(device)

    # 广播大小
    dist.broadcast(data_size, src=0)

    # 广播数据
    dist.broadcast(data_tensor, src=0)

    if rank != 0:
        # 反序列化
        import pickle
        data = bytes(data_tensor.cpu().numpy()[:data_size.item()])
        obj = pickle.loads(data)

    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HGLS')
    parser.add_argument("--gpu", type=int, default='0', help="gpu")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset to use")
    parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False, help="add entity prediction loss")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-k', type=int, default=5, help="Number of neighbors for DSEP")
    parser.add_argument('--model-type', type=str, required=True, help="Type of pre-trained model (e.g., bert, t5)")
    parser.add_argument('--plm', type=str, required=True, help="Pre-trained language model for DSEP")
    parser.add_argument("--n-epochs", type=int, default=500, help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    parser.add_argument("--task", type=float, default=0.7, help="weight of entity prediction task")
    parser.add_argument("--short", action='store_true', default=False, help="short-term")
    parser.add_argument("--long", action='store_true', default=False, help="long-term")
    parser.add_argument('--gnn', default='regcn')
    parser.add_argument('--fuse', default='con', help='entity fusion')
    parser.add_argument('--r_fuse', default='re', help='relation fusion')
    parser.add_argument("--record", action='store_true', default=False, help="save log file")
    parser.add_argument("--model_record", action='store_true', default=False, help="save model file")
    parser.add_argument("--write-output", action='store_true', default=False, help="Write output to file")
    parser.add_argument('--config', type=str, default='long_config.yaml')
    parser.add_argument("--history-rate", type=float, default=0.3, help="History rate")
    parser.add_argument("--decoder", type=str, default="seconvtranse")
    parser.add_argument("--evaluate-every", type=int, default=1, help="Perform evaluation every n epochs")
    parser.add_argument("--weight", type=float, default=0.5, help="weight of static constraint")

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--world-size", type=int, default=-1, help="World size for distributed training")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs per node")

    args = parser.parse_args().__dict__

    # 加载配置文件
    short_con = yaml.load(open('short_config.yaml'), Loader=SafeLoader)[args['dataset']]
    long_con = yaml.load(open('long_config.yaml'), Loader=SafeLoader)[args['dataset']]

    # 日志文件
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

    # 获取GPU数量
    world_size = torch.cuda.device_count()

    # 使用torch.multiprocessing启动分布式训练
    import torch.multiprocessing as mp

    if world_size > 1:
        # 多GPU训练
        mp.spawn(
            main,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # 单GPU训练（向后兼容）
        main(0, 1, args)