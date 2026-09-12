#!/usr/bin/env python
# launch_distributed_fixed.py
import os
import sys
import subprocess

# 添加项目路径
project_path = "/home/shangxuemeng/HGLS-main"
sys.path.insert(0, project_path)
sys.path.insert(0, os.path.join(project_path, "TKG"))
sys.path.insert(0, os.path.join(project_path, "src"))

# 设置环境变量
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTHONPATH'] = f"{project_path}:{project_path}/TKG:{project_path}/src:" + os.environ.get('PYTHONPATH', '')

import torch

# 获取GPU数量
num_gpus = torch.cuda.device_count()
print(f"检测到 {num_gpus} 个GPU")

# 构建命令
cmd = [
    'torchrun',
    f'--nproc_per_node={num_gpus}',
    '--nnodes=1',
    '--node_rank=0',
    '--master_addr=127.0.0.1',
    '--master_port=29500',
    'src/main_fenbushi.py',  # 注意：你的主文件可能是main.py而不是main_fenbushi.py
    '-d', 'ICEWS18',
    '--model-type', 'bert',
    '--plm', 'bert-base-cased',
    '--num-k', '5',
    '--self-loop',
    '--layer-norm',
    '--relation-prediction',
    '--short',
    '--long',
    '--lr=0.001',
    '--fuse=gate',
    '--r_fuse=gate',
    '--record',
    '--model_record',
    '--write-output',
    '--n-epochs', '500',
    '--batch-size', '32',
    '--n-hidden', '200',
    '--task', '0.7',
    '--decoder', 'seconvtranse',
    '--history-rate', '0.3',
    '--evaluate-every', '1',
    '--gnn', 'regcn',
    '--k_hop', '2',
    '--grad-norm', '1.0'
]

print("执行命令:", ' '.join(cmd))
print("=" * 80)

# 切换到项目目录
os.chdir(project_path)

# 运行命令
subprocess.run(cmd)