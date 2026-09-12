#!/bin/bash
# run_distributed_hgls.sh

# 分布式训练环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1  # 设置可见的GPU

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 数据集和模型参数
DATASET="ICEWS18"
MODEL_TYPE="bert"
PLM="bert-base-cased"
NUM_K=5

# 训练参数
LR=0.001
EPOCHS=500
BATCH_SIZE=32
N_HIDDEN=200
TASK=0.7
FUSE="gate"
R_FUSE="gate"
DECODER="seconvtranse"
HISTORY_RATE=0.3
EVALUATE_EVERY=1

# 分布式参数
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $NUM_GPUS 个GPU可用"

MASTER_PORT=29500
MASTER_ADDR="127.0.0.1"

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/results"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/distributed_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

echo "开始分布式训练..."
echo "参数:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL_TYPE"
echo "  PLM: $PLM"
echo "  Num GPUs: $NUM_GPUS"
echo "  Learning Rate: $LR"
echo "  Epochs: $EPOCHS"
echo "  日志文件: $LOG_FILE"

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="$MASTER_ADDR" \
    --master_port=$MASTER_PORT \
    src/main_fenbushi.py \
    --dataset="$DATASET" \
    --model-type="$MODEL_TYPE" \
    --plm="$PLM" \
    --num-k="$NUM_K" \
    --self-loop \
    --layer-norm \
    --relation-prediction \
    --short \
    --long \
    --lr="$LR" \
    --n-epochs="$EPOCHS" \
    --batch-size="$BATCH_SIZE" \
    --n-hidden="$N_HIDDEN" \
    --task="$TASK" \
    --fuse="$FUSE" \
    --r_fuse="$R_FUSE" \
    --decoder="$DECODER" \
    --history-rate="$HISTORY_RATE" \
    --evaluate-every="$EVALUATE_EVERY" \
    --record \
    --model_record \
    --write-output \
    --gnn=regcn \
    --k_hop=2 \
    --grad-norm=1.0 \
    2>&1 | tee "$LOG_FILE"

echo "训练完成！"