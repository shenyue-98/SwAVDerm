#This is the pretraining script on a 8 GPU server
#In the paper we perform this with 64 GPU.
DATAJSONL_PATH="data/pretraindatasample.jsonl"
EXPERIMENT_PATH="./experiments/swav_800ep_pretrain"

python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--jsonl_path $DATAJSONL_PATH \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epochs 800 \
--batch_size 20 \
--base_lr 4.8 \
--final_lr 0.0048 \
--freeze_prototypes_niters 313 \
--wd 0.000001 \
--warmup_epochs 10 \
--start_warmup 0.3 \
--arch resnet50 \
--use_fp16 true \
--checkpoint_freq 5 \
--sync_bn apex \
--dump_path $EXPERIMENT_PATH
