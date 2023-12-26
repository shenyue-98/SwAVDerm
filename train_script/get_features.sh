JSONL_PATH="data/finetunedatasample.jsonl"
PRETRAIN_PATH="path/to/pretrainmodel"
EXPERIMENT_PATH="./experiments/features"
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29515 get_features.py \
--jsonl_path $JSONL_PATH \
--pretrained $PRETRAIN_PATH \
--dump_path $EXPERIMENT_PATH \
--batch_size 1 \
--projection_flag 0

