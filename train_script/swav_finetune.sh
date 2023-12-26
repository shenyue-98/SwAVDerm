TRAINJSONL_PATH="data/finetunedatasample.jsonl"
VALJSONL_PATH="data/finetunedatasample.jsonl"
PRETRAIN_PATH="path/to/pretrainmodel"
EXPERIMENT_PATH="./experiments/finetune"

python -m torch.distributed.launch --nproc_per_node=8  --master_port 29000 eval_semisup_labeled.py \
--pretrained $PRETRAIN_PATH \
--lr 0.01 \
--lr_last_layer 0.2 \
--dump_path $EXPERIMENT_PATH \
--train_jsonl_path $TRAINJSONL_PATH \
--val_jsonl_path $VALJSONL_PATH \
--num 22 \
--image_size 224 \
--label_list data/label.list 