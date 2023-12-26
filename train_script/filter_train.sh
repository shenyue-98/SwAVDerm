TRAIN_FEATURE_PATH="/path/to/feature/of/train"
VAL_FEATURE_PATH="/path/to/feature/of/validation"
EXPERIMENT_PATH="./experiments/filtertrain"
python filter_with_validation.py \
--train_feature_path $TRAIN_FEATURE_PATH \
--val_feature_path $VAL_FEATURE_PATH \
--dump_path $EXPERIMENT_PATH 
