# Optimizing Skin Disease Diagnosis: Harnessing Online Community Data with Contrastive Learning and Clustering Techniques

This code provides a PyTorch implementation and pretrained models for the paper. [link to be added]

The training and finetuning of SwAV model mostly follow the original paper of [SwAV](https://github.com/facebookresearch/swav), with some adjustments in the reading of dataset to improve I/O flexibility. Please follow the original paper to get the instructions.

The 'Huifu' app mentioned in the paper can be found in Wechat app by searching '慧肤互联网医院'.

# Pretrained model weights
We release our pre-trained SwAV model on the dermatology images with the hope that other researchers might also benefit by transfering this to other dermatology tasks. You can find it in

```
models/derm_pretrained.pth
```
and get it with git lfs.

# Pretrain script
It is almost the same with the original paper of SwAV. We change the loading of dataset to use jsonline files for the dataset information. Also you can change --dump_path to determine the location of the results output. You can find sample jsonline file in the /data folder. 
```
bash train_script/swav_800epoch_pretrain.sh
```
# Finetune script
Still we only change the loading of dataset. The option of --label_list requires a list for all the label included in the downstream task.
```
bash train_script/swav_finetune.sh
```
# Use pretrained model as a feature extractor
We use our pretrained model as a feature extractor for validation images and coarse labeled training images. Run this script, and you will get a tsv file including the image paths, the features and labels.
```
bash train_script/get_features.sh
```
# Data cleansing based on distance
We use the feature of validation images to estimate class center and filter the training set.
```
bash train_script/filter_train.sh
```
