#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import time
from logging import getLogger
import urllib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
from src.data import data_loader, write_tsv, get_tsv
from tqdm import tqdm

from src.utils import (
    initialize_logger,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
import src.resnet50 as resnet_models

logger = getLogger()

parser = argparse.ArgumentParser(
    description="Evaluate models: Fine-tuning with 100% labels on real image data")

#########################
#### main parameters ####
#########################
parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                    help="fine-tune on either 1% or 10% of labels")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--jsonl_path", type=str, default="/path/to/imagenet",
                    help="path to imagenet")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50",
                    type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained weights")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=20, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.01, type=float,
                    help="initial learning rate - trunk")
parser.add_argument("--lr_last_layer", default=0.2,
                    type=float, help="initial learning rate - head")
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[12, 16],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.2, help="lr decay factor")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")
parser.add_argument("--num", default=1000, type=int,
                    help="")
parser.add_argument("--sample_file_path", default='data/10percent.txt', type=str,
                    help="")
# parser.add_argument('--val_path', default="val", type=str, help='验证集所在数据集中的路径')
# parser.add_argument("--old_model_path", default="", type=str, help="path to old pretrained weights")
parser.add_argument("--projection_flag", default=0,
                    type=int, help="是否加入projection head 层")
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def load_model(model_path, logger, projection_flag=True):
    print('current model path is {}'.format(model_path))
    # build model
    print('load model…………start')
    model = resnet_models.__dict__[args.arch](output_dim=args.num)
    #    print(model)
    # convert batch norm layers
    if not projection_flag:
        print('Use model as a feature extractor!')
        model.projection_head = None
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Optionally resume from a checkpoint
    # load weights
    if os.path.isfile(model_path):
        print("model:"+str(model_path))
        print(args.pretrained)
        state_dict = torch.load(
            args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info(
                    'key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info(
                    'key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info(
            "No pretrained weights found => training from random weights")
        model = None
    print('load model…………end')
    # model to gpu
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    return model


def main():
    global args, best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    if not args.rank:
        # 主进程验证是否存在文件夹
        if not os.path.exists(os.path.join(os.path.abspath('.'), args.dump_path)):
            print('log path is {}'.format(os.path.join(
                os.path.abspath('.'), args.dump_path)))
            os.makedirs(os.path.join(os.path.abspath('.'), args.dump_path))
    fix_random_seeds(args.seed)
    logger = initialize_logger(args)

    # build data
    val_data_path = os.path.join(args.jsonl_path)

    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    val_dataset_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset = data_loader(val_data_path, val_dataset_transform)

    assert args.batch_size == 1, print('args.batch_size is not 1')
    print('assert over')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    print('loader over')
    # build model
    model_path = args.pretrained
    if args.projection_flag == 1:
        print('projection_flag=True')
        model = load_model(model_path=model_path,
                           logger=logger, projection_flag=True)
    else:
        print('projection_flag=False')
        model = load_model(model_path=model_path,
                           logger=logger, projection_flag=False)

    to_restore = {"epoch": 0, "best_acc": (0., 0.)}
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True
    label_list, pred_list, image_list = validate_network(
        val_loader=val_loader, val_dataset=val_dataset, model=model)
    model_path: str
    model_name = model_path.split('.')[0].split(os.path.sep)[-1]
    res = get_tsv(label_list, pred_list, image_list)
    tsv_path = os.path.join(
        args.dump_path, "prediction_{}.tsv".format(model_name))
    write_tsv(res, tsv_path)


def validate_network(val_loader, model, val_dataset):
    batch_time = AverageMeter()
    global best_acc
    # switch to evaluate mode
    model.eval()

    label_list = []
    pred_list = []
    image_list = []

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(tqdm(val_loader)):
            # move to gpu
            target: torch.Tensor
            inp = inp.cuda(non_blocking=True)
            image_path, image_target = val_dataset.samples[i]
            image_list.append({'image_path': image_path})
            label_list.append({'label': image_target})

            output = model(inp)

            output: torch.Tensor
            output_array = output.cuda().data.cpu().numpy()

            for pred in output:
                pred_list.append(
                    {"predictions": pred.cuda().data.cpu().numpy().tolist()})
            # exit()
            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    return label_list, pred_list, image_list


if __name__ == "__main__":
    main()
