# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time
from logging import getLogger
import urllib
import jsonlines

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
    accuracy,
)
import src.resnet50 as resnet_models
from src.ftdataset import FtData
logger = getLogger()


parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with 1% or 10% labels on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--labels_perc", type=str, default="10", choices=["1", "10"],
                    help="fine-tune on either 1% or 10% of labels")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to imagenet")
parser.add_argument("--train_jsonl_path", type=str, default="/path/to/imagenet",
                    help="path to train jsonl dataset repository")
parser.add_argument("--val_jsonl_path", type=str, default="/path/to/imagenet",
                    help="path to val jsonl dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=20, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate - trunk")
parser.add_argument("--lr_last_layer", default=0.2, type=float, help="initial learning rate - head")
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
parser.add_argument("--label_list", default='/path/to/label_list', type=str,
                    help="")
parser.add_argument("--image_size", default=224, type=int,
                    help="")
def main():
    global args, best_acc
    args = parser.parse_args()
    init_distributed_mode(args)
    if not args.rank:
        # 主进程验证是否存在文件夹
        if not os.path.exists(os.path.join(os.path.abspath('.'), args.dump_path)):
            print('log path is {}'.format(os.path.join(os.path.abspath('.'), args.dump_path)))
            os.makedirs(os.path.join(os.path.abspath('.'), args.dump_path))
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )
    train_jsonl = os.path.join(args.train_jsonl_path)
    val_jsonl = os.path.join(args.val_jsonl_path)

    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        tr_normalize,
    ])


    train_dataset = FtData(train_jsonl, args.label_list,train_transform)
    val_dataset = FtData(val_jsonl, args.label_list,val_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = resnet_models.__dict__[args.arch](output_dim=args.num)
#    print(model)
    # convert batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training from random weights")

    # model to gpu
    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )

    # set optimizer
    trunk_parameters = []
    head_parameters = []
    for name, param in model.named_parameters():
        if 'head' in name:
            head_parameters.append(param)
        else:
            trunk_parameters.append(param)
    optimizer = torch.optim.SGD(
        [{'params': trunk_parameters},
         {'params': head_parameters, 'lr': args.lr_last_layer}],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.decay_epochs, gamma=args.gamma
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": (0., 0.)}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)

        # set samplers
        train_loader.sampler.set_epoch(epoch)

        scores = train(model, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model)
        training_stats.update(scores + scores_val)

        scheduler.step()

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            # model_save_path = args.dump_checkpoints
            torch.save(save_dict, os.path.join(args.dump_checkpoints, "model_{}.pth.tar".format(epoch)))
    logger.info("Fine-tuning with {}% of labels completed.\n"
                "Test accuracies: top-1 {acc1:.1f}, top-5 {acc5:.1f}".format(
                args.labels_perc, acc1=best_acc[0], acc5=best_acc[1]))


def train(model, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    """
    # running statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()

    model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        output = model(inp)

        # compute cross entropy loss
        loss = criterion(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # update stats
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inp.size(0))
        top1.update(acc1[0], inp.size(0))
        top5.update(acc5[0], inp.size(0))

        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        # verbose
        if args.rank == 0 and iter_epoch % 50 == 0:
            logger.info(
                "Epoch[{0}] - Iter: [{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec {top1.val:.3f} ({top1.avg:.3f})\t"
                "LR trunk {lr}\t"
                "LR head {lr_W}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                    lr_W=optimizer.param_groups[1]["lr"],
                )
            )
    return epoch, losses.avg, top1.avg.item(), top5.avg.item()

def validate_network(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc
    # switch to evaluate mode
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):

            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target_cuda = target.cuda(non_blocking=True)
            # compute output
            output = model(inp)
            loss = criterion(output, target_cuda)
            # output.cuda().data.cpu().numpy()
            acc1, acc5 = accuracy(output, target_cuda, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))
            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc[0]:
        best_acc = (top1.avg.item(), top5.avg.item())

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc[0]))

    return losses.avg, top1.avg.item(), top5.avg.item()


if __name__ == "__main__":
    main()
