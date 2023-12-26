# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 直接根据image_net训练得到一个分类器
#

import argparse
import os
import time
from logging import getLogger

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

logger = getLogger()

parser = argparse.ArgumentParser(description="Evaluate models: Linear classification on ImageNet")

#########################
#### main parameters ####
#########################
parser.add_argument("--dump_path", type=str, default="data/eval_log",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")

#########################
#### model parameters ###
#########################

#  arch
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
parser.add_argument("--global_pooling", default=True, type=bool_flag,
                    help="if True, we use the resnet50 global average pooling")
parser.add_argument("--use_bn", default=False, type=bool_flag,
                    help="optionally add a batchnorm layer before the linear classifier")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=32, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--lr", default=0.3, type=float, help="initial learning rate")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--nesterov", default=False, type=bool_flag, help="nesterov momentum")
parser.add_argument("--scheduler_type", default="cosine", type=str, choices=["step", "cosine"])
# for multi-step learning rate decay
parser.add_argument("--decay_epochs", type=int, nargs="+", default=[60, 80],
                    help="Epochs at which to decay learning rate.")
parser.add_argument("--gamma", type=float, default=0.1, help="decay factor")
# for cosine learning rate schedule
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")

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
                    help="最终输出的类别数")
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def main():
    global args, best_acc
    args = parser.parse_args()
    # 创建日志文件夹存储路径
    init_distributed_mode(args)
    if not args.rank:
        # 主进程输出123
        print('1234')
        # 主进程验证是否存在文件夹
        if not os.path.exists(os.path.join(os.path.abspath('.'), args.dump_path)):
            print('log path is {}'.format(os.path.join(os.path.abspath('.'), args.dump_path)))
            os.makedirs(os.path.join(os.path.abspath('.'), args.dump_path))
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(
        args, "epoch", "loss", "prec1", "prec5", "loss_val", "prec1_val", "prec5_val"
    )

    # build data
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"))
    # print('train_dataset.targets type \n {}'.format(type(train_dataset.targets)))
    # print('train_dataset.classes type \n {}'.format(type(train_dataset.classes)))
    # print('train_dataset.classes value \n {}'.format(train_dataset.classes[:5]))
    # print('train_dataset.class_to_idx type \n {}'.format(type(train_dataset.class_to_idx)))
    # print('train_dataset.classes class_to_idx value \n {}'.format([train_dataset.class_to_idx[i] for i in train_dataset.classes[:5]]))
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, "val"))
    # print('val_dataset.targets type \n {}'.format(type(val_dataset.targets)))
    # print('val_dataset.classes type \n {}'.format(type(val_dataset.classes)))
    # print('val_dataset.classes value \n {}'.format(val_dataset.classes[:5]))
    # print('val_dataset.classes class_to_idx type \n {}'.format(type(val_dataset.class_to_idx)))
    # print('val_dataset.classes class_to_idx value \n {}'.format([val_dataset.class_to_idx[i] for i in val_dataset.classes[:5]]))
    tr_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]
    )
    train_dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        tr_normalize,
    ])
    val_dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        tr_normalize,
    ])
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
    logger.info("Building data done")

    # build model  output维度为什么是0 ，因为后期又重新加载了一下模型的
    model = resnet_models.__dict__[args.arch](output_dim=0, eval_mode=True)
    linear_classifier = RegLog(args.num, args.arch, args.global_pooling, args.use_bn)

    # convert batch norm layers (if any)
    linear_classifier = nn.SyncBatchNorm.convert_sync_batchnorm(linear_classifier)

    # model to gpu
    model = model.cuda()
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(
        linear_classifier,
        device_ids=[args.gpu_to_work_on],
        find_unused_parameters=True,
    )
    model.eval()

    # load weights
    if os.path.isfile(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location="cuda:" + str(args.gpu_to_work_on))  # 反序列化模型
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."  # 分布式训练时就会出现“module”
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                logger.info('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                logger.info('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)  # 加载模型
        logger.info("Load pretrained model with msg: {}".format(msg))
    else:
        logger.info("No pretrained weights found => training with random weights")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        nesterov=args.nesterov,
        momentum=0.9,
        weight_decay=args.wd,
    )

    # set scheduler
    if args.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, args.decay_epochs, gamma=args.gamma
        )
    elif args.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.final_lr
        )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
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
        # 这里的线性分类器指的是：linear_classifier = RegLog(1000, args.arch, args.global_pooling, args.use_bn)
        # 这里的1000应该是imagenet总的类别
        # model则是指的是对比学习训练得到的模型
        scores = train(model, linear_classifier, optimizer, train_loader, epoch)
        scores_val = validate_network(val_loader, model, linear_classifier)
        training_stats.update(scores + scores_val)

        scheduler.step()  # 更改学习率

        # save checkpoint
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.dump_path, "checkpoint.pth.tar"))
    logger.info("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""

    def __init__(self, num_labels, arch="resnet50", global_avg=False, use_bn=True):
        super(RegLog, self).__init__()
        self.bn = None
        if global_avg:
            if arch == "resnet50":
                s = 2048
            elif arch == "resnet50w2":
                s = 4096
            elif arch == "resnet50w4":
                s = 8192
            self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            assert arch == "resnet50"
            s = 8192
            self.av_pool = nn.AvgPool2d(6, stride=1)
            if use_bn:
                self.bn = nn.BatchNorm2d(2048)
        self.linear = nn.Linear(s, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # average pool the final feature map
        x = self.av_pool(x)

        # optional BN
        if self.bn is not None:
            x = self.bn(x)

        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


def train(model, reglog, optimizer, loader, epoch):
    """
    Train the models on the dataset.
    model：是主动学习训练得到的模型
    reglog：逻辑回归模型（未训练）

    """
    # running statistics
    # 以后可以使用哦
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # training statistics
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    end = time.perf_counter()
    # eval()时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值，不然的话
    model.eval()
    #  果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每
    #  一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    reglog.train()
    criterion = nn.CrossEntropyLoss().cuda()

    for iter_epoch, (inp, target) in enumerate(loader):
        # measure data loading time-------------但是从程序上看并不是数据加载的时间，而是数据输出到控制台的时间
        data_time.update(time.perf_counter() - end)
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            output = model(inp)
        # 先使用对比学习得到的模型得到图片的
        output = reglog(output)

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
        # 测量每一个batch的时间：包括数据加载时间
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
                "LR {lr}".format(
                    epoch,
                    iter_epoch,
                    len(loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )

    return epoch, losses.avg, top1.avg.item(), top5.avg.item()


def validate_network(val_loader, model, linear_classifier):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    global best_acc

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()

    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        end = time.perf_counter()
        for i, (inp, target) in enumerate(val_loader):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = linear_classifier(model(inp))
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inp.size(0))
            top1.update(acc1[0], inp.size(0))
            top5.update(acc5[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    if args.rank == 0:
        logger.info(
            "Test:\t"
            "Time {batch_time.avg:.3f}\t"
            "Loss {loss.avg:.4f}\t"
            "Acc@1 {top1.avg:.3f}\t"
            "Best Acc@1 so far {acc:.1f}".format(
                batch_time=batch_time, loss=losses, top1=top1, acc=best_acc))

    return losses.avg, top1.avg.item(), top5.avg.item()

if __name__ == "__main__":
    main()
