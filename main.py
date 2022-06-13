# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
# import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, TensorboardLogger, mean_average_precision_R
from my_loss import My_Loss, My_Loss_eval

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, default='./configs/swin_config.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='./datasets/imagenet', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='./pretrained/swin_pre.pth',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, default=2, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./output/', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, date_str):
    # TODO data_loader_database; tensorboard
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn, data_loader_database = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = My_Loss(config.MODEL.NUM_CLASSES, config.MODEL.hash_length, mixup_fn, config.MODEL.LABEL_SMOOTHING,
                        config.MODEL.alph_param, config.MODEL.beta_param, config.MODEL.gamm_param)

    max_accuracy = 0.0
    max_map = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss, Map = validate(config, data_loader_val, model, data_loader_database, epoch, max_map)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        max_map = max(max_map, Map)
        logger.info(f'Max map: {max_map:.4f}')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, log_writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hash_loss_meter = AverageMeter()
    quanti_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        hash_out, cls_out = model(samples)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, targets)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        hash_loss_meter.update(hash_loss.item(), targets.size(0))
        quanti_loss_meter.update(quanti_loss.item(), targets.size(0))
        cls_loss_meter.update(cls_loss.item(), targets.size(0))

        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'hash_loss {hash_loss_meter.val:.4f} ({hash_loss_meter.avg:.4f})\t'
                f'quanti_loss {quanti_loss_meter.val:.4f} ({quanti_loss_meter.avg:.4f})\t'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    log_writer.update(train_loss=loss_meter.avg, head="train", step=epoch)
    log_writer.update(train_hash_loss=hash_loss_meter.avg, head="train", step=epoch)
    log_writer.update(train_cls_loss=cls_loss_meter.avg, head="train", step=epoch)
    log_writer.update(train_quanti_loss=quanti_loss_meter.avg, head="train", step=epoch)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, data_loader_database, epoch):

    criterion = My_Loss_eval(config.MODEL.NUM_CLASSES, config.MODEL.hash_length, config.MODEL.alph_param,
                             config.MODEL.beta_param, config.MODEL.gamm_param)
    model.eval()

    if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
        query_label_matrix = np.empty(shape=(0,))
        query_hash_matrix = np.empty(shape=(0, config.MODEL.hash_length))
        database_label_matrix = np.empty(shape=(0,))
        database_hash_matrix = np.empty(shape=(0, config.MODEL.hash_length))

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    hash_loss_meter = AverageMeter()
    quanti_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            hash_out, cls_out = model(images)

            # measure accuracy and record loss
            hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, target)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            hash_code = torch.sign(hash_out)
            hash_code = hash_code.cpu().numpy()

            query_label_matrix = np.concatenate((query_label_matrix, target.cpu().numpy()), axis=0)
            query_hash_matrix = np.concatenate((query_hash_matrix, hash_code), axis=0)


        acc1, acc5 = accuracy(cls_out, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        hash_loss_meter.update(hash_loss.item(), target.size(0))
        quanti_loss_meter.update(quanti_loss.item(), target.size(0))
        cls_loss_meter.update(cls_loss.item(), target.size(0))

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'hash_loss {hash_loss_meter.val:.4f} ({hash_loss_meter.avg:.4f})\t'
                f'quanti_loss {quanti_loss_meter.val:.4f} ({quanti_loss_meter.avg:.4f})\t'
                f'cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):

        database_batch_time = AverageMeter()
        database_loss_meter = AverageMeter()
        database_hash_loss_meter = AverageMeter()
        database_quanti_loss_meter = AverageMeter()
        database_cls_loss_meter = AverageMeter()
        database_acc1_meter = AverageMeter()
        database_acc5_meter = AverageMeter()

        for idx, (images, target) in enumerate(data_loader_database):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                hash_out, cls_out = model(images)

                # measure accuracy and record loss
                hash_loss, quanti_loss, cls_loss, loss = criterion(hash_out, cls_out, target)

            hash_code = torch.sign(hash_out)
            hash_code = hash_code.cpu().numpy()

            database_label_matrix = np.concatenate((database_label_matrix, target.cpu().numpy()), axis=0)
            database_hash_matrix = np.concatenate((database_hash_matrix, hash_code), axis=0)

            acc1, acc5 = accuracy(cls_out, target, topk=(1, 5))

            database_loss_meter.update(loss.item(), target.size(0))
            database_hash_loss_meter.update(hash_loss.item(), target.size(0))
            database_quanti_loss_meter.update(quanti_loss.item(), target.size(0))
            database_cls_loss_meter.update(cls_loss.item(), target.size(0))

            database_acc1_meter.update(acc1.item(), target.size(0))
            database_acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            database_batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Database: [{idx}/{len(data_loader_database)}]\t'
                    f'Time {database_batch_time.val:.3f} ({database_batch_time.avg:.3f})\t'
                    f'Loss {database_loss_meter.val:.4f} ({database_loss_meter.avg:.4f})\t'
                    f'hash_loss {database_hash_loss_meter.val:.4f} ({database_hash_loss_meter.avg:.4f})\t'
                    f'quanti_loss {database_quanti_loss_meter.val:.4f} ({database_quanti_loss_meter.avg:.4f})\t'
                    f'cls_loss {database_cls_loss_meter.val:.4f} ({database_cls_loss_meter.avg:.4f})\t'
                    f'Acc@1 {database_acc1_meter.val:.3f} ({database_acc1_meter.avg:.3f})\t'
                    f'Acc@5 {database_acc5_meter.val:.3f} ({database_acc5_meter.avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB')

        Map, Recall = mean_average_precision_R(database_hash_matrix, query_hash_matrix, database_label_matrix,
                                               query_label_matrix, config.DATA.TOP_K, config.MODEL.NUM_CLASSES)

        print("Map:", Map, "Recall:", Recall)
        return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, Map
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, 0.0


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
        _, config = parse_option()

        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

        seed = config.SEED
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
        # gradient accumulation also need to scale the learning rate
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()

        config.MODEL.alph_param = config.MODEL.alph_param
        config.MODEL.beta_param = config.MODEL.beta_param
        config.MODEL.hash_length = config.MODEL.hash_length

        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr

        date_str = '/' + str(config.MODEL.hash_length) + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        config.OUTPUT = config.OUTPUT + config.DATA.DATASET + date_str
        config.freeze()

        # print(config.OUTPUT)

        os.makedirs(config.OUTPUT, exist_ok=True)
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

        # if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())

        main(config)
