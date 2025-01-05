import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, epoch_saving, load_checkpoint, generate_text
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import hp_net

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type = str, default = 'configs/NTU/NTU120_XSub.yaml')
    parser.add_argument(
        "--opts",
        help = "Modify config options by adding 'KEY VALUE' pairs. ",
        default = None,
        nargs = '+',
    )
    parser.add_argument('--output', type = str, default = "/data-home/liujinfu/X-CLIP/output/test_0428")
    parser.add_argument('--resume', default = '/data-home/liujinfu/X-CLIP/output/test_0428/best.pth', type = str)
    parser.add_argument('--pretrained', type = str)
    parser.add_argument('--only_test', type = bool, default = False)
    parser.add_argument('--batch-size', type = int)
    parser.add_argument('--accumulation-steps', type = int)
    parser.add_argument("--distributed", type = bool, default = False, help = 'local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type = int, default = -1, help = 'local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(config): 
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = hp_net.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True) # find_unused_parameters=False -> find_unused_parameters=True

    start_epoch, max_accuracy = 0, 0.0

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)
    
    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST == True:
        acc1 = test(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)

        acc1, score_list, name_list, j_score_list, jm_score_list, b_score_list = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)
            
        if is_best:
            np.save(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_best_score.npy"), np.array(score_list))
            np.savetxt(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_best_name.txt"), name_list, fmt = "%s")
            np.save(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_best_j_score.npy"), np.array(j_score_list))
            np.save(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_best_jm_score.npy"), np.array(jm_score_list))
            np.save(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_best_b_score.npy"), np.array(b_score_list))

    logger.info("Inference and save last scores.....")
    acc1 = test(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    for idx, (batch_data, j_data, jm_data, b_data, _) in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        j_data = j_data.cuda(non_blocking=True)
        jm_data = jm_data.cuda(non_blocking=True)
        b_data = b_data.cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        images = images.view((-1, config.DATA.NUM_FRAMES,3) + images.size()[-2:])
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        output, j_ske_logits, jm_ske_logits, b_ske_logits = model(images, texts, j_data, jm_data, b_data)
        total_loss = criterion(output, label_id) + criterion(j_ske_logits, label_id) + criterion(jm_ske_logits, label_id) + criterion(b_ske_logits, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, (batch_data, j_data, jm_data, b_data, name) in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            j_tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            jm_tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            b_tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):   
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)
                j_data = j_data.cuda(non_blocking=True)
                jm_data = jm_data.cuda(non_blocking=True)
                b_data = b_data.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output, j_ske_logits, jm_ske_logits, b_ske_logits = model(image_input, text_inputs, j_data, jm_data, b_data)
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
                
                j_similarity = j_ske_logits.view(b, -1).softmax(dim=-1) # j
                j_tot_similarity += j_similarity
                jm_similarity = jm_ske_logits.view(b, -1).softmax(dim=-1) # jm
                jm_tot_similarity += jm_similarity
                b_similarity = b_ske_logits.view(b, -1).softmax(dim=-1) # b
                b_tot_similarity += b_similarity
            
            if(idx == 0):
                score_list = tot_similarity.data.cpu()
                name_list = name
                j_score_list = j_tot_similarity.data.cpu()
                jm_score_list = jm_tot_similarity.data.cpu()
                b_score_list = b_tot_similarity.data.cpu()
            else:
                score_list = torch.concat((score_list, tot_similarity.data.cpu()), dim = 0)
                name_list.extend(name)
                j_score_list = torch.concat((j_score_list, j_tot_similarity.data.cpu()), dim = 0) # j
                jm_score_list = torch.concat((jm_score_list, jm_tot_similarity.data.cpu()), dim = 0) # jm
                b_score_list = torch.concat((b_score_list, b_tot_similarity.data.cpu()), dim = 0) # b
                           
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1
           
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, score_list, name_list, j_score_list, jm_score_list, b_score_list

@torch.no_grad()
def test(val_loader, text_labels, model, config):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, (batch_data, j_data, jm_data, b_data, name) in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):   
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)
                j_data = j_data.cuda(non_blocking=True)
                jm_data = jm_data.cuda(non_blocking=True)
                b_data = b_data.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output, _, _, _ = model(image_input, text_inputs, j_data, jm_data, b_data)
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
            
            if(idx == 0):
                score_list = tot_similarity.data.cpu()
                name_list = name
            else:
                score_list = torch.concat((score_list, tot_similarity.data.cpu()), dim = 0)
                name_list.extend(name)
            
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    acc1 += 1
                if label_id[i] in indices_5[i]:
                    acc5 += 1
           
            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    f'Acc@1: {acc1_meter.avg:.3f}\t'
                )
    acc1_meter.sync()
    acc5_meter.sync()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    np.save(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_last_score.npy"), np.array(score_list))
    print("******example_count:", acc1_meter.count)
    np.savetxt(os.path.join(config.OUTPUT, str(dist.get_rank()) + "_last_name.txt"), name_list, fmt = "%s")
    return acc1_meter.avg

if __name__ == '__main__':
    # prepare config
    args, config = parse_option()
    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=5400))
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True # old
    torch.backends.cudnn.deterministic = True # new add
    torch.backends.cudnn.benchmark = False # new add

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    if (dist.get_rank() == 0) and (not os.path.exists(args.output)):
        os.makedirs(args.output)
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)
        
    # print(config)
    main(config)