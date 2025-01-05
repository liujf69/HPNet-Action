import argparse
from utils.config import get_config
from utils.logger import create_logger
from datasets.build import build_dataloader

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type = str, default = 'configs/Smarthome/CS.yaml')
    parser.add_argument(
        "--opts",
        help = "Modify config options by adding 'KEY VALUE' pairs. ",
        default = None,
        nargs = '+',
    )
    parser.add_argument('--output', type = str, default = "/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/output/Smarthome/CS_0104")
    parser.add_argument('--resume', default = '/data/liujinfu/TPAMI-24/TPAMI-24/HPNet/CKPT/k600_16_8.pth', type = str)
    parser.add_argument('--pretrained', type = str)
    parser.add_argument('--only_test', type = bool, default = False)
    parser.add_argument('--batch-size', type = int)
    parser.add_argument('--accumulation-steps', type = int, default = 4)
    parser.add_argument("--distributed", type = bool, default = False, help = 'local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", type = int, default = -1, help = 'local rank for DistributedDataParallel')
    args = parser.parse_args()
    config = get_config(args)
    return args, config


if __name__ == '__main__':  
    args, config = parse_option()
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    for idx, (batch_data, j_data, jm_data, b_data, _) in enumerate(train_loader):
        
        print("debug pause")
        
    print("All Done")