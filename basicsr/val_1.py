import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
from basicsr.utils.options import parse

def main():
    # 解析配置文件，设置分布式参数和随机种子
    #opt = parse_options(is_train=False)
    opt = parse('../options/train/HighREV/EFNet_HighREV_Deblur.yml', is_train=False)


    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 创建必要的文件夹，并初始化日志记录器
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # 使用 val 数据集作为测试数据集
    dataset_opt = opt['datasets']['val']
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(
        test_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])
    logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")

    # 创建模型
    model = create_model(opt)

    # 进行测试（验证），并保存测试结果
    logger.info(f'Testing {dataset_opt["name"]}...')
    rgb2bgr = opt['val'].get('rgb2bgr', True)
    use_image = opt['val'].get('use_image', True)
    model.validation(
        test_loader,
        current_iter=opt['name'],
        tb_logger=None,
        save_img=opt['val']['save_img'],
        rgb2bgr=rgb2bgr,
        use_image=use_image)

if __name__ == '__main__':
    main()
