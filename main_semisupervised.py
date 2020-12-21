import argparse
import os
import random
import warnings
import pathlib
import yaml

import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.distributed
import torch.multiprocessing

import obow.feature_extractor
import obow.classification
import obow.utils
import obow.datasets
from obow import project_root


def get_arguments():
    """ Parse input arguments. """
    default_dst_dir = str(pathlib.Path(project_root) / "experiments")
    parser = argparse.ArgumentParser(
        description='Semi-supervised ImageNet evaluation using a pre-trained '
                    'feature extractor.')
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='Number of data loading workers (default 4)')
    parser.add_argument(
        '-b', '--batch-size', default=256, type=int,
        help='Mini-batch size (default: 256), this is the total '
             'batch size of all GPUs on the current node when '
             'using Data Parallel or Distributed Data Parallel.')
    parser.add_argument(
        '--start-epoch', default=0, type=int,
        help='Manual epoch number to start training in case of restart.'
             'If -1, then it starts training from the last available checkpoint.')
    parser.add_argument(
        '-p', '--print-freq', default=200, type=int,
        help='Print frequency (default: 200)')
    parser.add_argument(
        '--world-size', default=1, type=int,
        help='Number of nodes for distributed training (default 1)')
    parser.add_argument(
        '--rank', default=0, type=int,
        help='Node rank for distributed training (default 0)')
    parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:4444', type=str,
        help='Url used to set up distributed training '
             '(default tcp://127.0.0.1:4444)')
    parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help='Distributed backend (default nccl)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='Seed for initializing training (default None)')
    parser.add_argument(
        '--gpu', default=None, type=int,
        help='GPU id to use (default: None). If None it will try to use all '
             'the available GPUs.')
    parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training')
    parser.add_argument(
        '--dst-dir', default=default_dst_dir, type=str,
        help='Base directory where the experiments data (i.e, checkpoints) of '
             'the pre-trained OBoW model is stored (default: '
             f'{default_dst_dir}). The final directory path would be: '
             '"dst-dir / config", where config is the name of the config file.')
    parser.add_argument(
        '--config', type=str, required=True, default="",
        help='Config file that was used for training the OBoW model.')
    parser.add_argument(
        '--evaluate', action='store_true', help='Evaluate the model.')
    parser.add_argument(
        '--name', default='semi_supervised', type=str,
        help='The directory name of the experiment. The final directory '
             'where the model and logs would be stored is: '
             '"dst-dir / config / name", where dst-dir is the base directory '
             'for the OBoW model and config is the name of the config file '
             'that was used for training the model.')
    parser.add_argument(
        '--data-dir', required=True, type=str, default="",
        help='Directory path to the ImageNet dataset.')
    parser.add_argument(
        '--percentage', default=1, type=int,
        help='Percentage of ImageNet annotated images (default 1). Only the '
             'values 1 (for 1 percent of annotated images) and 10 (for 10 '
             'percent of annotated images) are supported.')
    parser.add_argument('--epochs', default=40, type=int,
        help='Number of total epochs to run.')
    parser.add_argument('--lr', default=0.0002, type=float,
        help='Initial learning rate for the feature extractor trunk '
             '(default 0.0002).')
    parser.add_argument('--lr-head', default=0.5, type=float,
        help='Initial learning rate of the classification head (default 0.5).')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='Momentum (default 0.9).')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
        help='Weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument(
        '--lr-decay', default=0.2, type=float,
        help='Learning rate decay step (default 0.2).')
    parser.add_argument(
        '--schedule', default=[24, 32,], nargs='*', type=int,
        help='Learning rate schedule, i.e., when to drop lr by a lr-decay ratio'
             ' (default: 24, 32 which means after 24 and 32 epochs)')
    parser.add_argument('--nesterov', action='store_true')

    args = parser.parse_args()
    args.feature_extractor_dir = pathlib.Path(args.dst_dir) / args.config
    os.makedirs(args.feature_extractor_dir, exist_ok=True)
    args.exp_dir = args.feature_extractor_dir /  args.name
    os.makedirs(args.exp_dir, exist_ok=True)

    # Load the configuration params of the experiment
    full_config_path = pathlib.Path(project_root) / "config" / (args.config + ".yaml")
    print(f"Loading experiment {full_config_path}")
    with open(full_config_path, "r") as f:
        args.exp_config = yaml.load(f, Loader=yaml.SafeLoader)

    print(f"Logs and/or checkpoints will be stored on {args.exp_dir}")

    return args


def setup_model_for_distributed_training(model, args, ngpus_per_node):
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif (args.gpu is not None) or (ngpus_per_node == 1):
        if (args.gpu is None) and ngpus_per_node == 1:
            args.gpu = 0
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError(
            "torch.nn.DataParallel is not supported. "
            "Use DistributedDataParallel instead with the argument "
            "--multiprocessing-distributed).")

    return model, args


def main():
    args = get_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        torch.multiprocessing.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    print(
        f"gpu = {gpu} ngpus_per_node={ngpus_per_node} "
        f"distributed={args.distributed} args={args}")
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print(f"args.rank = {args.rank}")
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)

    torch.backends.cudnn.benchmark = True
    arch = args.exp_config['model']['feature_extractor_arch']
    if args.gpu == 0 or args.gpu is None:
        obow.utils.setup_logger(args.exp_dir, "obow")
        print(f"Creating classification model with {arch} backbone.")

    feature_extractor, num_channels = obow.feature_extractor.FeatureExtractor(
        arch=arch, opts={"global_pooling": True})
    linear_classifier_opts = {
        "num_classes": 1000,
        "num_channels": num_channels,
        "batch_norm": False,
        "pool_type": "none",
    }
    search_pattern = "feature_extractor_net_checkpoint_{epoch}.pth.tar"
    search_pattern = str(args.feature_extractor_dir / search_pattern)
    _, pretrained = obow.utils.find_last_epoch(search_pattern)
    print(f"Loading pre-trained feature extractor from: {pretrained}")
    out_msg = obow.utils.load_network_params(
        feature_extractor, pretrained, strict=False)
    print(f"Loading output msg: {out_msg}")

    model = obow.classification.SupervisedClassification(
        feature_extractor=feature_extractor,
        linear_classifier_opts=linear_classifier_opts,
    )
    model_without_ddp = model
    model, args = setup_model_for_distributed_training(
        model, args, ngpus_per_node)
    if args.gpu == 0 or args.gpu is None:
        print(f"Model:\n{model}")

    loader_train, sampler_train, _, loader_test, _, _ = (
        obow.datasets.get_data_loaders_semisupervised_classification(
        dataset_name="ImageNet",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        distributed=args.distributed,
        epoch_size=None,
        percentage=args.percentage))
    optim_opts = {
        "optim_type": "sgd",
        "lr": args.lr,
        "start_lr_head": args.lr_head,
        "num_epochs": args.epochs,
        "lr_schedule_type": "step_lr",
        "lr_schedule": args.schedule,
        "lr_decay": args.lr_decay,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nesterov": args.nesterov,
        "eval_freq": 4 if args.percentage == 1 else 1,
    }
    device = torch.device(args.gpu)
    solver = obow.classification.SupervisedClassifierSolver(
        model, args.exp_dir, device, optim_opts, args.print_freq)
    if args.start_epoch != 0:
        print(f"[Rank {args.gpu}] - Loading checkpoint of: {args.start_epoch}")
        solver.load_checkpoint(epoch=args.start_epoch)

    if args.start_epoch != 0 or args.evaluate:
        solver.evaluate(loader_test)

    if args.evaluate:
        return

    solver.solve(
        loader_train=loader_train,
        distributed=args.distributed,
        sampler_train=sampler_train,
        loader_test=loader_test)

if __name__ == '__main__':
    main()
