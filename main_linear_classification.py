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
        description='Linear classification evaluation using a pre-trained with '
                    'OBoW feature extractor (from the student network).')
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='Number of data loading workers (default: 4)')
    parser.add_argument(
        '-b', '--batch-size', default=256, type=int,
        help='Mini-batch size (default: 256), this is the total '
             'batch size of all GPUs on the current node when '
             'using Data Parallel or Distributed Data Parallel.')
    parser.add_argument(
        '--start-epoch', default=0, type=int,
        help='Manual epoch number to start training in case of restart (default 0).'
             'If -1, then it stargs training from the last available checkpoint.')
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
        '--name', default='semi_supervised', type=str,
        help='The directory name of the experiment. The final directory '
             'where the model and logs would be stored is: '
             '"dst-dir / config / name", where dst-dir is the base directory '
             'for the OBoW model and config is the name of the config file '
             'that was used for training the model.')
    parser.add_argument(
        '--evaluate', action='store_true', help='Evaluate the model.')
    parser.add_argument(
        '--dataset', required=True, default='', type=str,
        help='Dataset that will be used for the linear classification '
             'evaluation. Supported options: ImageNet, Places205.')
    parser.add_argument(
        '--data-dir', required=True, type=str, default='',
        help='Directory path to the ImageNet or Places205 datasets.')
    parser.add_argument('--subset', default=-1, type=int,
        help='The number of images per class  that they would be use for '
             'training (default -1). If -1, then all the availabe images are '
             'used.')
    parser.add_argument(
        '-n', '--batch-norm', action='store_true',
        help='Use batch normalization (without affine transform) on the linear '
             'classifier. By default this option is deactivated.')
    parser.add_argument('--epochs', default=100, type=int,
        help='Number of total epochs to run (default 100).')
    parser.add_argument('--lr', '--learning-rate', default=10.0, type=float,
        help='Initial learning rate (default 10.0)', dest='lr')
    parser.add_argument('--cos-schedule', action='store_true',
        help='If True then a cosine learning rate schedule is used. Otherwise '
             'a step-wise learning rate schedule is used. In this latter case, '
             'the schedule and lr-decay arguments must be specified.')
    parser.add_argument(
        '--schedule', default=[15, 30, 45,], nargs='*', type=int,
        help='Learning rate schedule (when to drop lr by a lr-decay ratio) '
             '(default: 15, 30, 45). This argument is only used in case of '
             'step-wise learning rate schedule (when the cos-schedule flag is '
             'not activated).')
    parser.add_argument(
        '--lr-decay', default=0.1, type=float,
        help='Learning rate decay step (default 0.1). This argument is only '
        'used in case of step-wise learning rate schedule (when the '
        'cos-schedule flag is not activated).' )
    parser.add_argument('--momentum', default=0.9, type=float,
        help='Momentum (default 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
        help='Weight decay (default: 0.)', dest='weight_decay')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument(
        '--precache', action='store_true',
        help='Precache features for the linear classifier. Those features are '
             'deleted after the end of training.')
    parser.add_argument(
        '--cache-dir', default='', type=str,
        help='destination directory for the precached features.')
    parser.add_argument(
        '--cache-5crop', action='store_true',
        help='Use five crops when precaching features (only for the train set).')

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

    if args.precache:
        if args.cache_dir == '':
            raise ValueError(
                'To precache the features (--precache argument) you need to '
                'specify with the --cache-dir argument the directory where the '
                'features will be stored.')
        cache_dir_name = f"{args.config}"
        args.cache_dir = pathlib.Path(args.cache_dir) / cache_dir_name
        os.makedirs(args.cache_dir, exist_ok=True)
        args.cache_dir = pathlib.Path(args.cache_dir) / "cache_features"
        os.makedirs(args.cache_dir, exist_ok=True)

    return args


def setup_model_for_distributed_training(model, args, ngpus_per_node):
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.linear_classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                model.linear_classifier)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
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

        # According to Distributed Data Paralled (DDP) pytorch page:
        # https://pytorch.org/docs/stable/notes/ddp.html?highlight=distributed
        # "The DDP constructor takes a reference to the local module, and
        #  broadcasts state_dict() from the process with rank 0 to all other
        #  processes in the group to make sure that all model replicas start
        #  from the exact same state"
        # So, all processes have exactly the same replica of the model at this
        # moment.
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

    print(f'==> workers={args.workers}')

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
        print(f"Creating linear classifier model with {arch} backbone.")
    feature_extractor, channels = obow.feature_extractor.FeatureExtractor(
        arch=arch, opts=args.exp_config['model']['feature_extractor_opts'])
    dataset_to_num_classes = {
        "ImageNet": 1000,
        "Places205": 205,
    }
    assert args.dataset in dataset_to_num_classes
    linear_classifier_opts = {
        "num_classes": dataset_to_num_classes[args.dataset],
        "num_channels": channels,
        "batch_norm": args.batch_norm,
        "pool_type": "none",
    }
    search_pattern = "feature_extractor_net_checkpoint_{epoch}.pth.tar"
    search_pattern = str(args.feature_extractor_dir / search_pattern)
    _, filename = obow.utils.find_last_epoch(search_pattern)
    print(f"Loading pre-trained feature extractor from: {filename}")
    out_msg = obow.utils.load_network_params(
        feature_extractor, filename, strict=False)
    print(f"Loading output msg: {out_msg}")
    #assert str(out_msg) == "<All keys matched successfully>"

    model = obow.classification.FrozenFeaturesLinearClassifier(
        feature_extractor=feature_extractor,
        linear_classifier_opts=linear_classifier_opts,
    )
    if args.gpu == 0 or args.gpu is None:
        print(f"Model:\n{model}")

    model_without_ddp = model
    model, args = setup_model_for_distributed_training(
        model, args, ngpus_per_node)

    if args.precache:
        feature_extractor = model.precache_feature_extractor()
        if args.distributed:
            raise NotImplementedError(
                "Precaching with distributed is not supported.")
        loader_train, sampler_train, _, loader_test, _, _ = (
            obow.datasets.get_data_loaders_linear_classification_precache(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                workers=args.workers,
                epoch_size=None,
                feature_extractor=feature_extractor,
                cache_dir=args.cache_dir,
                device=torch.device(args.gpu),
                precache_batch_size=200,
                five_crop=args.cache_5crop,
                subset=args.subset))
    else:
        loader_train, sampler_train, _, loader_test, _, _ = (
            obow.datasets.get_data_loaders_classification(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                workers=args.workers,
                distributed=args.distributed,
                epoch_size=None,
                subset=args.subset))

    if args.cos_schedule:
        optim_opts = {
            "optim_type": "sgd",
            "lr": args.lr,
            "num_epochs": args.epochs,
            "lr_schedule_type": "cos",
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": args.nesterov,
        }
    else:
        optim_opts = {
            "optim_type": "sgd",
            "lr": args.lr,
            "num_epochs": args.epochs,
            "lr_schedule_type": "step_lr",
            "lr_schedule": args.schedule,
            "lr_decay": args.lr_decay,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": args.nesterov,
        }
    device = torch.device(args.gpu) # or torch.device('cuda')
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

    if args.precache:
        # Delete precached features.
        import shutil
        shutil.rmtree(args.cache_dir)

if __name__ == '__main__':
    main()
