import argparse
import copy
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

import obow.builder_obow
import obow.feature_extractor
import obow.utils
import obow.datasets
import obow.visualization
import numpy as np

from obow import project_root


def get_arguments():
    """ Parse input arguments. """
    default_dst_dir = str(pathlib.Path(project_root) / "experiments")
    parser = argparse.ArgumentParser(
        description="Trains OBoW self-supervised models."
    )
    parser.add_argument(
        '-j', '--workers', default=4, type=int,
        help='Number of data loading workers (default: 4).')
    parser.add_argument(
        '-b', '--batch-size', default=256, type=int,
        help='Mini-batch size (default: 256), this is the total '
             'batch size of all GPUs on the current node when '
             'using Distributed Data Parallel. Note that if batch_size has '
             'specified in the config file, then the batch_size of the config '
             'file overloads this agruement.')
    parser.add_argument(
        '--start-epoch', default=0, type=int,
        help='Manual epoch number to start training in case of restart. '
             'If -1, then it restarts from the last available checkpoint.')
    parser.add_argument(
        '-p', '--print-freq', default=200, type=int,
        help='print frequency (default: 200)')
    parser.add_argument(
        '--world-size', default=1, type=int,
        help='Number of nodes for distributed training (default: 1)')
    parser.add_argument(
        '--rank', default=0, type=int,
        help='Node rank for distributed training (default: 0)')
    parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:4444', type=str,
        help='url used to set up distributed training '
             '(default: tcp://127.0.0.1:4444)')
    parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help='Distributed backend (default: nccl)')
    parser.add_argument(
        '--seed', default=None, type=int,
        help='Seed for initializing training (default: None).')
    parser.add_argument(
        '--gpu', default=None, type=int,
        help='GPU id to use (default: None). If None it will try to use all '
             'the available GPUs.')
    parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training.')
    parser.add_argument(
        '--dst-dir', default=default_dst_dir, type=str,
        help='Base directory where the experiments data '
             '(i.e., checkpoints, logts, etc) would be stored (default: '
             f'{default_dst_dir}). The final directory path would be: '
             '"dst-dir / config", where config is the name of the config file.')
    parser.add_argument(
        '--config', type=str, required=True, default="",
        help='Config file with parameters of the experiment.')
    parser.add_argument(
        '--data-dir', required=True, type=str, default="",
        help='Directory path to the ImageNet dataset.')

    # Arguments related to the few-shot evaluation of the learned features.
    parser.add_argument(
        '--evaluate', action='store_true',
        help='Evaluate the model. No training is performed in this case.'
             'By default it evaluates the model of the last available checkpoint.')
    parser.add_argument(
        '--episodes', default=0, type=int,
        help='Number of episodes for few-shot evaluation (default 0).')
    parser.add_argument('--fewshot-k', default=[1,], nargs='*', type=int,
        help='Number of training examples per class for few-shot evaluatation.')
    parser.add_argument(
        '--fewshot-n', default=50, type=int,
        help='Number of novel classes per episode for few-shot evaluation.')
    parser.add_argument(
        '--fewshot-q', default=1, type=int,
        help='Number of test examples per class for few-shot evaluatation.')
    parser.add_argument(
        '--convert-to-torchvision', action='store_true',
        help='Converts and saves the student resnet backbone in torchvision '
             'format. No training or evaluation is performed in this case. '
             'Note that it converts the model of the last available checkpoint.')
    parser.add_argument(
        '--visualize-words', action='store_true',
        help='Visualize the visual words of OBoW. '
             'No training or evaluation is performed in this case. '
             'Note that it visualizes the model of the last available checkpoint.')

    args = parser.parse_args()
    exp_directory = pathlib.Path(args.dst_dir) / args.config
    os.makedirs(exp_directory, exist_ok=True)

    # Load the configuration params of the experiment
    full_config_path = pathlib.Path(project_root) / "config" / (args.config + ".yaml")
    print(f"Loading experiment {full_config_path}")
    with open(full_config_path, "r") as f:
        args.exp_config = yaml.load(f, Loader=yaml.SafeLoader)
    args.exp_dir = exp_directory

    if "batch_size" in args.exp_config["data"]:
        args.batch_size = args.exp_config["data"].pop("batch_size")

    print(f"Logs and/or checkpoints will be stored on {exp_directory}")

    return args


def setup_model_distributed_data_parallel(model, args):
    if args.distributed:
        if args.gpu is not None:
            model.cuda(args.gpu)
            make_bn_sync = torch.nn.SyncBatchNorm.convert_sync_batchnorm
            model.feature_extractor = make_bn_sync(model.feature_extractor)
            model.feature_extractor_teacher = make_bn_sync(model.feature_extractor_teacher)
            # The BN layer of the weight generator in the bow_predictor is
            # not converted to synchrorized batchnorm because it is
            # unecessary: the weight generators in all GPUs get as input the
            # same vocabulary.
            print("Use synchronized BN for the feature extractors.")
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif (args.gpu is not None):
        model = model.cuda(args.gpu)

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

    if args.world_size > 1:
        raise NotImplementedError(
            f"Multi-machine distributed training (ie, "
            f"world_size={args.world_size} > 1) is not supported. "
            f"Only single-machine single-GPU and single-machine multi-GPU "
            f"training is supported.")

    if ((torch.cuda.device_count() > 1) and
        (args.gpu is not None) and
        (not args.multiprocessing_distributed)):
        raise NotImplementedError(
            f"There are {torch.cuda.device_count()} GPUs available in the "
             "machine.\nHowever, Multi-GPU training is only supported via "
             "DistributedDataParallel and requires to activate the argument "
             "--multiprocessing-distributed.\nOtherwise choose a single GPU to "
             "run the experiment, e.g., by adding the argument --gpu=0.")

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
        # Single-machine single-GPU training setting.
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def parse_model_opts(model_opts, num_channels, num_iters_total):
    bow_extractor_opts = model_opts["bow_extractor_opts"]
    num_words = bow_extractor_opts["num_words"]
    inv_delta = bow_extractor_opts["inv_delta"]
    bow_levels = model_opts["bow_levels"]
    num_bow_levels = len(bow_levels)
    if not isinstance(inv_delta, (list, tuple)):
        inv_delta = [inv_delta for _ in range(num_bow_levels)]
    if not isinstance(num_words, (list, tuple)):
        num_words = [num_words for _ in range(num_bow_levels)]

    bow_extractor_opts_list = []
    for i in range(num_bow_levels):
        bow_extr_this = copy.deepcopy(bow_extractor_opts)
        if isinstance(bow_extr_this["inv_delta"], (list, tuple)):
            bow_extr_this["inv_delta"] = bow_extr_this["inv_delta"][i]
        if isinstance(bow_extr_this["num_words"], (list, tuple)):
            bow_extr_this["num_words"] = bow_extr_this["num_words"][i]
        bow_extr_this["num_channels"] = num_channels // (2**(num_bow_levels - 1 - i))
        bow_extractor_opts_list.append(bow_extr_this)

    model_opts["bow_extractor_opts_list"] = bow_extractor_opts_list

    if model_opts.pop("alpha_cosine", False):
        alpha_base = model_opts["alpha"]
        model_opts["alpha"] = (alpha_base, num_iters_total)

    return model_opts


def visualize_words(model, args, data_opts, dataset_name, data_dir):
    loader, dataset = obow.datasets.get_data_loaders_for_visualization(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        distributed=args.distributed,
        split="train",
        **data_opts)

    all_vword_ids, all_vword_mag, num_words = (
        obow.visualization.extract_visual_words(model, loader))

    num_words_freq, vwords_order = [], []
    for i, v in enumerate(all_vword_ids):
        print(f"all_vword_ids[{i}]: {v.shape}")
        num_words_freq.append(np.bincount(v.reshape(-1), minlength=num_words[i]))
        num_words_freq[i] = num_words_freq[i].reshape(-1)
        print(f"num_words_freq[{i}]: {num_words_freq[i].shape}")
        vwords_order.append(np.argsort(-num_words_freq[i]))
        print(f"vwords_order[{i}]: {vwords_order[i].shape}")

    num_patches = 8
    patch_size = 64
    num_levels = len(num_words)
    levels = list(range(num_levels))
    levels.reverse()
    for i in levels:
        dst_dir = os.path.join(str(args.exp_dir), f"visual_words_L{i}")
        print(f"Saving visualizations on {dst_dir}")
        os.makedirs(dst_dir, exist_ok=True)
        obow.visualization.visualize_visual_words(
            num_words[i], num_patches, patch_size, dataset, all_vword_ids[i],
            all_vword_mag[i], vwords_order[i], dst_dir)


def main_worker(gpu, ngpus_per_node, args):
    print(f"main_worker(gpu={gpu} ngpus_per_node={ngpus_per_node} args={args})")
    args.gpu = gpu

    if args.gpu is not None:
        print(f"==> Use GPU: {args.gpu} for training.")

    if args.distributed:
        # Single-machine Multi-GPU training setting.
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.rank < 0 or args.rank > (args.world_size // ngpus_per_node):
            raise ValueError(
                f"Invalid rank argument {args.rank}. "
                 "Rank must specify the id of the current machine in the "
                 "multi-machine distributed training setting. In case of "
                 "single-machine multi-gpu distributed setting (which is the "
                 "most common) then rank must be 0, ie, the id of the single "
                 "machine.")
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            print(f'Rank={args.rank}: workers={args.workers} batch_size={args.batch_size}')
    else:
        # Single-machine single-GPU training setting.
        if (args.gpu is None) and ngpus_per_node == 1:
            args.gpu = 0
        torch.cuda.set_device(args.gpu)

    torch.backends.cudnn.benchmark = True

    if args.gpu == 0 or args.gpu is None:
        obow.utils.setup_logger(args.exp_dir, "obow")

    data_opts = args.exp_config["data"]
    dataset_name = data_opts.pop("dataset_name")
    epoch_size = data_opts.pop("epoch_size", None)

    loader_train, sampler_train, _, loader_test, _, _ = (
        obow.datasets.get_data_loaders_for_OBoW(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            distributed=args.distributed,
            epoch_size=epoch_size,
            **data_opts))
    num_iters_total = len(loader_train) * args.exp_config['optim']["num_epochs"]

    model_opts = args.exp_config["model"]
    print(f"Creating an OBoW model with opts: {model_opts}")
    feature_extractor, num_channels = obow.feature_extractor.FeatureExtractor(
        arch=model_opts['feature_extractor_arch'],
        opts=model_opts['feature_extractor_opts'])
    model_opts = parse_model_opts(model_opts, num_channels, num_iters_total)

    model = obow.builder_obow.OBoW(
        feature_extractor=feature_extractor,
        num_channels=num_channels,
        bow_levels=model_opts["bow_levels"],
        bow_extractor_opts_list=model_opts["bow_extractor_opts_list"],
        bow_predictor_opts=model_opts["bow_predictor_opts"],
        alpha=model_opts["alpha"],
        num_classes=model_opts.get("num_classes", None))

    model_without_ddp = model
    model, args = setup_model_distributed_data_parallel(model, args)
    print(f"Model:\n{model}")

    optim_opts = args.exp_config['optim']
    device = torch.device(args.gpu) # or torch.device('cuda')
    solver = obow.builder_obow.OBoWSolver(
        model, args.exp_dir, device, optim_opts, args.print_freq)

    if args.evaluate or args.convert_to_torchvision or args.visualize_words:
        args.start_epoch = -1 # load the last available checkpoint.
    if args.start_epoch != 0:
        print(f"==> [Rank {args.gpu}] - Loading checkpoint of: {args.start_epoch}")
        solver.load_checkpoint(epoch=args.start_epoch)

    if args.convert_to_torchvision:
        arch = model_opts['feature_extractor_arch']
        assert arch in ("resnet50", "resnet18")
        solver.save_feature_extractor_in_torchvision_format(
            arch=model_opts['feature_extractor_arch'])
        return

    if args.visualize_words:
        model = solver.model.module if args.distributed else solver.model
        args.batch_size = 128
        visualize_words(model, args, data_opts, dataset_name, args.data_dir)
        return

    loaders_test_all = [loader_test,]
    if args.episodes > 0:
        print(f"==> Few-shot evaluation: #{args.episodes} "
              f"{args.fewshot_n}-way {args.fewshot_k}-shot "
              f"(q={args.fewshot_q}) tasks")
        loaders_fewshot_test, _, _ = obow.datasets.get_data_loaders_fewshot(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            batch_size=1,
            workers=args.workers,
            distributed=args.distributed,
            split="test" if args.evaluate else "val",
            epoch_size=args.episodes,
            num_novel=args.fewshot_n,
            num_train=args.fewshot_k,
            num_test=args.fewshot_q*args.fewshot_n)
        loaders_test_all += loaders_fewshot_test


    if args.start_epoch != 0 or args.evaluate:
        for i, loaders_test_this in enumerate(loaders_test_all):
            solver.evaluate(loaders_test_this)

    if args.evaluate:
        return

    solver.solve(
        loader_train=loader_train,
        distributed=args.distributed,
        sampler_train=sampler_train,
        loader_test=loaders_test_all)

    solver.save_feature_extractor(distributed=args.distributed)

if __name__ == '__main__':
    main()
