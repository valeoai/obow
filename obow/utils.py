import glob
import os
import pathlib
import datetime
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed


from collections import defaultdict


def setup_printing(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


class setup_dist_logger(object): # very hacky...
    def __init__(self, logger):
        self.logger = logger
        self.is_main_process = is_main_process()

    def info(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)


def setup_logger(dst_dir, name):
    logger = logging.getLogger(name)

    strHandler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
    strHandler.setFormatter(formatter)
    logger.addHandler(strHandler)
    logger.setLevel(logging.INFO)

    log_dir = dst_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)
    now_str = datetime.datetime.now().__str__().replace(' ','_')
    now_str = now_str.replace(' ','_').replace('-','').replace(':','')
    logger.addHandler(logging.FileHandler(log_dir / f'LOG_INFO_{now_str}.txt'))

    return logger


logger = setup_dist_logger(logging.getLogger(__name__))


@torch.no_grad()
def reduce_all(tensor):
    if get_world_size() > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def concat_all_gather(tensor):
    if get_world_size() > 1:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)
    else:
        return tensor


@torch.no_grad()
def top1accuracy(output, target):
    pred = output.max(dim=1)[1]
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy


@torch.no_grad()
def sanity_check_for_distributed_training(model, buffers_only_bow_extr=True):
    """ Verifies that all nodes have the same copy of params & bow buffers. """
    if get_world_size() > 1:
        world_size = get_world_size()
        rank = get_rank()
        is_close_all = True
        list_of_failed_states = []
        torch.distributed.barrier()
        for name, state in model.named_parameters():
            state = state.data.detach()
            state_src = state.clone()
            torch.distributed.barrier()
            torch.distributed.broadcast(state_src, src=0)
            torch.distributed.barrier()
            is_close = torch.allclose(state, state_src, rtol=1e-05, atol=1e-08)
            is_close_tensor = torch.tensor(
                [is_close], dtype=torch.float64, device='cuda')
            torch.distributed.barrier()
            is_close_all_nodes = concat_all_gather(is_close_tensor)
            is_close_all_nodes = [v >= 0.5 for v in is_close_all_nodes.tolist()]
            is_close_all_nodes_reduce = all(is_close_all_nodes)
            is_close_all &= is_close_all_nodes_reduce

            status = "PASSED" if is_close_all_nodes_reduce else "FAILED"

            logger.info(f"====> Check {name}: [{status}]")
            if not is_close_all_nodes_reduce:
                logger.info(f"======> Failed nodes: [{is_close_all_nodes}]")
                list_of_failed_states.append(name)

        for name, state in model.named_buffers():
            if buffers_only_bow_extr and name.find("module.bow_extractor") == -1:
                continue
            state = state.data.detach().float()
            state_src = state.clone()
            torch.distributed.barrier()
            torch.distributed.broadcast(state_src, src=0)
            torch.distributed.barrier()
            is_close = torch.allclose(state, state_src, rtol=1e-05, atol=1e-08)
            is_close_tensor = torch.tensor(
                [is_close], dtype=torch.float64, device='cuda')
            torch.distributed.barrier()
            is_close_all_nodes = concat_all_gather(is_close_tensor)
            is_close_all_nodes = [v >= 0.5 for v in is_close_all_nodes.tolist()]
            is_close_all_nodes_reduce = all(is_close_all_nodes)
            is_close_all &= is_close_all_nodes_reduce

            status = "PASSED" if is_close_all_nodes_reduce else "FAILED"

            logger.info(f"====> Check {name}: [{status}]")
            if not is_close_all_nodes_reduce:
                logger.info(f"======> Failed nodes: [{is_close_all_nodes}]")
                list_of_failed_states.append(name)

        status = "ALL PASSED" if is_close_all else "FAILED"
        logger.info(f"==> Sanity checked [{status}]")
        if not is_close_all:
            logger.info(f"====> List of failed states:\n{list_of_failed_states}")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt=':.4f', out_val=False):
        self.fmt = fmt
        self.out_val = out_val
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        values = torch.tensor(
            [self.count, self.sum], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(values)
        values = values.tolist()
        self.count = int(values[0])
        self.sum = values[1]

    def __str__(self):
        if self.out_val:
            fmtstr = '{avg' + self.fmt + '} ({val' + self.fmt + '})'
            return fmtstr.format(avg=self.avg, val=self.val)
        else:
            fmtstr = '{avg' + self.fmt + '}'
            return fmtstr.format(avg=self.avg)


class MetricLogger(object):
    def __init__(self, delimiter="\t", prefix=""):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getitem__(self, attr):
        if not (attr in self.meters):
            self.meters[attr] = AverageMeter()
        return self.meters[attr]

    def __str__(self):
        meters_str = []
        for key, meter in self.meters.items():
            meters_str.append("{}: {}".format(key, str(meter)))
        return self.delimiter.join(meters_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, sync=True):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter(out_val=True)
        data_time = AverageMeter(out_val=True)
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        log_msg_fmt = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'])
        if torch.cuda.is_available():
            log_msg_cuda_fmt = 'max mem: {memory:.0f}'
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                log_msg = log_msg_fmt.format(
                    i, len(iterable),
                    eta=eta_string,
                    meters=str(self),
                    time=str(iter_time),
                    data=str(data_time))
                if torch.cuda.is_available():
                    log_msg_cuda = log_msg_cuda_fmt.format(
                        memory=torch.cuda.max_memory_allocated() / MB)
                    log_msg = self.delimiter.join([log_msg, log_msg_cuda])
                logger.info(log_msg)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f'{header} Total time: {total_time}')



def global_pooling(x, type):
    assert x.dim() == 4
    if type == 'max':
        return F.max_pool2d(x, (x.size(2), x.size(3)))
    elif type == 'avg':
        return F.avg_pool2d(x, (x.size(2), x.size(3)))
    else:
        raise ValueError(
            f"Unknown pooling type '{type}'. Supported types: ('avg', 'max').")


class GlobalPooling(nn.Module):
    def __init__(self, type):
        super(GlobalPooling, self).__init__()
        assert type in ("avg", "max")
        self.type = type

    def forward(self, x):
        return global_pooling(x, self.type)

    def extra_repr(self):
        s = f'type={self.type}'
        return s


class L2Normalize(nn.Module):
    def __init__(self, dim):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def convert_from_5d_to_4d(tensor_5d):
    _, _, channels, height, width = tensor_5d.size()
    return tensor_5d.view(-1, channels, height, width)


def add_dimension(tensor, dim_size):
    assert((tensor.size(0) % dim_size) == 0)
    return tensor.view(
        [dim_size, tensor.size(0) // dim_size,] + list(tensor.size()[1:]))


def find_last_epoch(search_pattern):
    print(f"Search the last checkpoint with pattern {str(search_pattern)}")

    search_pattern = search_pattern.format(epoch="*")

    all_files = glob.glob(search_pattern)
    if len(all_files) == 0:
        raise ValueError(f"{search_pattern}: no such file.")

    substrings = search_pattern.split("*")
    assert(len(substrings) == 2)
    start, end = substrings
    all_epochs = [fname.replace(start,"").replace(end,"") for fname in all_files]
    all_epochs = [int(epoch) for epoch in all_epochs if epoch.isdigit()]
    assert(len(all_epochs) > 0)
    all_epochs = sorted(all_epochs)
    last_epoch = int(all_epochs[-1])

    checkpoint_filename = search_pattern.replace("*", str(last_epoch))
    print(f"Last epoch: {str(last_epoch)} ({checkpoint_filename})")

    checkpoint_filename = pathlib.Path(checkpoint_filename)
    assert checkpoint_filename.is_file()

    return last_epoch, checkpoint_filename


def load_network_params(network, filename, strict=True):
    if isinstance(filename, str):
        filename = pathlib.Path(filename)

    print(f"[Rank {get_rank()}: load network params from: {filename}")
    assert filename.is_file()
    checkpoint = torch.load(filename, map_location="cpu")
    return network.load_state_dict(checkpoint["network"], strict=strict)
