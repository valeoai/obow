"""Define a generic class for training and testing learning algorithms."""
import sys
import os
import os.path
import pathlib
import datetime
import glob
import logging
import time
import math

import torch
import torch.optim
import obow.utils as utils


logger = utils.setup_dist_logger(logging.getLogger(__name__))


def initialize_optimizer(parameters, opts):
    if opts is None:
        return None

    optim_type = opts["optim_type"]
    learning_rate = opts["lr"]

    if optim_type == "adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=opts["beta"],
            weight_decay=opts["weight_decay"])
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=opts["momentum"],
            weight_decay=opts["weight_decay"],
            nesterov=opts["nesterov"])
    else:
        raise NotImplementedError(f"Unrecognized optim_type: {optim_type}.")

    return optimizer


def compute_cosine_learning_rate(epoch, start_lr, end_lr, num_epochs, warmup_epochs=0):
    if (warmup_epochs > 0) and (epoch < warmup_epochs):
        # Warm-up period.
        return start_lr * (float(epoch) / warmup_epochs)
    assert epoch >= warmup_epochs

    scale = 0.5 * (1. + math.cos((math.pi * (epoch-warmup_epochs)) / (num_epochs-warmup_epochs)))
    return end_lr + (start_lr - end_lr) * scale


class Solver:
    def __init__(
        self,
        model,
        exp_dir,
        device,
        opts,
        print_freq=100,
        optimizer=None,
        use_fp16=False,
        amp=None):
        logger.info(f"Initialize solver: {opts}")
        self.exp_dir = pathlib.Path(exp_dir)
        self.exp_name = self.exp_dir.name
        if utils.get_rank() == 0:
            os.makedirs(self.exp_dir, exist_ok=True)

        self.model = model
        self.opts = opts
        self.optimizer = optimizer
        self.use_fp16 = use_fp16
        if self.use_fp16:
            assert amp is not None
        self.amp = amp

        self.start_lr = self.opts["lr"]
        self.current_lr = self.start_lr
        self.num_epochs = self.opts["num_epochs"]
        self.lr_schedule_type = self.opts["lr_schedule_type"]
        assert self.lr_schedule_type in ("cos", "step_lr", "cos_warmup")
        # TODO: use torch.optim.lr_scheduler the package.
        if self.lr_schedule_type == "step_lr":
            self.lr_schedule = self.opts["lr_schedule"]
            self.lr_decay = self.opts["lr_decay"]
        elif self.lr_schedule_type in ("cos", "cos_warmup"):
            self.end_lr = self.opts.pop("end_lr", 0.0)
            if self.lr_schedule_type == "cos_warmup":
                self.warmup_epochs = self.opts.pop("warmup_epochs")
            else:
                self.warmup_epochs = 0
        self.eval_freq = self.opts.pop("eval_freq", 1)
        self._best_metric_name = self.opts.get("best_metric_name")
        self._best_largest = (
            self.opts["best_largest"]
            if ("best_largest" in self.opts) else True)
        self.reset_best_model_record()

        self.permanent = self.opts.get("permanent", -1)
        self.print_freq = print_freq
        self.device = device
        self._epoch = 0

    def set_device(self, device):
        self.device = device

    def reset_best_model_record(self):
        self._best_metric_val = None
        self._best_model_meters = None
        self._best_epoch = None

    def initialize_optimizer(self):
        if self.optimizer is None:
            logger.info(f"Initialize optimizer")
            parameters = filter(lambda p: p.requires_grad, self.model.parameters())
            self.optimizer = initialize_optimizer(parameters, self.opts)
        assert self.optimizer is not None

    def adjust_learning_rate(self, epoch):
        """Decay the learning rate based on schedule"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.lr_schedule_type in ("cos", "cos_warmup"):
                start_lr = param_group.get("start_lr", self.start_lr)
                end_lr = param_group.get("end_lr", self.end_lr)
                learning_rate = compute_cosine_learning_rate(
                    epoch, start_lr, end_lr, self.num_epochs, self.warmup_epochs)
            elif self.lr_schedule_type == "step_lr":  # stepwise lr schedule
                learning_rate = param_group.get("start_lr", self.start_lr)
                for milestone in self.lr_schedule:
                    learning_rate *= self.lr_decay if (epoch >= milestone) else 1.
            else:
                raise NotImplementedError(
                    f"Not supported learning rate schedule type: {self.lr_schedule_type}")

            param_group["lr"] = learning_rate
            logger.info(f"==> Set lr for group {i}: {learning_rate:.10f}")

    def adjust_learning_rate_per_iter(self, epoch, iter, num_batches):
        # TODO: the code for adjusting the learning rate needs cleaning up and
        # refactoring.
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.lr_schedule_type != "cos_warmup" or epoch >= self.warmup_epochs:
                continue
            total_iter = epoch * num_batches + iter
            start_lr = param_group.get("start_lr", self.start_lr)
            learning_rate = start_lr * (float(total_iter) / (self.warmup_epochs * num_batches))
            param_group["lr"] = learning_rate
            if (iter % 100) == 0:
                logger.info(f"==> Set lr for group {i}: {learning_rate:.10f}")

    def find_last_epoch(self, suffix):
        search_pattern = self.net_checkpoint_filename("{epoch}", suffix)
        last_epoch, _ = utils.find_last_epoch(search_pattern)
        logger.info(f"Load checkpoint of last epoch: {str(last_epoch)}")
        return last_epoch

    def delete_checkpoint(self, epoch, suffix=""):
        if utils.get_rank() == 0:
            filename = pathlib.Path(self.net_checkpoint_filename(epoch, suffix))
            if filename.is_file():
                logger.info(f"Deleting {filename}")
                os.remove(filename)
            filename = pathlib.Path(self.optim_checkpoint_filename(epoch, suffix))
            if filename.is_file():
                logger.info(f"Deleting {filename}")
                os.remove(filename)

    def save_checkpoint(self, epoch, suffix="", meters=None):
        if utils.get_rank() == 0:
            self.save_network(epoch, suffix, meters)
            self.save_optimizer(epoch, suffix)

    def save_network(self, epoch, suffix="", meters=None):
        filename = self.net_checkpoint_filename(epoch, suffix)
        logger.info(f"Saving model params to: {filename}")
        state = {
            "epoch": epoch,
            "network": self.model.state_dict(),
            "meters": meters,}
        if self.use_fp16:
            state["amp"] = self.amp.state_dict()
        torch.save(state, filename)

    def save_optimizer(self, epoch, suffix=""):
        assert self.optimizer is not None
        filename = self.optim_checkpoint_filename(epoch, suffix)
        logger.info(f"Saving model optimizer to: {filename}")
        state = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state, filename)

    def load_checkpoint(self, epoch, suffix="", load_optimizer=True):
        if epoch == -1:
            epoch = self.find_last_epoch(suffix)
        assert isinstance(epoch, int)
        assert epoch >= 0
        self.load_network(epoch, suffix) # load network parameters
        if load_optimizer: # initialize and load optimizer
            self.load_optimizer(epoch, suffix)
        self._epoch = epoch

    def load_network(self, epoch, suffix=""):
        filename = pathlib.Path(self.net_checkpoint_filename(epoch, suffix))
        logger.info(f"Loading model params from: {filename}")
        assert filename.is_file()
        checkpoint = torch.load(filename, map_location="cpu")
        self.model.load_state_dict(checkpoint["network"])
        if self.use_fp16:
            self.amp.load_state_dict(checkpoint["amp"])
        return checkpoint["epoch"]

    def load_optimizer(self, epoch, suffix=""):
        self.initialize_optimizer()
        filename = pathlib.Path(self.optim_checkpoint_filename(epoch, suffix))
        logger.info(f"Loading model optimizer from: {filename}")
        assert filename.is_file()
        checkpoint = torch.load(filename, map_location="cpu")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["epoch"]

    def net_checkpoint_filename(self, epoch, suffix=""):
        return str(self.exp_dir / f"model_net_checkpoint_{epoch}{suffix}.pth.tar")

    def optim_checkpoint_filename(self, epoch, suffix=""):
        return str(self.exp_dir / f"model_optim_checkpoint_{epoch}{suffix}.pth.tar")

    def solve(
        self,
        loader_train,
        distributed,
        sampler_train,
        loader_test=None):

        assert isinstance(distributed, bool)

        self.initialize_optimizer()
        self.reset_best_model_record()
        num_epochs = self.num_epochs
        start_epoch = self._epoch
        self._start_epoch = start_epoch
        logger.info(f"Start training from epoch {start_epoch}")
        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            self._epoch = epoch
            logger.info(f"Training epoch: [{epoch+1}/{num_epochs}] ({self.exp_name})")

            if distributed:
                logger.info(
                    f"Setting epoch={epoch} for distributed sampling.")
                assert not (sampler_train is None)
                sampler_train.set_epoch(epoch)

            self.adjust_learning_rate(epoch)
            self.run_train_epoch(loader_train, epoch)

            self.save_checkpoint(epoch + 1) # create a checkpoint in the current epoch
            is_permanent = (self.permanent > 0) and (epoch % self.permanent) == 0
            if (start_epoch != epoch) and (is_permanent is False):
                # delete the checkpoint of the previous epoch
                self.delete_checkpoint(epoch)

            if (loader_test is not None) and ((epoch+1) % self.eval_freq) == 0:
                if not isinstance(loader_test, (list, tuple)):
                    loader_test = [loader_test,]
                logger.info(f"Evaluate ({self.exp_name})")
                test_metric_logger = []
                for i, loader_test_this in enumerate(loader_test):
                    test_metric_logger.append(
                        self.evaluate(loader_test_this, test_name=str(i)))

                self.keep_best_model_record(test_metric_logger[0], epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")

    def run_train_epoch(self, loader_train, epoch):
        self.model.train()
        self.start_of_training_epoch()
        num_batches = len(loader_train)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("iter/s", utils.AverageMeter(":.2f", out_val=True))
        header = f"Epoch: [{epoch+1}]"
        for self._iter, mini_batch in enumerate(
            metric_logger.log_every(loader_train, self.print_freq, header)):
            start_time = time.time()
            self.adjust_learning_rate_per_iter(epoch, self._iter, num_batches)
            self._total_iter = self._iter + epoch * num_batches
            self.train_step(mini_batch, metric_logger)
            metric_logger["iter/s"].update(1.0 / (time.time() - start_time))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info(f"==> Results: {str(metric_logger)}")
        self.end_of_training_epoch()

        return metric_logger

    def evaluate(self, loader_test, test_name=None):
        self.model.eval()
        num_batches = len(loader_test)
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("iter/s", utils.AverageMeter(":.2f", out_val=True))
        header = "Test :" if (test_name is None) else  f"Test {test_name}:"
        for iter, mini_batch in enumerate(
            metric_logger.log_every(loader_test, self.print_freq, header)):
            start_time = time.time()
            self.evaluation_step(mini_batch, metric_logger)
            metric_logger["iter/s"].update(1.0 / (time.time() - start_time))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info(f"==> Results: {str(metric_logger)}")

        return metric_logger

    def keep_best_model_record(self, test_metric_logger, epoch):
        if self._best_metric_name is None:
            return
        if (self._best_metric_name not in test_metric_logger.meters):
            raise Warning(
                f"The provided metric {self._best_metric_name} for keeping the "
                "best model is not computed by the evaluation routine.")
            return

        val = test_metric_logger[self._best_metric_name].val
        if ((self._best_metric_val is None) or
            (self._best_largest and (val >= self._best_metric_val)) or
            ((not self._best_largest) and (val <= self._best_metric_val))):

            self._best_metric_val = val
            self._best_model_meters = str(test_metric_logger)
            self.save_checkpoint(epoch+1, suffix=".best")
            if (self._best_epoch is not None):
                self.delete_checkpoint(self.best_epoch+1, suffix=".best")
            self._best_epoch = epoch
            logger.info(
                f"==> Best results w.r.t. {self._best_metric_name}: "
                f"Epoch: [{self._best_epoch+1}] {self._best_model_meters}")

    # FROM HERE ON THERE ARE ABSTRACT FUNCTIONS THAT MUST BE IMPLEMENTED BY THE
    # CLASS THAT INHERITS THE Solver CLASS
    def train_step(self, mini_batch, metric_logger):
        """Implements a training step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es)
            * Backward propagation through the networks
            * Apply optimization step(s)
        """
        pass

    def evaluation_step(self, mini_batch, metric_logger):
        """Implements an evaluation step that includes:
            * Forward a batch through the network(s)
            * Compute loss(es) or any other evaluation metrics.
        """
        pass


    def end_of_training_epoch(self):
        pass


    def start_of_training_epoch(self):
        pass
