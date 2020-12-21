import torch
import torch.nn as nn
import torch.nn.functional as F
import obow.utils as utils
import obow.solver as solver

import logging
logger = utils.setup_dist_logger(logging.getLogger(__name__))


class PredictionHead(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        batch_norm=False,
        pred_type="linear",
        pool_type="global_avg",
        pool_params=None,
    ):
        """ Builds a prediction head for the classification task."""

        super(PredictionHead, self).__init__()

        if pred_type != "linear":
            raise NotImplementedError(
                f"Not recognized / supported prediction head type '{pred_type}'."
                f" Currently, only pred_type 'linear' is implemented.")
        self.pred_type = pred_type
        total_num_channels = num_channels

        self.layers = nn.Sequential()
        if pool_type == "none":
            if isinstance(pool_params, int):
                output_size = pool_params
                total_num_channels *= (output_size * output_size)
        elif pool_type == "global_avg":
            self.layers.add_module(
                "pooling", utils.GlobalPooling(type="avg"))
        elif pool_type == "avg":
            assert isinstance(pool_params, (list, tuple))
            assert len(pool_params) == 4
            kernel_size, stride, padding, output_size = pool_params
            total_num_channels *= (output_size * output_size)
            self.layers.add_module(
                "pooling", nn.AvgPool2d(kernel_size, stride, padding))
        elif pool_type == "adaptive_avg":
            assert isinstance(pool_params, int)
            output_size = pool_params
            total_num_channels *= (output_size * output_size)
            self.layers.add_module(
                "pooling", nn.AdaptiveAvgPool2d(output_size))
        else:
            raise NotImplementedError(
                f"Not supported pool_type '{pool_type}'. Valid pooling types: "
                "('none', 'global_avg', 'avg', 'adaptive_avg').")

        assert isinstance(batch_norm, bool)
        if batch_norm:
            # Affine is set to False. So, this batch norm layer does not have
            # any learnable (scale and bias) parameters. It's only purpose is
            # to normalize the features. So, the prediction layer is still
            # linear. It is only used for the Places205 linear classification
            # setting to make it the same as the benchmark code:
            # https://github.com/facebookresearch/fair_self_supervision_benchmark
            self.layers.add_module(
                "batch_norm", nn.BatchNorm2d(num_channels, affine=False))
        self.layers.add_module("flattening", nn.Flatten())

        prediction_layer = nn.Linear(total_num_channels, num_classes)
        prediction_layer.weight.data.normal_(0.0,  0.01)
        prediction_layer.bias.data.fill_(0.0)

        self.layers.add_module("prediction_layer", prediction_layer)

    def forward(self, features):
        return self.layers(features)


class SupervisedClassification(nn.Module):
    def __init__(
        self,
        feature_extractor,
        linear_classifier_opts,
    ):
        super(SupervisedClassification, self).__init__()
        self.feature_extractor = feature_extractor
        self.linear_classifier = PredictionHead(**linear_classifier_opts)

    def forward(self, images, labels):
        features = self.feature_extractor(images)
        scores = self.linear_classifier(features)
        loss = F.cross_entropy(scores, labels)
        with torch.no_grad():
            accuracies = utils.accuracy(scores, labels, topk=(1,5))
            accuracies = [a.item() for a in accuracies]

        return loss, accuracies


class FrozenFeaturesLinearClassifier(nn.Module):
    def __init__(
        self,
        feature_extractor,
        linear_classifier_opts,
        feature_levels=None,
    ):
        super(FrozenFeaturesLinearClassifier, self).__init__()
        self.feature_levels = feature_levels
        self.feature_extractor = feature_extractor
        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.linear_classifier = PredictionHead(**linear_classifier_opts)

    @torch.no_grad()
    def precache_feature_extractor(self):
        """Returns the feature extractor for precaching features."""
        out_feature_extractor = self.feature_extractor.get_subnetwork(
            self.feature_levels)
        self.feature_extractor = nn.Sequential()
        liner_classifier_layers = self.linear_classifier._modules["layers"]
        pooling_layer = liner_classifier_layers._modules.pop("pooling", None)
        if (pooling_layer is not  None):
            out_feature_extractor.add_module(
                "pooling_layer_from_linear_classifier",
                pooling_layer)
        return out_feature_extractor

    def forward(self, images, labels):
        # Set to evaluation mode the feature extractor to avoid training /
        # updating its batch norm statistics.
        self.feature_extractor.eval()
        with torch.no_grad():
            features = (
                self.feature_extractor(images) if self.feature_levels is None
                else self.feature_extractor(images, self.feature_levels))
        scores = self.linear_classifier(features)
        loss = F.cross_entropy(scores, labels)
        with torch.no_grad():
            accuracies = utils.accuracy(scores, labels, topk=(1,5))
            accuracies = [a.item() for a in accuracies]

        return loss, accuracies


def get_parameters(model, start_lr_head=None):
    if start_lr_head is not None:
        # Use different learning rate for the classification head of the model.
        def is_linear_head(key):
            return key.find('linear_classifier.layers.prediction_layer.') != -1
        param_group_head = [
            param for key, param in model.named_parameters()
            if param.requires_grad and is_linear_head(key)]
        param_group_trunk = [
            param for key, param in model.named_parameters()
            if param.requires_grad and (not is_linear_head(key))]
        param_group_all = [
            param for key, param in model.named_parameters()
            if param.requires_grad]
        assert len(param_group_all) == (len(param_group_head) + len(param_group_trunk))
        parameters = [
            {"params": iter(param_group_head), "start_lr": start_lr_head},
            {"params": iter(param_group_trunk)}]
        logger.info(f"#params in head: {len(param_group_head)}")
        logger.info(f"#params in trunk: {len(param_group_trunk)}")
        return parameters
    else:
        return filter(lambda p: p.requires_grad, model.parameters())


def initialize_optimizer(model, opts):
    logger.info(f"Initialize optimizer")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return solver.initialize_optimizer(parameters, opts)


class SupervisedClassifierSolver(solver.Solver):
    def end_of_training_epoch(self):
        if self._epoch == self._start_epoch:
            utils.sanity_check_for_distributed_training(self.model, False)

    def start_of_training_epoch(self):
        if self._epoch == self._start_epoch:
            utils.sanity_check_for_distributed_training(self.model, False)

    def save_feature_extractor(self, distributed=False):
        if utils.get_rank() == 0:
            epoch = self._epoch + 1
            filename = f"feature_extractor_net_checkpoint_{epoch}.pth.tar"
            filename = str(self.exp_dir / filename)
            model = self.model.module if distributed else self.model
            state = {
                "epoch": epoch,
                "network": model.feature_extractor.state_dict(),
                "meters": None,}
            torch.save(state, filename)

    def initialize_optimizer(self):
        if self.optimizer is None:
            logger.info(f"Initialize optimizer")
            start_lr_head = self.opts.get("start_lr_head", None)
            parameters = get_parameters(self.model, start_lr_head=start_lr_head)
            self.optimizer = solver.initialize_optimizer(parameters, self.opts)
        assert self.optimizer is not None

    def evaluation_step(self, mini_batch, metric_logger):
        return self._process(mini_batch, metric_logger, training=False)

    def train_step(self, mini_batch, metric_logger):
        return self._process(mini_batch, metric_logger, training=True)

    def _process(self, mini_batch, metric_logger, training):
        assert isinstance(training, bool)
        images, labels = mini_batch
        if self.device  is not None:
            images = images.cuda(self.device , non_blocking=True)
            labels = labels.cuda(self.device , non_blocking=True)

        with torch.set_grad_enabled(training):
            loss, accuracies = self.model(images, labels)

        if training:
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        batch_size = images.size(0)
        assert isinstance(accuracies, (list, tuple)) and len(accuracies) == 2
        metric_logger[f"loss"].update(loss.item(), batch_size)
        metric_logger["acc@1"].update(accuracies[0], batch_size)
        metric_logger["acc@5"].update(accuracies[1], batch_size)

        return metric_logger
