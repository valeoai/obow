import copy
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import obow.utils as utils
import obow.solver as solver
import obow.fewshot as fewshot
from obow.classification import PredictionHead


logger = utils.setup_dist_logger(logging.getLogger(__name__))


@torch.no_grad()
def compute_bow_perplexity(bow_target):
    """ Compute the per image and per batch perplexity of the bow_targets. """
    assert isinstance(bow_target, (list, tuple))

    perplexity_batch, perplexity_img = [], []
    for bow_target_level in bow_target: # For each bow level.
        assert bow_target_level.dim() == 2
        # shape of bow_target_level: [batch_size x num_words]

        probs = F.normalize(bow_target_level, p=1, dim=1)
        perplexity_img_level = torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-5), dim=1)).mean()

        bow_target_sum_all = bow_target_level.sum(dim=0)
        # Uncomment the following line if you want to compute the perplexity of
        # of the entire batch in case of distributed training.
        # bow_target_sum_all = utils.reduce_all(bow_target_sum_all)
        probs = F.normalize(bow_target_sum_all, p=1, dim=0)
        perplexity_batch_level = torch.exp(
            -torch.sum(probs * torch.log(probs + 1e-5), dim=0))

        perplexity_img.append(perplexity_img_level)
        perplexity_batch.append(perplexity_batch_level)

    perplexity_batch = torch.stack(perplexity_batch, dim=0).view(-1).tolist()
    perplexity_img = torch.stack(perplexity_img, dim=0).view(-1).tolist()

    return perplexity_batch, perplexity_img


def expand_target(target, prediction):
    """Expands the target in case of BoW predictions from multiple crops."""
    assert prediction.size(1) == target.size(1)
    batch_size_x_num_crops, num_words = prediction.size()
    batch_size = target.size(0)
    assert batch_size_x_num_crops % batch_size == 0
    num_crops = batch_size_x_num_crops // batch_size

    if num_crops > 1:
        target = target.unsqueeze(1).repeat(1, num_crops, 1).view(-1, num_words)

    return target


class OBoW(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_channels,
        bow_levels,
        bow_extractor_opts_list,
        bow_predictor_opts,
        alpha=0.99,
        num_classes=None,
    ):
        """Builds an OBoW model.

        Args:
        feature_extractor: essentially the convnet model that is going to be
            trained in order to learn image representations.
        num_channels: number of channels of the output global feature vector of
            the feature_extractor.
        bow_levels: a list with the names (strings) of the feature levels from
            which the teacher network in OBoW will create BoW targets.
        bow_extractor_opts_list: a list of dictionaries with the configuration
            options for the BoW extraction (at teacher side) for each BoW level.
            Each dictionary should define the following keys (1) "num_words"
            with the vocabulary size of this level, (2) "num_channels",
            optionally (3) "update_type" (default: "local_averaging"),
            optionally (4) "inv_delta" (default: 15), which is the inverse
            temperature that is used for computing the soft assignment codes,
            and optionally (5) "bow_pool" (default: "max"). For more details
            see the documentation of the BoWExtractor class.
        bow_predictor_opts: a dictionary with configuration options for the
            BoW prediction head of the student. The dictionary must define
            the following keys (1) "kappa", a coefficent for scaling the
            magnitude of the predicted weights, and optionally (2) "learn_kappa"
            (default: False),  a boolean value that if true kappa becomes a
            learnable parameter. For all the OBoW experiments "learn_kappa" is
            set to False. For more details see the documentation of the
            BoWPredictor class.
        alpha: the momentum coefficient between 0.0 and 1.0 for the teacher
            network updates. If alpha is a scalar (e.g., 0.99) then a static
            momentum coefficient is used during training. If alpha is tuple of
            two values, e.g., alpha=(alpha_base, num_iterations), then OBoW
            uses a cosine schedule that starts from alpha_base and it increases
            it to 1.0 over num_iterations.
        num_classes: (optional) if not None, then it creates a
            linear classification head with num_classes outputs that would be
            on top of the teacher features for on-line monitoring the quality
            of the learned features. No gradients would back-propagated from
            this head to the feature extractor trunks. So, it does not
            influence the learning of the feature extractor. Note, at the end
            the features that are used are those of the student network, not
            of the teacher.
        """
        super(OBoW, self).__init__()
        assert isinstance(bow_levels, (list, tuple))
        assert isinstance(bow_extractor_opts_list, (list, tuple))
        assert len(bow_extractor_opts_list) == len(bow_levels)

        self._bow_levels = bow_levels
        self._num_bow_levels = len(bow_levels)
        if isinstance(alpha, (tuple, list)):
            # Use cosine schedule in order to increase the alpha from
            # alpha_base (e.g., 0.99) to 1.0.
            alpha_base, num_iterations = alpha
            self._alpha_base = alpha_base
            self._num_iterations = num_iterations
            self.register_buffer("_alpha", torch.FloatTensor(1).fill_(alpha_base))
            self.register_buffer("_iteration", torch.zeros(1))
            self._alpha_cosine_schedule = True
        else:
            self._alpha = alpha
            self._alpha_cosine_schedule = False

        # Build the student network components.
        self.feature_extractor = feature_extractor
        assert "kappa" in bow_predictor_opts
        bow_predictor_opts["num_channels_out"] = num_channels
        bow_predictor_opts["num_channels_hidden"] = num_channels * 2
        bow_predictor_opts["num_channels_in"] = [
            d["num_channels"] for d in bow_extractor_opts_list]
        self.bow_predictor = BoWPredictor(**bow_predictor_opts)

        # Build the teacher network components.
        self.feature_extractor_teacher = copy.deepcopy(self.feature_extractor)
        self.bow_extractor = BoWExtractorMultipleLevels(bow_extractor_opts_list)

        if (num_classes is not None):
            self.linear_classifier = PredictionHead(
                num_channels=num_channels, num_classes=num_classes,
                batch_norm=True, pool_type="global_avg")
        else:
            self.linear_classifier = None

        for param, param_teacher in zip(
            self.feature_extractor.parameters(),
            self.feature_extractor_teacher.parameters()):
            param_teacher.data.copy_(param.data)  # initialize
            param_teacher.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _get_momentum_alpha(self):
        if self._alpha_cosine_schedule:
            scale = 0.5 * (1. + math.cos((math.pi * self._iteration.item()) / self._num_iterations))
            self._alpha.fill_(1.0 - (1.0 - self._alpha_base) * scale)
            self._iteration += 1
            return self._alpha.item()
        else:
            return self._alpha

    @torch.no_grad()
    def _update_teacher(self):
        """ Exponetial moving average for the feature_extractor_teacher params:
            param_teacher = param_teacher * alpha + param * (1-alpha)
        """
        if not self.training:
            return
        alpha = self._get_momentum_alpha()
        if alpha >= 1.0:
            return
        for param, param_teacher in zip(
            self.feature_extractor.parameters(),
            self.feature_extractor_teacher.parameters()):
            param_teacher.data.mul_(alpha).add_(
                param.detach().data, alpha=(1. - alpha))

    def _bow_loss(self, bow_prediction, bow_target):
        assert isinstance(bow_prediction, (list, tuple))
        assert isinstance(bow_target, (list, tuple))
        assert len(bow_prediction) == self._num_bow_levels
        assert len(bow_target) == self._num_bow_levels

        # Instead of using a custom made cross-entropy loss for soft targets,
        # we use the pytorch kl-divergence loss that is defined as the
        # cross-entropy plus the entropy of targets. Since there is no gradient
        # back-propagation from the targets, it is equivalent to cross entropy.
        loss = [
            F.kl_div(F.log_softmax(p, dim=1), expand_target(t, p), reduction="batchmean")
            for (p, t) in zip(bow_prediction, bow_target)]
        return torch.stack(loss).mean()

    def _linear_classification(self, features, labels):
        # With .detach() no gradients of the classification loss are
        # back-propagated to the feature extractor.
        # The reason for training such a linear classifier is in order to be
        # able to monitor while training the quality of the learned features.
        features = features.detach()
        if (labels is None) or (self.linear_classifier is None):
            return (features.new_full((1,), 0.0).squeeze(),
                    features.new_full((1,), 0.0).squeeze())

        scores = self.linear_classifier(features)
        loss = F.cross_entropy(scores, labels)
        with torch.no_grad():
            accuracy = utils.top1accuracy(scores, labels).item()

        return loss, accuracy

    def generate_bow_targets(self, image):
        features = self.feature_extractor_teacher(image, self._bow_levels)
        if isinstance(features, torch.Tensor):
            features = [features,]
        bow_target, _ = self.bow_extractor(features)
        return bow_target, features

    def forward_test(self, img_orig, labels):
        with torch.no_grad():
            features = self.feature_extractor_teacher(img_orig, self._bow_levels)
            features = features if isinstance(features, torch.Tensor) else features[-1]
            features = features.detach()
            loss_cls, accuracy = self._linear_classification(features, labels)

        return loss_cls, accuracy

    def forward(self, img_orig, img_crops, labels=None):
        """ Applies the OBoW self-supervised task to a mini-batch of images.

        Args:
        img_orig: 4D tensor with shape [batch_size x 3 x img_height x img_width]
            with the mini-batch of images from which the teacher network
            generates the BoW targets.
        img_crops: list of 4D tensors where each of them is a mini-batch of
            image crops with shape [(batch_size * num_crops) x 3 x crop_height x crop_width]
            from which the student network predicts the BoW targets. For
            example, in the full version of OBoW this list will iclude a
            [(batch_size * 2) x 3 x 160 x 160]-shaped tensor with two image crops
            of size [160 x 160] pixels and a [(batch_size * 5) x 3 x 96 x 96]-
            shaped tensor with five image patches of size [96 x 96] pixels.
        labels: (optional) 1D tensor with shape [batch_size] with the class
            labels of the img_orig images. If available, it would be used for
            on-line monitoring the performance of the linear classifier.

        Returns:
        losses: a tensor with the losses for each type of image crop and
            (optionally) the loss of the linear classifier.
        logs: a list of metrics for monitoring the training progress. It
            includes the perplexity of the bow targets in a mini-batch
            (perp_b), the perplexity of the bow targets in an image (perp_i),
            and (optionally) the accuracy of a linear classifier on-line
            trained on the teacher features (this is a proxy for monitoring
            during training the quality of the learned features; Note, at the
            end the features that are used are those of the student).
        """
        if self.training is False:
            # For testing, it only computes the linear classification accuracy.
            return self.forward_test(img_orig, labels)

        #*********************** MAKE BOW PREDICTIONS **************************
        dictionary = self.bow_extractor.get_dictionary()
        features = [self.feature_extractor(x) for x in img_crops]
        bow_predictions = self.bow_predictor(features, dictionary)
        #***********************************************************************
        #******************** COMPUTE THE BOW TARGETS **************************
        with torch.no_grad():
            self._update_teacher()
            bow_target, features_t = self.generate_bow_targets(img_orig)
            perp_b, perp_i = compute_bow_perplexity(bow_target)
        #***********************************************************************
        #***************** COMPUTE THE BOW PREDICTION LOSSES *******************
        losses = [self._bow_loss(pred, bow_target) for pred in bow_predictions]
        #***********************************************************************
        #****** MONITORING: APPLY LINEAR CLASSIFIER ON TEACHER FEATURES ********
        loss_cls, accuracy = self._linear_classification(features_t[-1], labels)
        #***********************************************************************

        losses = torch.stack(losses + [loss_cls,], dim=0).view(-1)
        logs = list(perp_b + perp_i) + [accuracy,]

        return losses, logs


class BoWExtractor(nn.Module):
    def __init__(
        self,
        num_words,
        num_channels,
        update_type="local_average",
        inv_delta=15,
        bow_pool="max"):
        """Builds a BoW extraction module for the teacher network.

        It builds a BoW extraction module for the teacher network in which the
        visual words vocabulary is on-line updated during training via a
        queue-based vocabular/dictionary of randomly sampled local features.

        Args:
        num_words: the number of visual words in the vocabulary/dictionary.
        num_channels: the number of channels in the teacher feature maps and
            visual word embeddings (of the vocabulary).
        update_type: with what type of local features to update the queue-based
            visual words vocabulary. Three update types are implemenented:
            (a) "no_averaging": to update the queue it samples with uniform
            distribution one local feature vector per image from the given
            teacher feature maps.
            (b) "global_averaging": to update the queue it computes from each
            image a feature vector by globally average pooling the given
            teacher feature maps.
            (c) "local_averaging" (default option): to update the queue it
            computes from each image a feature vector by first locally averaging
            the given teacher feature map with a 3x3 kernel and then samples one
            of the resulting feature vectors with uniform distribution.
        inv_delta: the base value for the inverse temperature that is used for
            computing the soft assignment codes over the visual words, used for
            building the BoW targets. If inv_delta is None, then hard assignment
            is used instead.
        bow_pool: (default "max") how to reduce the assignment codes to BoW
            vectors. Two options are supported, "max" for max-pooling and "avg"
            for average-pooling.
        """
        super(BoWExtractor, self).__init__()

        if inv_delta is not None:
            assert isinstance(inv_delta, (float, int))
            assert inv_delta > 0.0
        assert bow_pool in ("max", "avg")
        assert update_type in ("local_average", "global_average", "no_averaging")

        self._num_channels = num_channels
        self._num_words = num_words
        self._update_type = update_type
        self._inv_delta = inv_delta
        self._bow_pool = bow_pool
        self._decay = 0.99

        embedding = torch.randn(num_words, num_channels).clamp(min=0)
        self.register_buffer("_embedding", embedding)
        self.register_buffer("_embedding_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_track_num_batches", torch.zeros(1))
        self.register_buffer("_min_distance_mean", torch.ones(1) * 0.5)

    @torch.no_grad()
    def _update_dictionary(self, features):
        """Given a teacher feature map it updates the queue-based vocabulary."""
        assert features.dim() == 4
        if self._update_type in ("local_average", "no_averaging"):
            if self._update_type == "local_average":
                features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=0)
            features = features.flatten(2)
            batch_size, _, num_locs = features.size()
            index = torch.randint(0, num_locs, (batch_size,), device=features.device)
            index += torch.arange(batch_size, device=features.device) * num_locs
            selected_features = features.permute(0,2,1).reshape(batch_size*num_locs, -1)
            selected_features = selected_features[index].contiguous()
        elif self._update_type == "global_average":
            selected_features = utils.global_pooling(features, type="avg").flatten(1)

        assert selected_features.dim() == 2
        # Gather the selected_features from all nodes in the distributed setting.
        selected_features = utils.concat_all_gather(selected_features)

        # To simplify the queue update implementation, it is assumed that the
        # number of words is a multiple of the batch-size.
        assert self._num_words % selected_features.shape[0] == 0
        batch_size = selected_features.shape[0]
        # Replace the oldest visual word embeddings with the selected ones
        # using the self._embedding_ptr pointer. Note that each training step
        # self._embedding_ptr points to the older visual words.
        ptr = int(self._embedding_ptr)
        self._embedding[ptr:(ptr + batch_size),:] = selected_features
        # move the pointer.
        self._embedding_ptr[0] = (ptr + batch_size) % self._num_words

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the visual word embeddings of the dictionary/vocabulary."""
        return self._embedding.detach().clone()

    @torch.no_grad()
    def _broadast_initial_dictionary(self):
        # Make sure every node in the distributed setting starts with the
        # same dictionary. Maybe this is not necessary and copying the buffers
        # across the models on all gpus is handled by nn.DistributedDataParallel
        embedding = self._embedding.data.clone()
        torch.distributed.broadcast(embedding, src=0)
        self._embedding.data.copy_(embedding)

    def forward(self, features):
        """Given a teacher feature maps, it generates BoW targets."""
        features = features[:, :, 1:-1, 1:-1].contiguous()

        # Compute distances between features and visual words embeddings.
        embeddings_b = self._embedding.pow(2).sum(1)
        embeddings_w = -2*self._embedding.unsqueeze(2).unsqueeze(3)
        # dist = ||features||^2 + |embeddings||^2 + conv(features, -2 * embedding)
        dist = (features.pow(2).sum(1, keepdim=True) +
                F.conv2d(features, weight=embeddings_w, bias=embeddings_b))
        # dist shape: [batch_size, num_words, height, width]
        min_dist, enc_indices = torch.min(dist, dim=1)
        mu_min_dist = min_dist.mean()
        mu_min_dist = utils.reduce_all(mu_min_dist) / utils.get_world_size()

        if self.training:
            # exponential moving average update of self._min_distance_mean.
            self._min_distance_mean.data.mul_(self._decay).add_(
                mu_min_dist, alpha=(1. - self._decay))
            self._update_dictionary(features)
            self._track_num_batches += 1

        if self._inv_delta is None:
            # Hard assignment codes.
            codes = dist.new_full(list(dist.shape), 0.0)
            codes.scatter_(1, enc_indices.unsqueeze(1), 1)
        else:
            # Soft assignment codes.
            inv_delta_adaptive = self._inv_delta / self._min_distance_mean
            codes = F.softmax(-inv_delta_adaptive * dist, dim=1)

        # Reduce assignment codes to bag-of-word vectors with global pooling.
        bow = utils.global_pooling(codes, type=self._bow_pool).flatten(1)
        bow = F.normalize(bow, p=1, dim=1) # L1-normalization.
        return bow, codes

    def extra_repr(self):
        str_options = (
            f"num_words={self._num_words}, num_channels={self._num_channels}, "
            f"update_type={self._update_type}, inv_delta={self._inv_delta}, "
            f"pool={self._bow_pool}, "
            f"decay={self._decay}, "
            f"track_num_batches={self._track_num_batches.item()}")
        return str_options


class BoWExtractorMultipleLevels(nn.Module):
    def __init__(self, opts_list):
        """Builds a BoW extractor for each BoW level."""
        super(BoWExtractorMultipleLevels, self).__init__()
        assert isinstance(opts_list, (list, tuple))
        self.bow_extractor = nn.ModuleList([
            BoWExtractor(**opts) for opts in opts_list])

    @torch.no_grad()
    def get_dictionary(self):
        """Returns the dictionary of visual words from each BoW level."""
        return [b.get_dictionary() for b in self.bow_extractor]

    def forward(self, features):
        """Given a list of feature levels, it generates multi-level BoWs."""
        assert isinstance(features, (list, tuple))
        assert len(features) == len(self.bow_extractor)
        out = list(zip(*[b(f) for b, f in zip(self.bow_extractor, features)]))
        return out


class BoWPredictor(nn.Module):
    def __init__(
        self,
        num_channels_out=2048,
        num_channels_in=[1024, 2048],
        num_channels_hidden=4096,
        kappa=8,
        learn_kappa=False
    ):
        """ Builds the dynamic BoW prediction head of the student network.

        It essentially builds a weight generation module for each BoW level for
        which the student network needs to predict BoW. For example, in its
        full version, OBoW uses two BoW levels, one for conv4 of ResNet (i.e.,
        penultimate feature scale of ResNet) and one for conv5 of ResNet (i.e.,
        final feature scale of ResNet). Therefore, in this case, the dynamic
        BoW prediction head has two weight generation modules.

        Args:
        num_channels_in: a list with the number of input feature channels for
            each weight generation module. For example, if OBoW uses two BoW
            levels and a ResNet50 backbone, then num_channels_in should be
            [1024, 2048], where the first number is the number of channels of
            the conv4 level of ResNet50 and the second number is the number of
            channels of the conv5 level of ResNet50.
        num_channels_out: the number of output feature channels for the weight
            generation modules.
        num_channels_hidden: the number of feature channels at the hidden
            layers of the weight generator modules.
        kappa: scalar with scale coefficient for the output weight vectors that
            the weight generation modules produce.
        learn_kappa (default False): if True kappa is a learnable parameter.
        """
        super(BoWPredictor, self).__init__()

        assert isinstance(num_channels_in, (list, tuple))
        num_bow_levels = len(num_channels_in)

        generators = []
        for i in range(num_bow_levels):
            generators.append(nn.Sequential())
            generators[i].add_module(f"b{i}_l2norm_in", utils.L2Normalize(dim=1))
            generators[i].add_module(f"b{i}_fc", nn.Linear(num_channels_in[i], num_channels_hidden, bias=False))
            generators[i].add_module(f"b{i}_bn", nn.BatchNorm1d(num_channels_hidden))
            generators[i].add_module(f"b{i}_rl", nn.ReLU(inplace=True))
            generators[i].add_module(f"b{i}_last_layer", nn.Linear(num_channels_hidden, num_channels_out))
            generators[i].add_module(f"b{i}_l2norm_out", utils.L2Normalize(dim=1))
        self.layers_w = nn.ModuleList(generators)

        self.scale = nn.Parameter(
            torch.FloatTensor(num_bow_levels).fill_(kappa),
            requires_grad=learn_kappa)

    def forward(self, features, dictionary):
        """Dynamically predicts the BoW from the features of cropped images.

        During the forward pass, it gets as input a list with the features from
        each type of extracted image crop and a list with the visual word
        dictionaries of each BoW level. First, it uses the weight generation
        modules for producing from each dictionary level the weight vectors
        that would be used for the BoW prediction. Then, it applies the
        produced weight vectors of each dictionary level to the given features
        to compute the BoW prediction logits.

        Args:
        features: list of 2D tensors where each of them is a mini-batch of
            features (extracted from the image crops) with shape
            [(batch_size * num_crops) x num_channels_out] from which the BoW
            prediction head predicts the BoW targets. For example, in the full
            version of OBoW, in which it reconstructs BoW from (a) 2 image crops
            of size [160 x 160] and (b) 5 image patches of size [96 x 96], the
            features argument includes a 2D tensor of shape
            [(batch_size * 2) x num_channels_out] (extracted from the 2
            160x160-sized crops) and a 2D tensor of shape
            [(batch_size * 5) x num_channels_out] (extractted from the 5
            96x96-sized crops).
        dictionary: list of 2D tensors with the visual word embeddings
            (i.e., dictionaries) for each BoW level. So, the i-th item of
            dictionary has shape [num_words x num_channels_in[i]], where
            num_channels_in[i] is the number of channels of the visual word
            embeddings at the i-th BoW level.

        Output:
        logits_list: list of lists of 2D tensors. Specifically, logits_list[i][j]
            contains the 2D tensor of size [(batch_size * num_crops) x num_words]
            with the BoW predictions from features[i] for the j-th BoW level
            (made using the dictionary[j]).
        """
        assert isinstance(dictionary, (list, tuple))
        assert len(dictionary) == len(self.layers_w)

        weight = [gen(dict).t() for gen, dict in zip(self.layers_w, dictionary)]
        kappa = torch.split(self.scale, 1, dim=0)
        logits_list = [
            [torch.mm(f.flatten(1) * k, w) for k, w in zip(kappa, weight)]
            for f in features]

        return logits_list

    def extra_repr(self):
        kappa = self.scale.data
        s = f"(kappa, learnable={kappa.requires_grad}): {kappa.tolist()}"
        return s


class OBoWSolver(solver.Solver):
    def end_of_training_epoch(self):
        if self._epoch == self._start_epoch:
            # In case of distributed training, it checks if all processes have
            # the same parameters.
            utils.sanity_check_for_distributed_training(self.model)

    def start_of_training_epoch(self):
        if self._epoch == 0:
            if utils.is_dist_avail_and_initialized():
                # In case of distributed training, it ensures that alls
                # processes start with the same version of the dictionaries.
                for b in self.model.module.bow_extractor.bow_extractor:
                    b._broadast_initial_dictionary()

        if self._epoch == self._start_epoch:
            # In case of distributed training, it checks if all processes have
            # the same parameters.
            utils.sanity_check_for_distributed_training(self.model)

        if utils.get_rank() == 0:
            model = (
                self.model.module
                if utils.is_dist_avail_and_initialized() else self.model)
            alpha = (
                model._alpha.item()
                if isinstance(model._alpha, torch.Tensor) else model._alpha)
            logger.info(f"alpha: {alpha}")

    def evaluation_step(self, mini_batch, metric_logger):
        if len(mini_batch) == 6:
            return self._process_fewshot(mini_batch, metric_logger)
        else:
            return self.eval_lincls(mini_batch, metric_logger)

    def eval_lincls(self, mini_batch, metric_logger):
        assert len(mini_batch) == 2
        images, labels = mini_batch
        img_orig = images[0]

        if self.device is not None:
            img_orig = img_orig.cuda(self.device , non_blocking=True)
            labels = labels.cuda(self.device , non_blocking=True)

        batch_size = img_orig.size(0)
        with torch.no_grad():
            # Forward model and compute lossses.
            lincls, acc1 = self.model(img_orig, None, labels)
            lincls = lincls.item()

        metric_logger["lincls"].update(lincls, batch_size)
        metric_logger["acc@1"].update(acc1, batch_size)

        return metric_logger

    def train_step(self, mini_batch, metric_logger):
        assert len(mini_batch) == 2
        images, labels = mini_batch

        if self.device is not None:
            images = [img.cuda(self.device , non_blocking=True) for img in images]
            labels = labels.cuda(self.device , non_blocking=True)

        img_orig = images[0]
        img_crops = images[1:]
        for i in range(len(img_crops)):
            if img_crops[i].dim() == 5:
                # [B x C x 3 x H x W] ==> [(B * C) x 3 x H x W]
                img_crops[i] = utils.convert_from_5d_to_4d(img_crops[i])
        batch_size = img_orig.size(0)

        # Forward model and compute lossses.
        losses, logs = self.model(img_orig, img_crops, labels)
        loss_total = losses.sum()

        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        losses = losses.view(-1).tolist()
        num_levels = (len(logs)-1) // 2
        assert len(logs) == (2*num_levels+1)
        assert len(losses) == (len(img_crops) + 1)

        for i in range(len(img_crops)):
            crop_sz = img_crops[i].size(3)
            metric_logger[f"loss_crop{crop_sz}"].update(losses[0], batch_size)

        for i in range(num_levels):
            metric_logger[f"perp_b_lev@{i}"].update(logs[i], batch_size)
            metric_logger[f"perp_i_lev@{i}"].update(logs[i+num_levels], batch_size)

        # Linear classification accuracy.
        metric_logger["linear_acc@1"].update(logs[-1], batch_size)

        return metric_logger

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

    def save_feature_extractor_in_torchvision_format(self, arch="resnet50"):
        import torchvision.models as torchvision_models
        distributed = utils.is_dist_avail_and_initialized()
        model = self.model.module if distributed else self.model
        model.eval()

        dictionary = model.bow_extractor.get_dictionary()
        dictionary_w = [
            model.bow_predictor.layers_w[d](dictionary[d])
            for d in range(len(dictionary))]
        scale = torch.chunk(model.bow_predictor.scale, len(dictionary), dim=0)
        weight = dictionary_w[-1] * scale[-1].item()
        num_words = weight.size(0)
        logger.info('==> Converting and saving the OBoW student resnet '
                    'backbone to torchvision format.')
        torchvision_resnet = torchvision_models.__dict__[arch](
            num_classes=num_words)
        torchvision_resnet.eval()

        logger.info('====> Converting 1st convolutional layer (aka conv1)')
        torchvision_resnet.conv1.load_state_dict(
            model.feature_extractor._feature_blocks[0][0].state_dict())
        torchvision_resnet.bn1.load_state_dict(
            model.feature_extractor._feature_blocks[0][1].state_dict())

        logger.info('====> Converting 1st residual block (aka conv2).')
        torchvision_resnet.layer1.load_state_dict(
            model.feature_extractor._feature_blocks[1].state_dict())

        logger.info('====> Converting 2nd residual block (aka conv3).')
        torchvision_resnet.layer2.load_state_dict(
            model.feature_extractor._feature_blocks[2].state_dict())

        logger.info('====> Converting 3rd residual block (aka conv4).')
        torchvision_resnet.layer3.load_state_dict(
            model.feature_extractor._feature_blocks[3].state_dict())

        logger.info('====> Converting 4th residual block (aka conv5).')
        torchvision_resnet.layer4.load_state_dict(
            model.feature_extractor._feature_blocks[4].state_dict())

        logger.info('====> Converting and fixing the BoW classification '
                    'head for the last BoW level.')
        with torch.no_grad():
            torchvision_resnet.fc.weight.copy_(weight)
            torchvision_resnet.fc.bias.fill_(0)

        epoch = self._epoch
        filename = f"tochvision_{arch}_student_K{num_words}_epoch{epoch}.pth.tar"
        filename = str(self.exp_dir / filename)
        logger.info(f'==> Saving the torchvision resnet model at: {filename}')

        if utils.get_rank() == 0:
            torch.save({'network': torchvision_resnet.state_dict()}, filename)

    def _process_fewshot(self, episode, metric_logger):
        """ Evaluates the OBoW's feature extractor on few-shot classifcation """
        images_train, labels_train, images_test, labels_test, _, _ = episode
        if (self.device is not None):
            images_train = images_train.cuda(self.device , non_blocking=True)
            images_test = images_test.cuda(self.device , non_blocking=True)
            labels_train = labels_train.cuda(self.device , non_blocking=True)
            labels_test = labels_test.cuda(self.device , non_blocking=True)

        nKnovel = 1 + labels_train.max().item()
        labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
        labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
        labels_train_1hot = images_train.new_full(labels_train_1hot_size, 0.0)
        labels_train_1hot.scatter_(
            len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)

        model = (
            self.model.module if
            utils.is_dist_avail_and_initialized() else self.model)

        model.feature_extractor.eval()
        with torch.no_grad():
            _, accuracies = fewshot.fewshot_classification(
                feature_extractor=model.feature_extractor,
                images_train=images_train,
                labels_train=labels_train,
                labels_train_1hot=labels_train_1hot,
                images_test=images_test,
                labels_test=labels_test,
                feature_levels=None)

            accuracies = accuracies.view(-1)
            for i in range(accuracies.numel()):
                metric_logger[f"acc_novel_@{i}"].update(accuracies[i].item(), 1)

        return metric_logger
