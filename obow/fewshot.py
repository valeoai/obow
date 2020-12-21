from __future__ import print_function

import torch
import torch.nn.functional as F
import obow.utils as utils


def preprocess_5D_features(features, global_pooling):
    meta_batch_size, num_examples, channels, height, width = features.size()
    features = features.view(
        meta_batch_size * num_examples, channels, height, width)

    if global_pooling:
        features = utils.global_pooling(features, "avg")

    features = features.view(meta_batch_size, num_examples, -1)

    return features


def average_train_features(features_train, labels_train):
    labels_train_transposed = labels_train.transpose(1,2)
    weight_novel = torch.bmm(labels_train_transposed, features_train)
    weight_novel = weight_novel.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(
            weight_novel))

    return weight_novel


def few_shot_classifier_with_prototypes(
    features_test, features_train, labels_train,
    scale_cls=10.0, global_pooling=True):

    #******* Generate classification weights for the novel categories ******
    if features_train.dim() == 5:
        features_train = preprocess_5D_features(features_train, global_pooling)
        features_test = preprocess_5D_features(features_test, global_pooling)

    assert features_train.dim() == 3
    assert features_test.dim() == 3

    meta_batch_size = features_train.size(0)
    num_novel = labels_train.size(2)
    features_train = F.normalize(features_train, p=2, dim=2)
    prototypes = average_train_features(features_train, labels_train)
    prototypes = prototypes.view(meta_batch_size, num_novel, -1)
    #***********************************************************************
    features_test = F.normalize(features_test, p=2, dim=2)
    prototypes = F.normalize(prototypes, p=2, dim=2)
    scores = scale_cls * torch.bmm(features_test, prototypes.transpose(1,2))

    return scores


def few_shot_feature_classification(
    classifier, features_test, features_train, labels_train_1hot, labels_test):

    scores = few_shot_classifier_with_prototypes(
        features_test=features_test,
        features_train=features_train,
        labels_train=labels_train_1hot)

    assert scores.dim() == 3

    scores = scores.view(scores.size(0) * scores.size(1), -1)
    labels_test = labels_test.view(-1)
    assert scores.size(0) == labels_test.size(0)

    loss = F.cross_entropy(scores, labels_test)

    with torch.no_grad():
        accuracy = utils.accuracy(scores, labels_test, topk=(1,))

    return scores, loss, accuracy


@torch.no_grad()
def fewshot_classification(
    feature_extractor,
    images_train,
    labels_train,
    labels_train_1hot,
    images_test,
    labels_test,
    feature_levels):
    assert images_train.dim() == 5
    assert images_test.dim() == 5
    assert images_train.size(0) == images_test.size(0)
    assert images_train.size(2) == images_test.size(2)
    assert images_train.size(3) == images_test.size(3)
    assert images_train.size(4) == images_test.size(4)
    assert labels_train.dim() == 2
    assert labels_test.dim() == 2
    assert labels_train.size(0) == labels_test.size(0)
    assert labels_train.size(0) == images_train.size(0)
    assert (feature_levels is None) or isinstance(feature_levels, (list, tuple))
    meta_batch_size = images_train.size(0)

    images_train = utils.convert_from_5d_to_4d(images_train)
    images_test = utils.convert_from_5d_to_4d(images_test)
    labels_test = labels_test.view(-1)
    batch_size_train = images_train.size(0)
    images = torch.cat([images_train, images_test], dim=0)

    # Extract features from the train and test images.
    features = feature_extractor(images, feature_levels)
    if isinstance(features, torch.Tensor):
        features = [features,]

    labels_test =labels_test.view(-1)

    loss, accuracy = [], []
    for i, features_i in enumerate(features):
        features_train = features_i[:batch_size_train]
        features_test = features_i[batch_size_train:]
        features_train = utils.add_dimension(features_train, meta_batch_size)
        features_test = utils.add_dimension(features_test, meta_batch_size)

        scores = few_shot_classifier_with_prototypes(
            features_test, features_train, labels_train_1hot,
            scale_cls=10.0, global_pooling=True)

        scores = scores.view(scores.size(0) * scores.size(1), -1)
        assert scores.size(0) == labels_test.size(0)
        loss.append(F.cross_entropy(scores, labels_test))
        with torch.no_grad():
            accuracy.append(utils.accuracy(scores, labels_test, topk=(1,))[0])

    loss = torch.stack(loss, dim=0)
    accuracy = torch.stack(accuracy, dim=0)

    return loss, accuracy
