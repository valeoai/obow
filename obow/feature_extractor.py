import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import obow.utils as utils


class SequentialFeatureExtractorAbstractClass(nn.Module):
    def __init__(self, all_feat_names, feature_blocks):
        super(SequentialFeatureExtractorAbstractClass, self).__init__()

        assert(isinstance(feature_blocks, list))
        assert(isinstance(all_feat_names, list))
        assert(len(all_feat_names) == len(feature_blocks))

        self.all_feat_names = all_feat_names
        self._feature_blocks = nn.ModuleList(feature_blocks)


    def _parse_out_keys_arg(self, out_feat_keys):
        # By default return the features of the last layer / module.
        out_feat_keys = (
            [self.all_feat_names[-1],] if out_feat_keys is None else
            out_feat_keys)

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')

        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. '
                    'Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError(
                    'Duplicate output feature key: {0}.'.format(key))

    	# Find the highest output feature in `out_feat_keys
        max_out_feat = max(
            [self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def get_subnetwork(self, out_feat_key):
        if isinstance(out_feat_key, str):
            out_feat_key = [out_feat_key,]
        _, max_out_feat = self._parse_out_keys_arg(out_feat_key)
        subnetwork = nn.Sequential()
        for f in range(max_out_feat+1):
            subnetwork.add_module(
                self.all_feat_names[f],
                self._feature_blocks[f]
            )
        return subnetwork

    def forward(self, x, out_feat_keys=None):
        """Forward the image `x` through the network and output the asked features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. If out_feat_keys is None (
                default value) then the last feature of the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat+1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = (out_feats[0] if len(out_feats) == 1 else out_feats)

        return out_feats


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride,
        drop_rate=0.0,
        kernel_size=3):
        super(BasicBlock, self).__init__()

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        assert isinstance(kernel_size, (list, tuple))
        assert len(kernel_size) == 2

        kernel_size1, kernel_size2 = kernel_size

        assert kernel_size1 == 1 or kernel_size1 == 3
        padding1 = 1 if kernel_size1 == 3 else 0
        assert kernel_size2 == 1 or kernel_size2 == 3
        padding2 = 1 if kernel_size2 == 3 else 0


        self.equalInOut = (in_planes == out_planes and stride == 1)

        self.convResidual = nn.Sequential()

        if self.equalInOut:
            self.convResidual.add_module('bn1', nn.BatchNorm2d(in_planes))
            self.convResidual.add_module('relu1', nn.ReLU(inplace=True))

        self.convResidual.add_module(
            'conv1',
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size1,
                stride=stride, padding=padding1, bias=False))

        self.convResidual.add_module('bn2', nn.BatchNorm2d(out_planes))
        self.convResidual.add_module('relu2', nn.ReLU(inplace=True))
        self.convResidual.add_module(
            'conv2',
            nn.Conv2d(
                out_planes, out_planes, kernel_size=kernel_size2,
                stride=1, padding=padding2, bias=False))

        if drop_rate > 0:
            self.convResidual.add_module('dropout', nn.Dropout(p=drop_rate))

        if self.equalInOut:
            self.convShortcut = nn.Sequential()
        else:
            self.convShortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        return self.convShortcut(x) + self.convResidual(x)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, drop_rate):

        layers = []
        for i in range(nb_layers):
            in_planes_arg = i == 0 and in_planes or out_planes
            stride_arg = i == 0 and stride or 1
            layers.append(
                block(in_planes_arg, out_planes, stride_arg, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(SequentialFeatureExtractorAbstractClass):
    def __init__(
        self,
        depth,
        widen_factor=1,
        drop_rate=0.0,
        strides=[2, 2, 2],
        global_pooling=True):

        assert (depth - 4) % 6 == 0
        num_channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        num_layers = [int((depth - 4) / 6) for _ in range(3)]

        block = BasicBlock

        all_feat_names = []
        feature_blocks = []

        # 1st conv before any network block
        conv1 = nn.Sequential()
        conv1.add_module(
            'Conv',
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1, bias=False))
        conv1.add_module('BN', nn.BatchNorm2d(num_channels[0]))
        conv1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(conv1)
        all_feat_names.append('conv1')

        # 1st block.
        block1 = nn.Sequential()
        block1.add_module(
            'Block',
            NetworkBlock(
                num_layers[0], num_channels[0], num_channels[1], BasicBlock,
                strides[0], drop_rate))
        block1.add_module('BN', nn.BatchNorm2d(num_channels[1]))
        block1.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block1)
        all_feat_names.append('block1')

        # 2nd block.
        block2 = nn.Sequential()
        block2.add_module(
            'Block',
            NetworkBlock(
                num_layers[1], num_channels[1], num_channels[2], BasicBlock,
                strides[1], drop_rate))
        block2.add_module('BN', nn.BatchNorm2d(num_channels[2]))
        block2.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block2)
        all_feat_names.append('block2')

        # 3rd block.
        block3 = nn.Sequential()
        block3.add_module(
            'Block',
            NetworkBlock(
                num_layers[2], num_channels[2], num_channels[3], BasicBlock,
                strides[2], drop_rate))
        block3.add_module('BN', nn.BatchNorm2d(num_channels[3]))
        block3.add_module('ReLU', nn.ReLU(inplace=True))
        feature_blocks.append(block3)
        all_feat_names.append('block3')

        # global average pooling.
        if global_pooling:
            feature_blocks.append(utils.GlobalPooling(type="avg"))
            all_feat_names.append('GlobalPooling')

        super(WideResNet, self).__init__(all_feat_names, feature_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet(SequentialFeatureExtractorAbstractClass):
    def __init__(self, arch, pretrained=False, global_pooling=True):
        net = models.__dict__[arch](num_classes=1000, pretrained=pretrained)
        print(f'==> Pretrained parameters: {pretrained}')
        all_feat_names = []
        feature_blocks = []

        # 1st conv before any resnet block
        conv1 = nn.Sequential()
        conv1.add_module('Conv', net.conv1)
        conv1.add_module('bn', net.bn1)
        conv1.add_module('relu', net.relu)
        conv1.add_module('maxpool', net.maxpool)
        feature_blocks.append(conv1)
        all_feat_names.append('conv1')

        # 1st block.
        feature_blocks.append(net.layer1)
        all_feat_names.append('block1')
        # 2nd block.
        feature_blocks.append(net.layer2)
        all_feat_names.append('block2')
        # 3rd block.
        feature_blocks.append(net.layer3)
        all_feat_names.append('block3')
        # 4th block.
        feature_blocks.append(net.layer4)
        all_feat_names.append('block4')
        # global average pooling.
        if global_pooling:
            feature_blocks.append(utils.GlobalPooling(type="avg"))
            all_feat_names.append('GlobalPooling')

        super(ResNet, self).__init__(all_feat_names, feature_blocks)
        self.num_channels = net.fc.in_features


def FeatureExtractor(arch, opts):
    all_architectures = (
        'wrn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext101_32x8d', 'resnext50_32x4d', 'wide_resnet101_2',
        'wide_resnet50_2')

    assert arch in all_architectures
    if arch == 'wrn':
        num_channels = opts["widen_factor"] * 64
        return WideResNet(**opts), num_channels
    else:
        resnet_extractor = ResNet(arch=arch, **opts)
        return resnet_extractor, resnet_extractor.num_channels
