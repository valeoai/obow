from __future__ import print_function

import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.datasets
import random
import json

from PIL import ImageFilter


_MEAN_PIXEL_IMAGENET = [0.485, 0.456, 0.406]
_STD_PIXEL_IMAGENET = [0.229, 0.224, 0.225]


def generate_element_list(list_size, dataset_size):
    if list_size == dataset_size:
        return list(range(dataset_size))
    elif list_size < dataset_size:
        return np.random.choice(
            dataset_size, list_size, replace=False).tolist()
    else: # list_size > list_size
        num_times = list_size // dataset_size
        residual = list_size % dataset_size
        assert((num_times * dataset_size + residual) == list_size)
        elem_list = list(range(dataset_size)) * num_times
        if residual:
            elem_list += np.random.choice(
                dataset_size, residual, replace=False).tolist()

        return elem_list


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class ParallelTransforms:
    def __init__(self, transform_list):
        assert isinstance(transform_list, (list, tuple))
        self.transform_list = transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.transform_list]

    def __str__(self):
        str_transforms = f"ParallelTransforms(["
        for i, transform in enumerate(self.transform_list):
            str_transforms += f"\nTransform #{i}:\n{transform}, "
        str_transforms += "\n])"
        return str_transforms


class StackMultipleViews:
    def __init__(self, transform, num_views):
        assert num_views >= 1
        self.transform = transform
        self.num_views = num_views

    def __call__(self, x):
        if self.num_views == 1:
            return self.transform(x).unsqueeze(dim=0)
        else:
            x_views = [self.transform(x) for _ in range(self.num_views)]
            return torch.stack(x_views, dim=0)

    def __str__(self):
        str_transforms = f"StackMultipleViews({self.num_views} x \n{self.transform})"
        return str_transforms


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __str__(self):
        str_transforms = f"GaussianBlur(sigma={self.sigma})"
        return str_transforms


class CropImagePatches:
    """Crops from an image 3 x 3 overlapping patches."""
    def __init__(
        self,
        patch_size=96,
        patch_jitter=24,
        num_patches=5,
        split_per_side=3):

        self.split_per_side = split_per_side
        self.patch_size = patch_size
        assert patch_jitter >= 0
        self.patch_jitter = patch_jitter
        if num_patches is None:
            num_patches = split_per_side**2
        assert num_patches > 0 and num_patches <= (split_per_side**2)
        self.num_patches = num_patches

    def __call__(self, img):
        _, height, width = img.size()
        offset_y = ((height - self.patch_size - self.patch_jitter)
                    // (self.split_per_side - 1))
        offset_x = ((width - self.patch_size - self.patch_jitter)
                    // (self.split_per_side - 1))

        patches = []
        for i in range(self.split_per_side):
            for j in range(self.split_per_side):
                y_top = i * offset_y + random.randint(0, self.patch_jitter)
                x_left = j * offset_x + random.randint(0, self.patch_jitter)
                y_bottom = y_top + self.patch_size
                x_right = x_left + self.patch_size
                patches.append(img[:, y_top:y_bottom, x_left:x_right])

        if self.num_patches < (self.split_per_side * self.split_per_side):
            indices = torch.randperm(len(patches))[:self.num_patches]
            patches = [patches[i] for i in indices.tolist()]

        return torch.stack(patches, dim=0)

    def __str__(self):
        print_str = (
            f'{self.__class__.__name__}('
            f'split_per_side={self.split_per_side}, '
            f'patch_size={self.patch_size}, '
            f'patch_jitter={self.patch_jitter}, '
            f'num_patches={self.num_patches}/{self.split_per_side**2})'
        )
        return print_str


def subset_of_ImageNet_train_split(dataset_train, subset):
    assert isinstance(subset, int)
    assert subset > 0

    all_indices = []
    for _, img_indices in buildLabelIndex(dataset_train.targets).items():
        assert len(img_indices) >= subset
        all_indices += img_indices[:subset]

    dataset_train.imgs = [dataset_train.imgs[idx] for idx in all_indices]
    dataset_train.samples = [dataset_train.samples[idx] for idx in all_indices]
    dataset_train.targets = [dataset_train.targets[idx] for idx in all_indices]
    assert len(dataset_train) == (subset * 1000)

    return dataset_train


def get_ImageNet_data_for_obow(
    data_dir,
    subset=None,
    cjitter=[0.4, 0.4, 0.4, 0.1],
    cjitter_p=0.8,
    gray_p=0.2,
    gaussian_blur=[0.1, 2.0],
    gaussian_blur_p=0.5,
    num_img_crops=2,
    image_crop_size=160,
    image_crop_range=[0.08, 0.6],
    num_img_patches=0,
    img_patch_preresize=256,
    img_patch_preresize_range=[0.6, 1.0],
    img_patch_size=96,
    img_patch_jitter=24,
    only_patches=False):

    normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)

    image_crops_transform = T.Compose([
        T.RandomResizedCrop(image_crop_size, scale=image_crop_range),
        T.RandomApply([T.ColorJitter(*cjitter)], p=cjitter_p),
        T.RandomGrayscale(p=gray_p),
        T.RandomApply([GaussianBlur(gaussian_blur)], p=gaussian_blur_p),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    image_crops_transform = StackMultipleViews(
        image_crops_transform, num_views=num_img_crops)

    transform_original_train = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(), # So as, to see both image views.
        T.ToTensor(),
        normalize,
    ])
    transform_train = [transform_original_train, image_crops_transform]

    if num_img_patches > 0:
        assert num_img_patches <= 9
        image_patch_transform = T.Compose([
            T.RandomResizedCrop(img_patch_preresize, scale=img_patch_preresize_range),
            T.RandomApply([T.ColorJitter(*cjitter)], p=cjitter_p),
            T.RandomGrayscale(p=gray_p),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            CropImagePatches(
                patch_size=img_patch_size, patch_jitter=img_patch_jitter,
                num_patches=num_img_patches, split_per_side=3),
        ])
        if only_patches:
            transform_train[-1] = image_patch_transform
        else:
            transform_train.append(image_patch_transform)

    transform_train = ParallelTransforms(transform_train)

    transform_original = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])
    transform_test = ParallelTransforms([transform_original,])

    print(f"Image transforms during training: {transform_train}")
    print(f"Image transforms during testing: {transform_test}")

    print("Loading data.")
    dataset_train = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transform_train)
    dataset_test = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=transform_test)

    if (subset is not None) and (subset >= 1):
        dataset_train = subset_of_ImageNet_train_split(dataset_train, subset)

    return dataset_train, dataset_test


def get_data_loaders_for_OBoW(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    distributed,
    epoch_size,
    **kwargs):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert isinstance(distributed, bool)
    assert (epoch_size is None) or isinstance(epoch_size, int)

    if dataset_name == "ImageNet":
        dataset_train, dataset_test = get_ImageNet_data_for_obow(data_dir, **kwargs)
    else:
        raise NotImplementedError(f"Not supported dataset {dataset_name}")

    if (epoch_size is not None) and (epoch_size != len(dataset_train)):
        elem_list = generate_element_list(epoch_size, len(dataset_train))
        dataset_train = torch.utils.data.Subset(dataset_train, elem_list)

    print("Creating data loaders")
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_test,
        drop_last=False)

    return (
        loader_train, sampler_train, dataset_train,
        loader_test, sampler_test, dataset_test)


#**************** DATA LOADERS FOR IMAGE CLASSIFICATIONS ***********************
def get_ImageNet_data_classification(data_dir, subset=None):
    normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)
    transform_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print("Loading data.")
    dataset_train = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transform_train)
    dataset_test = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=transform_test)

    if (subset is not None) and (subset >= 1):
        dataset_train = subset_of_ImageNet_train_split(dataset_train, subset)

    return dataset_train, dataset_test


def get_ImageNet_data_semisupervised_classification(data_dir, percentage=1):
    normalize = T.Normalize(
        mean=_MEAN_PIXEL_IMAGENET,
        std=_STD_PIXEL_IMAGENET
    )
    transform_train = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print("Loading data.")
    train_data_path = os.path.join(data_dir, "train")
    dataset_train = torchvision.datasets.ImageFolder(
        train_data_path, transform=transform_train)
    dataset_test = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=transform_test)

    # take either 1% or 10% of images
    assert percentage in (1, 10)
    import urllib.request
    BASE_URL_PATH = "https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/"
    subset_file = urllib.request.urlopen(
        BASE_URL_PATH + str(percentage) + "percent.txt")
    list_imgs = [li.decode("utf-8").split('\n')[0] for li in subset_file]

    samples, imgs, targets = [], [], []
    for file in list_imgs:
        file_path = pathlib.Path(os.path.join(train_data_path, file.split('_')[0], file))
        assert file_path.is_file()
        file_path_str = str(file_path)
        target = dataset_train.class_to_idx[file.split('_')[0]]
        imgs.append((file_path_str, target))
        targets.append(targets)
        samples.append((file_path_str, target))

    dataset_train.imgs = imgs
    dataset_train.targets = targets
    dataset_train.samples = samples

    assert len(dataset_train) == len(list_imgs)

    return dataset_train, dataset_test


def get_Places205_data_classification(data_dir):
    normalize = T.Normalize(
        mean=_MEAN_PIXEL_IMAGENET,
        std=_STD_PIXEL_IMAGENET
    )
    transform_train = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    transform_test = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print("Loading data.")
    dataset_train = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=transform_train)
    dataset_test = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=transform_test)

    return dataset_train, dataset_test


def get_data_loaders_classification(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    distributed,
    epoch_size,
    **kwargs):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert isinstance(distributed, bool)
    assert (epoch_size is None) or isinstance(epoch_size, int)

    if dataset_name == "ImageNet":
        dataset_train, dataset_test = get_ImageNet_data_classification(
            data_dir, **kwargs)
    elif dataset_name == "Places205":
        dataset_train, dataset_test = get_Places205_data_classification(
            data_dir)
    else:
        raise NotImplementedError(f"Not supported dataset {dataset_name}")

    if (epoch_size is not None) and (epoch_size != len(dataset_train)):
        elem_list = generate_element_list(epoch_size, len(dataset_train))
        dataset_train = torch.utils.data.Subset(dataset_train, elem_list)

    print("Creating data loaders")
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_test,
        drop_last=False)

    return (
        loader_train, sampler_train, dataset_train,
        loader_test, sampler_test, dataset_test)


def get_data_loaders_semisupervised_classification(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    distributed,
    epoch_size,
    percentage,
    **kwargs):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert isinstance(distributed, bool)
    assert (epoch_size is None) or isinstance(epoch_size, int)

    if dataset_name == "ImageNet":
        dataset_train, dataset_test = get_ImageNet_data_semisupervised_classification(
            data_dir, percentage=percentage, **kwargs)
    else:
        raise NotImplementedError(f"Not supported dataset {dataset_name}")

    if (epoch_size is not None) and (epoch_size != len(dataset_train)):
        elem_list = generate_element_list(epoch_size, len(dataset_train))
        dataset_train = torch.utils.data.Subset(dataset_train, elem_list)

    print("Creating data loaders")
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            dataset_train)
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_test,
        drop_last=False)

    return (
        loader_train, sampler_train, dataset_train,
        loader_test, sampler_test, dataset_test)


#*************** DATA LOADERS FOR FEW-SHOT EVALUATION **************************
def load_ImageNet_fewshot_split(class_names, version=1):
    _IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH = (
        './data/IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS.json')
    with open(_IMAGENET_LOWSHOT_BENCHMARK_CATEGORY_SPLITS_PATH, 'r') as f:
        label_idx = json.load(f)

    assert len(label_idx['label_names']) == len(class_names)

    def get_class_indices(class_indices1):
        class_indices2 = []
        for index in class_indices1:
            class_name_this = label_idx['label_names'][index]
            assert class_name_this in class_names
            class_indices2.append(class_names.index(class_name_this))

        class_names_tmp1 = [
            label_idx['label_names'][index] for index in class_indices1]
        class_names_tmp2 = [class_names[index] for index in class_indices2]

        assert class_names_tmp1 == class_names_tmp2

        return class_indices2

    if version == 1:
        base_classes = label_idx['base_classes']
        base_classes_val = label_idx['base_classes_1']
        base_classes_test = label_idx['base_classes_2']
        novel_classes_val = label_idx['novel_classes_1']
        novel_classes_test = label_idx['novel_classes_2']
    elif version == 2:
        base_classes = get_class_indices(label_idx['base_classes'])
        base_classes_val = get_class_indices(label_idx['base_classes_1'])
        base_classes_test = get_class_indices(label_idx['base_classes_2'])
        novel_classes_val = get_class_indices(label_idx['novel_classes_1'])
        novel_classes_test = get_class_indices(label_idx['novel_classes_2'])

    return (base_classes,
            base_classes_val, base_classes_test,
            novel_classes_val, novel_classes_test)


class ImageNetLowShot:
    def __init__(self, dataset, phase='train'):
        assert phase in ('train', 'test', 'val')
        self.data = dataset
        self.phase = phase
        print(f'Loading ImageNet dataset (few-shot benchmark) - phase {phase}')
        #***********************************************************************
        (base_classes, _, _, novel_classes_val, novel_classes_test) = (
            load_ImageNet_fewshot_split(self.data.classes, version=1))
        #***********************************************************************

        self.labels = [item[1] for item in self.data.imgs]
        self.label2ind = buildLabelIndex(self.labels)
        self.labelIds = sorted(self.label2ind.keys())
        self.num_cats = len(self.labelIds)
        assert self.num_cats == 1000

        self.labelIds_base = base_classes
        self.num_cats_base = len(self.labelIds_base)
        if self.phase=='val' or self.phase=='test':
            self.labelIds_novel = (
                novel_classes_val if (self.phase=='val') else
                novel_classes_test)
            self.num_cats_novel = len(self.labelIds_novel)

            intersection = set(self.labelIds_base) & set(self.labelIds_novel)
            assert len(intersection) == 0

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class FewShotDataset:
    def __init__(
        self,
        dataset,
        nKnovel=5,
        nKbase=0,
        nExemplars=1,
        nTestNovel=15*5,
        nTestBase=0,
        epoch_size=500):

        self.dataset = dataset
        self.phase = self.dataset.phase
        max_possible_nKnovel = (
            self.dataset.num_cats_base if (
                self.phase=='train' or self.phase=='trainval')
            else self.dataset.num_cats_novel)

        assert 0 <= nKnovel <= max_possible_nKnovel
        self.nKnovel = nKnovel

        max_possible_nKbase = self.dataset.num_cats_base
        nKbase = nKbase if nKbase >= 0 else max_possible_nKbase
        if (self.phase=='train' or self.phase=='trainval') and nKbase > 0:
            nKbase -= self.nKnovel
            max_possible_nKbase -= self.nKnovel

        assert 0 <= nKbase <= max_possible_nKbase
        self.nKbase = nKbase
        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.nTestBase = nTestBase
        self.epoch_size = epoch_size
        self.is_eval_mode = (self.phase=='test') or (self.phase=='val')

        # remeber this state
        state = random.getstate()
        np_state = np.random.get_state()

        random.seed(0)
        np.random.seed(0)
        self._all_episodes = []
        for i in range(self.epoch_size):
            Exemplars, Test, Kall, nKbase = self.sample_episode()
            self._all_episodes.append((Exemplars, Test, Kall, nKbase))

        # restore state
        random.setstate(state)
        np.random.set_state(np_state)


    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
        """
        assert(cat_id in self.dataset.label2ind.keys())
        assert(len(self.dataset.label2ind[cat_id]) >= sample_size)
        # Note: random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id], sample_size)

    def sampleCategories(self, cat_set, sample_size=1):
        """
        Samples `sample_size` number of unique categories picked from the
        `cat_set` set of categories. `cat_set` can be either 'base' or 'novel'.

        Args:
            cat_set: string that specifies the set of categories from which
                categories will be sampled.
            sample_size: number of categories that will be sampled.

        Returns:
            cat_ids: a list of length `sample_size` with unique category ids.
        """
        if cat_set=='base':
            labelIds = self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds = self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognized category set {}'.format(cat_set))

        assert(len(labelIds) >= sample_size)
        # return sample_size unique categories chosen from labelIds set of
        # categories (that can be either self.labelIds_base or self.labelIds_novel)
        # Note: random.sample samples elements without replacement.
        return random.sample(labelIds, sample_size)

    def sample_base_and_novel_categories(self, nKbase, nKnovel):
        """
        Samples `nKbase` number of base categories and `nKnovel` number of novel
        categories.

        Args:
            nKbase: number of base categories
            nKnovel: number of novel categories

        Returns:
            Kbase: a list of length 'nKbase' with the ids of the sampled base
                categories.
            Knovel: a list of lenght 'nKnovel' with the ids of the sampled novel
                categories.
        """
        if self.is_eval_mode:
            assert(nKnovel <= self.dataset.num_cats_novel)
            # sample from the set of base categories 'nKbase' number of base
            # categories.
            Kbase = sorted(self.sampleCategories('base', nKbase))
            # sample from the set of novel categories 'nKnovel' number of novel
            # categories.
            Knovel = sorted(self.sampleCategories('novel', nKnovel))
        else:
            # sample from the set of base categories 'nKnovel' + 'nKbase' number
            # of categories.
            cats_ids = self.sampleCategories('base', nKnovel+nKbase)
            assert(len(cats_ids) == (nKnovel+nKbase))
            # Randomly pick 'nKnovel' number of fake novel categories and keep
            # the rest as base categories.
            random.shuffle(cats_ids)
            Knovel = sorted(cats_ids[:nKnovel])
            Kbase = sorted(cats_ids[nKnovel:])

        return Kbase, Knovel

    def sample_test_examples_for_base_categories(self, Kbase, nTestBase):
        """
        Sample `nTestBase` number of images from the `Kbase` categories.

        Args:
            Kbase: a list of length `nKbase` with the ids of the categories from
                where the images will be sampled.
            nTestBase: the total number of images that will be sampled.

        Returns:
            Tbase: a list of length `nTestBase` with 2-element tuples. The 1st
                element of each tuple is the image id that was sampled and the
                2nd elemend is its category label (which is in the range
                [0, len(Kbase)-1]).
        """
        Tbase = []
        if len(Kbase) > 0:
            # Sample for each base category a number images such that the total
            # number sampled images of all categories to be equal to `nTestBase`.
            KbaseIndices = np.random.choice(
                np.arange(len(Kbase)), size=nTestBase, replace=True)
            KbaseIndices, NumImagesPerCategory = np.unique(
                KbaseIndices, return_counts=True)
            for Kbase_idx, NumImages in zip(KbaseIndices, NumImagesPerCategory):
                imd_ids = self.sampleImageIdsFrom(
                    Kbase[Kbase_idx], sample_size=NumImages)
                Tbase += [(img_id, Kbase_idx) for img_id in imd_ids]

        assert len(Tbase) == nTestBase

        return Tbase

    def sample_train_and_test_examples_for_novel_categories(
            self, Knovel, nTestExamplesTotal, nExemplars, nKbase):
        """Samples train and test examples of the novel categories.

        Args:
    	    Knovel: a list with the ids of the novel categories.
            nTestExamplesTotal: the total number of test images that will be sampled
                from all the novel categories.
            nExemplars: the number of training examples per novel category that
                will be sampled.
            nKbase: the number of base categories. It is used as offset of the
                category index of each sampled image.

        Returns:
            Tnovel: a list of length `nTestNovel` with 2-element tuples. The
                1st element of each tuple is the image id that was sampled and
                the 2nd element is its category label (which is in the range
                [nKbase, nKbase + len(Knovel) - 1]).
            Exemplars: a list of length len(Knovel) * nExemplars of 2-element
                tuples. The 1st element of each tuple is the image id that was
                sampled and the 2nd element is its category label (which is in
                the ragne [nKbase, nKbase + len(Knovel) - 1]).
        """

        if len(Knovel) == 0:
            return [], []

        nKnovel = len(Knovel)
        Tnovel = []
        Exemplars = []

        assert (nTestExamplesTotal % nKnovel) == 0
        nTestExamples = nTestExamplesTotal // nKnovel

        for Knovel_idx in range(len(Knovel)):
            img_ids = self.sampleImageIdsFrom(
                Knovel[Knovel_idx],
                sample_size=(nTestExamples + nExemplars))

            img_labeled = img_ids[:(nTestExamples + nExemplars)]
            img_tnovel = img_labeled[:nTestExamples]
            img_exemplars = img_labeled[nTestExamples:]

            Tnovel += [
                (img_id, nKbase+Knovel_idx) for img_id in img_tnovel]
            Exemplars += [
                (img_id, nKbase+Knovel_idx) for img_id in img_exemplars]

        assert len(Tnovel) == nTestExamplesTotal
        assert len(Exemplars) == len(Knovel) * nExemplars
        random.shuffle(Exemplars)

        return Tnovel, Exemplars

    def sample_episode(self):
        """Samples a training episode."""
        nKnovel = self.nKnovel
        nKbase = self.nKbase
        nTestNovel = self.nTestNovel
        nTestBase = self.nTestBase
        nExemplars = self.nExemplars

        Kbase, Knovel = self.sample_base_and_novel_categories(nKbase, nKnovel)
        Tbase = self.sample_test_examples_for_base_categories(Kbase, nTestBase)
        Tnovel, Exemplars = self.sample_train_and_test_examples_for_novel_categories(
            Knovel, nTestNovel, nExemplars, nKbase)

        # concatenate the base and novel category examples.
        Test = Tbase + Tnovel
        random.shuffle(Test)
        Kall = Kbase + Knovel
        return Exemplars, Test, Kall, nKbase


    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """
        images = torch.stack(
            [self.dataset[img_idx][0] for img_idx, _ in examples],
            dim=0)
        labels = torch.LongTensor(
            [label for _, label in examples])
        return images, labels

    def __getitem__(self, index):
        Exemplars, Test, Kall, nKbase = self._all_episodes[index]
        Xt, Yt = self.createExamplesTensorData(Test)
        Kall = torch.LongTensor(Kall)
        if len(Exemplars) > 0:
            Xe, Ye = self.createExamplesTensorData(Exemplars)
            return Xe, Ye, Xt, Yt, Kall, nKbase
        else:
            return Xt, Yt, Kall, nKbase

    def __len__(self):
        return self.epoch_size // self.batch_size


def get_ImageNet_fewshot_data(data_dir, split):
    normalize = T.Normalize(
        mean=_MEAN_PIXEL_IMAGENET,
        std=_STD_PIXEL_IMAGENET
    )
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    print("Loading data.")
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=transform)

    assert split in ("train", "val", "test")
    dataset = ImageNetLowShot(dataset, phase=split)

    return dataset


def get_data_loaders_fewshot(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    distributed,
    split='val',
    epoch_size=500,
    num_novel=5,
    num_train=[1,5,10],
    num_test=15*5):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert isinstance(distributed, bool)

    if dataset_name == "ImageNet":
        dataset = get_ImageNet_fewshot_data(data_dir, split=split)
    else:
        raise NotImplementedError(f"Not supported dataset {dataset_name}.")

    assert isinstance(epoch_size, int)
    sampler = torch.utils.data.SubsetRandomSampler(list(range(epoch_size)))

    if isinstance(num_train, int):
        num_train = [num_train,]

    dataset_fewshot = []
    loader_fewshot = []
    for num_train_this in num_train:
        dataset_fewshot_this = FewShotDataset(
            dataset=dataset,
            nKnovel=num_novel,
            nKbase=0,
            nExemplars=num_train_this,
            nTestNovel=num_test,
            nTestBase=0,
            epoch_size=epoch_size)

        loader_fewshot_this = torch.utils.data.DataLoader(
            dataset_fewshot_this,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=False,
            sampler=sampler,
            drop_last=False)
        dataset_fewshot.append(dataset_fewshot_this)
        loader_fewshot.append(loader_fewshot_this)

    return loader_fewshot, sampler, dataset_fewshot


#*******************************************************************************
def get_ImageNet_data_for_visualization(data_dir, subset=None, split="train"):
    normalize = T.Normalize(
        mean=_MEAN_PIXEL_IMAGENET,
        std=_STD_PIXEL_IMAGENET
    )
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_dir, split), transform=transform)

    if (split == "train") and subset is not None:
        assert isinstance(subset, int)
        assert subset > 0

        all_indices = []
        for _, img_indices in buildLabelIndex(dataset.targets).items():
            assert len(img_indices) >= subset
            all_indices += img_indices[:subset]

        dataset.imgs = [dataset.imgs[idx] for idx in all_indices]
        dataset.samples = [dataset.samples[idx] for idx in all_indices]
        dataset.targets = [dataset.targets[idx] for idx in all_indices]
        assert len(dataset) == (subset * 1000)

    return dataset


def get_data_loaders_for_visualization(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    distributed,
    split,
    **kwargs):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert isinstance(distributed, bool)

    if dataset_name == "ImageNet":
        subset = kwargs.pop("subset", None)
        dataset = get_ImageNet_data_for_visualization(
            data_dir, split=split, subset=subset)
    else:
        raise NotImplementedError(f"Not supported dataset {dataset_name}.")

    print("Creating data loaders")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=torch.utils.data.SequentialSampler(dataset),
        drop_last=False)


    return loader, dataset
#*******************************************************************************


#*******************************************************************************
# Code for pre-caching the features of the linear classifier.
#*******************************************************************************
import pathlib
from tqdm import tqdm

NUM_VIEWS = 10
CENTRAL_VIEW = 4
COMMON_NP_TYPES = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64
]

def str_to_dtype(string):
    for dtype in COMMON_NP_TYPES:
        if dtype.__name__ == string:
            return dtype
    raise KeyError


def create_memmap(dir_path, dtype, shape):
    dir_path = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    def dtype_to_str(dtype):
        return dtype.__name__

    metadata_dict = dict(dtype=dtype_to_str(dtype), shape=list(shape), count=0)
    metadata_file = dir_path / 'metadata.json'

    def update_metadata(count):
        metadata_dict['count'] = count
        print(f'Update metadata with metadata_dict={metadata_dict}')
        metadata_file.write_text(json.dumps(metadata_dict, indent=4))

    update_metadata(count=0)
    memmap_file = dir_path / 'memmap.npy'
    memmap = np.memmap(memmap_file, dtype, mode='w+', shape=shape)
    return memmap, update_metadata


def open_memmap(dir_path):
    dir_path = pathlib.Path(dir_path)
    metadata_dict = json.loads((dir_path / 'metadata.json').read_text())
    dtype = str_to_dtype(metadata_dict['dtype'])
    shape = tuple(metadata_dict['shape'])
    #count = metadata_dict['count']
    return np.memmap(dir_path / 'memmap.npy', dtype, 'r+', shape=shape)


class ExtractCropsDataset:
    def __init__(self, dataset, init_size, crop_size, mean, std, five_crop=True):
        self.data = dataset
        self.init_size = init_size
        self.crop_size = crop_size
        self.five_crop = five_crop

        if five_crop:
            self.crop = T.Compose([
                T.Resize(self.init_size),
                T.FiveCrop((self.crop_size, self.crop_size)),
            ])
        else:
            self.crop = T.Compose([
                T.Resize(self.init_size),
                T.CenterCrop(self.crop_size),
            ])

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __repr__(self):
        suffix = f'(\nfive_crop={self.crop}, normalize={self.normalize}\n)'
        return self.__class__.__name__ + suffix

    def __getitem__(self, index):
        img, labels = self.data[index]
        crops = self.crop(img)
        if self.five_crop:
            crops = torch.stack([self.normalize(x) for x in crops], dim=0)
        else:
            crops = self.normalize(crops).unsqueeze(0)

        assert crops.dim() == 4
        assert (
            (self.five_crop  and crops.size(0) == 5) or
            (not self.five_crop and crops.size(0) == 1))
        assert crops.size(1) == 3
        assert crops.size(2) == self.crop_size
        assert crops.size(3) == self.crop_size

        return crops, labels

    def __len__(self):
        return len(self.data)


def make_memmap_crops(
    memmap_path,
    feature_extractor,
    dataset,
    device,
    num_workers,
    batch_size,
    num_views=5):

    feature_extractor = feature_extractor.cuda()
    feature_extractor.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    update_metadata_step = 100
    memmap = None
    count = 0
    num_imgs = len(dataset)
    num_views *= 2
    num_features = len(dataset) * num_views

    with torch.no_grad():
        for i, (crops, _) in enumerate(tqdm(dataloader)):
            crops = crops.cuda(device, non_blocking=True)
            assert crops.dim() == 5
            # Add crop flips.
            crops = torch.cat([crops, torch.flip(crops, dims=(4,))], dim=1)
            assert crops.size(1) == num_views
            batch_size_x_num_views = crops.size(0) * num_views
            crops = crops.view([batch_size_x_num_views, ] +  list(crops.size()[2:]))
            features = feature_extractor(crops)
            features_np = features.detach().cpu().numpy()

            if memmap is None:
                memmap_shape = (num_features,) + features_np.shape[1:]
                print(f'Creating dataset of size: {memmap_shape}')
                memmap, update_metadata = create_memmap(
                    memmap_path, np.float32, memmap_shape)

            memmap[count:(count+batch_size_x_num_views)] = features_np
            count += batch_size_x_num_views

            if ((i+1) % update_metadata_step) == 0:
                # Update metadata every update_metadata_step mini-batches
                update_metadata(count=count)

    if count != num_features:
        raise ValueError(f'Count ({count}) must be equal to {num_features}.')

    update_metadata(count=count)


class PrecacheFeaturesDataset:
    def __init__(
        self,
        data,
        labels,
        feature_extractor,
        cache_dir,
        random_view,
        device,
        init_size=256,
        crop_size=224,
        mean=_MEAN_PIXEL_IMAGENET,
        std=_STD_PIXEL_IMAGENET,
        precache_num_workers=4,
        precache_batch_size=10,
        epoch_size=None,
        five_crop=True):
        """ If the cache is made, we don't need the feature extractor."""

        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f'cache_dir: {cache_dir}')
        if five_crop:
            done_file = cache_dir / 'cache_done'
            memmap_dir = cache_dir / 'ten_crop'
        else:
            done_file = cache_dir / 'cache_done_1crop'
            memmap_dir = cache_dir / 'single_crop'

        if (epoch_size is not None) and (epoch_size != len(data)):
            elem_list = generate_element_list(epoch_size, len(data))
            data = torch.utils.data.Subset(data, elem_list)
            labels = [labels[i] for i in elem_list]
        assert len(labels) == len(data)

        self.labels = labels
        self.data = ExtractCropsDataset(
            data, init_size=init_size, crop_size=crop_size, mean=mean, std=std,
            five_crop=five_crop)

        if not done_file.exists():
            print("Creating the memmap cache. It's going to take a while")
            make_memmap_crops(
                memmap_path=memmap_dir,
                feature_extractor=feature_extractor,
                dataset=self.data,
                device=device,
                num_workers=precache_num_workers,
                batch_size=precache_batch_size,
                num_views=(5 if five_crop else 1))
            done_file.touch()

        self._num_view = 10 if five_crop else 2
        self._central_view = CENTRAL_VIEW if five_crop else 0
        self.all_features = open_memmap(memmap_dir)
        self.random_view = random_view
        #self.view_index = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        view_index = (
            random.randint(0, self._num_view-1)
            if self.random_view else self._central_view)

        total_index = index * self._num_view  + view_index
        feature = torch.from_numpy(self.all_features[total_index])
        label = self.labels[index]
        return feature, label


def get_data_loaders_linear_classification_precache(
    dataset_name,
    data_dir,
    batch_size,
    workers,
    epoch_size,
    feature_extractor,
    cache_dir,
    device,
    precache_batch_size=10,
    five_crop=True,
    subset=None):

    assert isinstance(dataset_name, str)
    assert isinstance(batch_size, int)
    assert isinstance(workers, int)
    assert (epoch_size is None) or isinstance(epoch_size, int)

    if dataset_name in ("ImageNet", "Places205"):
        print("Loading data.")
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=None)
        dataset_test = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "val"), transform=None)
        train_split_str = "train"
        if (subset is not None and subset >= 1):
            train_split_str += f"_subset{subset}"
            dataset_train = subset_of_ImageNet_train_split(dataset_train, subset)

        precache_batch_size_train = (
            (precache_batch_size // 10)
            if five_crop else precache_batch_size)
        dataset_train = PrecacheFeaturesDataset(
            data=dataset_train,
            labels=dataset_train.targets,
            feature_extractor=feature_extractor,
            cache_dir=cache_dir / dataset_name / train_split_str,
            random_view=True,
            device=device,
            init_size=256,
            crop_size=224,
            mean=_MEAN_PIXEL_IMAGENET,
            std=_STD_PIXEL_IMAGENET,
            precache_num_workers=workers,
            precache_batch_size=precache_batch_size_train,
            epoch_size=epoch_size,
            five_crop=five_crop)

        dataset_test = PrecacheFeaturesDataset(
            data=dataset_test,
            labels=dataset_test.targets,
            feature_extractor=feature_extractor,
            cache_dir=cache_dir / dataset_name / "val",
            random_view=False,
            device=device,
            init_size=256,
            crop_size=224,
            mean=_MEAN_PIXEL_IMAGENET,
            std=_STD_PIXEL_IMAGENET,
            precache_num_workers=workers,
            precache_batch_size=precache_batch_size,
            epoch_size=None,
            five_crop=False)
    else:
        raise VallueError(f"Unknown/not supported dataset {dataset_name}")

    print("Creating data loaders")
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(sampler_train is None),
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_train,
        drop_last=True)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler_test,
        drop_last=False)

    return (
        loader_train, sampler_train, dataset_train,
        loader_test, sampler_test, dataset_test)
