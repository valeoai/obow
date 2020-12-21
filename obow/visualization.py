import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import obow.utils as utils
import numpy as np
import datetime
import time

from tqdm import tqdm


logger = utils.setup_dist_logger(logging.getLogger(__name__))


def extract_visual_words(model, dataloader, dtype='uint32'):
    model.eval()
    all_vword_ids, all_vword_mag, num_words = [], [], []
    count = 0
    for i, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            img = batch[0] if isinstance(batch, (list, tuple)) else batch
            img = img[0] if isinstance(img, (list, tuple)) else img
            img = img.cuda()
            assert img.dim()==4

            # Add horizontal flip:
            img = torch.stack([img, torch.flip(img, dims=(3,))], dim=1)
            assert img.dim() == 5 and img.size(1) == 2
            img = img.view(2 * img.size(0), 3, img.size(3), img.size(4))

            features = model.feature_extractor_teacher(img, model._bow_levels)
            _, vword_codes = model.bow_extractor(features)
            assert isinstance(vword_codes, (list, tuple))
            num_levels = len(vword_codes)
            batch_size = img.size(0) // 2

            if i == 0:
                max_count = len(dataloader) * batch_size
                logger.info(f'image size: {img.size()}')
                logger.info(f'max count: {max_count}')
                logger.info(f'batch size: {batch_size}')
                for level in range(num_levels):
                    _, num_words_this, height, width = vword_codes[level].size()
                    dshape = [max_count * 2, height, width]
                    all_vword_ids.append(np.zeros(dshape, dtype=dtype))
                    all_vword_mag.append(np.zeros(dshape, dtype='float32'))
                    num_words.append(num_words_this)
                    logger.info(
                        f'Level {level}: shape: {dshape[1:]}, '
                        f'num_words: {num_words[level]}')

            for level in range(num_levels):
                vwords_mag, vwords_ids = vword_codes[level].max(dim=1)
                assert vwords_mag.dim() == 3
                assert vwords_ids.dim() == 3
                vwords_ids = vwords_ids.cpu().numpy()
                vwords_mag = vwords_mag.cpu().numpy().astype('float32')
                all_vword_ids[level][count:(count + batch_size*2)] = vwords_ids.astype(dtype)
                all_vword_mag[level][count:(count + batch_size*2)] = vwords_mag

        count += batch_size*2

    for level in range(num_levels):
        all_vword_ids[level] = all_vword_ids[level][:count]
        all_vword_mag[level] = all_vword_mag[level][:count]
        logger.info(f'Shape of extracted dataset: {all_vword_ids[level].shape}')

    return all_vword_ids, all_vword_mag, num_words


def visualize_visual_words(
    num_words,
    num_patches,
    patch_size,
    dataset_images,
    all_vword_ids,
    all_vword_mag,
    words_order,
    dst_dir,
    rank=0,
    offset_k=0,
    mean_pixel=[0.485, 0.456, 0.406],
    std_pixel=[0.229, 0.224, 0.225],
    skip_border=True):

    assert all_vword_mag.shape == all_vword_ids.shape
    assert (len(dataset_images) * 2) == all_vword_ids.shape[0]

    mean_pixel = torch.Tensor(mean_pixel).view(1, 3, 1, 1)
    std_pixel = torch.Tensor(std_pixel).view(1, 3, 1, 1)

    num_images, height, width = all_vword_ids.shape
    num_images = num_images // 2
    all_vword_ids = all_vword_ids.reshape(num_images, 2, height, width)
    all_vword_mag = all_vword_mag.reshape(num_images, 2, height, width)
    all_vword_mag_flat = all_vword_mag.reshape(-1)
    all_vword_ids_flat = all_vword_ids.reshape(-1)

    num_locs = height * width
    num_locs_flip = 2 * height * width

    assert height == width
    size_out = (height+2) if skip_border else height

    def parse_index(index):
        img = index // num_locs_flip
        flip_loc = index % num_locs_flip
        flip = flip_loc // num_locs
        loc = flip_loc % num_locs
        y = loc // width
        x = loc % width
        return img, flip, y, x

    def extract_patch(image, flip, y, x):
        assert image.dim() == 3
        assert image.size(0) == 3
        assert image.size(1) == image.size(2)
        assert (image.size(2) % size_out) == 0
        size_in = image.size(2)
        stride = size_in // size_out
        offset = stride // 2 # offset due to image padding
        halfp = patch_size // 2

        if skip_border:
            x = x + 1
            y = y + 1

        if flip == 1:
            image = torch.flip(image, dims=(2,))

        image_pad = F.pad(image, (halfp, halfp, halfp, halfp), 'constant', 0.0)

        xc = x * stride + offset + halfp
        yc = y * stride + offset + halfp
        x1 = xc - halfp
        y1 = yc - halfp
        x2 = xc + halfp
        y2 = yc + halfp
        assert x1 > 0
        assert y1 > 0
        assert y2 < image_pad.size(1)
        assert x2 < image_pad.size(2)

        #print(x1, x2, y1, y2)
        return image_pad[:, y1:y2, x1:x2]

    num_words_order = words_order.shape[0]
    iter_start_time = time.time()
    total_time = 0
    for k in range(num_words_order):
        visual_word_id = words_order[k]
        indices_k = np.nonzero(all_vword_ids_flat == visual_word_id)[0]
        if indices_k.shape[0] == 0:
            print(f"==> The visual word with id {visual_word_id} is empty.")
        else:
            vword_mag_k = all_vword_mag_flat[indices_k]
            order = np.argsort(-vword_mag_k)
            vword_mag_k = vword_mag_k[order]
            indices_k = indices_k[order]

            if order.shape[0] >= 2:
                assert vword_mag_k[0] >= vword_mag_k[1]

            count_patches = 0
            count = 0
            used_image = np.zeros(num_images, dtype='uint8')
            patches_k = torch.zeros(num_patches, 3, patch_size, patch_size)
            while (count_patches < num_patches) and (count < order.shape[0]):
                index = indices_k[count]
                img, flip, y, x = parse_index(index)
                assert all_vword_mag_flat[index] == vword_mag_k[count]
                if used_image[img] == 0:
                    used_image[img] = 1
                    assert all_vword_ids[img,flip,y,x] == visual_word_id
                    assert all_vword_mag[img,flip,y,x] == vword_mag_k[count]
                    image = dataset_images[img][0] # get image
                    patch_this = extract_patch(image, flip, y, x) # extract patch
                    patches_k[count_patches].copy_(patch_this) # copy patch.
                    count_patches += 1
                count += 1

            #image_normalized = (image - mean) / std
            patches_k_unnormalized = patches_k.mul(std_pixel).add(mean_pixel)
            patches_k_unnormalized = patches_k_unnormalized.clamp_(0.0, 1.0)
            patches_k_vis = torchvision.utils.make_grid(
                patches_k_unnormalized, nrow=8, padding=5, normalize=False)

            dst_file = dst_dir + f'/freq_{offset_k+k}_visual_word_{visual_word_id}.jpg'
            torchvision.utils.save_image(patches_k_vis, dst_file)

        iter_time = time.time() - iter_start_time
        total_time += iter_time
        if (k % 20) == 0:
            avg_time = total_time / (k+1)
            eta_secs = avg_time * (num_words_order - k)
            elaphsed_time_string = str(datetime.timedelta(seconds=int(total_time)))
            eta_string = str(datetime.timedelta(seconds=int(eta_secs)))
            print(f"Iteration [{k}/{num_words_order}][rank={rank}]: elapsed_time = {elaphsed_time_string}, eta = {eta_string}.")

        iter_start_time = time.time()
