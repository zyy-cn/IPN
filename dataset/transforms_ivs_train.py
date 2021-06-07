import random
import cv2
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def convert_one_hot(oh, max_obj):

    mask = np.zeros(oh.shape[:2], dtype=np.uint8)
    for k in range(max_obj+1):
        mask[oh[:, :, k]==1] = k

    return mask


def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    oh = np.stack(oh, axis=2)

    return oh


class RandomAffine(object):

    """
    Affine Transformation to each frame
    """

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']



        num = len(frames)
        for idx in range(num):
            img = frames[idx]
            anno = masks[idx]
            num_objs = len(np.unique(anno))

            segmap = SegmentationMapsOnImage(anno, shape=img.shape)

            timing = 0
            while True:
                seq = iaa.Sequential([
                    iaa.Crop(percent=(0.0, 0.1), keep_size=True, seed=random.randint(0, 2020)),
                    iaa.Affine(scale=(0.9, 1.1), shear=(-15, 15), rotate=(-25, 25), seed=random.randint(0, 2020))
                ])
                img_aug, segmap_aug = seq(image=img, segmentation_maps=segmap)
                if len(np.unique(segmap_aug.get_arr())) == num_objs:
                    frames[idx] = img_aug
                    masks[idx] = segmap_aug.get_arr()
                    break
                elif timing > 10:
                    frames[idx] = img
                    masks[idx] = anno
                    break
                else:
                    timing += 1

        sample['frames'] = frames
        sample['masks'] = masks

        return sample


class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']

        frames = frames.astype(np.float64)
        v = np.random.uniform(-self.delta, self.delta)
        for id, img in enumerate(frames):
            frames[id] += v

        frames = frames.astype(np.uint8)
        frames = np.clip(frames, 0, 255)
        sample['frames'] = frames
        sample['masks'] = masks

        return sample


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']
        frames = frames.astype(np.float64)
        v = np.random.uniform(self.lower, self.upper)
        for id, img in enumerate(frames):
            frames[id] *= v
        frames = frames.astype(np.uint8)
        frames = np.clip(frames, 0, 255)
        sample['frames'] = frames
        sample['masks'] = masks

        return sample


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']

        v = random.randint(0, 1)
        if v == 0:
            sample['frames'] = frames
            sample['masks'] = masks
            return sample

        # sample = frames[0]

        for id, img in enumerate(frames):
            frames[id] = img[:, ::-1, :]

        for id, anno in enumerate(masks):
            masks[id] = anno[:, ::-1]

        sample['frames'] = frames
        sample['masks'] = masks
        return sample


class Rotate(object):

    def __init__(self, rots=(-30, 30)):
        self.rots = rots

    def __call__(self, sample):
        frames = sample['frames']
        masks = sample['masks']
        num_frames = len(frames)
        (h, w) = frames.shape[1:3]
        center = (w / 2, h / 2)

        assert (center != 0)  # Strange behaviour warpAffine
        num_objs = len(np.unique(masks))
        while True:
            rots = [(self.rots[1] - self.rots[0]) * random.random() - (self.rots[1] - self.rots[0])/2
                    for _ in range(num_frames)]
            Ms = [cv2.getRotationMatrix2D(center, rot, 1) for rot in rots]
            masks_target = np.stack([cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST) for M, mask in zip(Ms, masks)])
            if len(np.unique(masks_target)) == num_objs:
                frames_target = np.stack([cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC) for M, frame in zip(Ms, frames)])
                break

        sample['frames'] = frames_target
        sample['masks'] = masks_target

        return sample


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, shortest_edge=480):
        self.shortest_edge = shortest_edge

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']
        size = np.array(frames.shape[1:3])
        ratio = size.max() / size.min()

        size_target = (self.shortest_edge, int(self.shortest_edge * ratio)) if size[0] < size[1] \
            else (int(self.shortest_edge * ratio), self.shortest_edge)

        frames_target = np.stack([cv2.resize(frame, size_target[::-1], interpolation=cv2.INTER_LINEAR) for frame in frames])
        masks_target = np.stack([cv2.resize(mask, size_target[::-1], interpolation=cv2.INTER_NEAREST) for mask in masks])

        sample['frames'] = frames_target
        sample['masks'] = masks_target

        return sample


class RandomCrop(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, patch_size=400):
        self.patch_size = patch_size

    def __call__(self, sample):

        frames = sample['frames']
        masks = sample['masks']
        size = np.array(frames.shape[1:3])
        assert size.min() > self.patch_size
        while True:
            h_start = random.randint(0, size[0] - self.patch_size)
            w_start = random.randint(0, size[1] - self.patch_size)

            masks_target = masks[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
            indices = np.sort(np.unique(masks_target))  # include 0
            if len(indices) > 1 and 0 in indices:
                # re-arrange the indices to ensure the consecutiveness
                for i, indice in enumerate(indices):
                    masks_target[masks_target == indice] = i
                frames_target = frames[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
                break

        sample['frames'] = frames_target
        sample['masks'] = masks_target
        return sample
