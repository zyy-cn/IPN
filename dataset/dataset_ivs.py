import os
import numpy as np
from PIL import Image

from torch.utils import data

import random
import glob
import json


def check_preprocess(root_dir, split):
    seq_obj_list_file = os.path.join(os.path.join(root_dir, 'ImageSets', '2017', f'davis_2017_{split}.json'))
    if not os.path.exists(seq_obj_list_file):
        seq_obj_dict = preprocess(root_dir, split)
    else:
        seq_obj_dict = json.load(open(seq_obj_list_file))
    return seq_obj_dict


def preprocess(root_dir, split):
    print("davis 2017 {} dataset preprocessing".format(split))
    seq_obj_list_file = os.path.join(os.path.join(root_dir, 'ImageSets', '2017', f'davis_2017_{split}.json'))

    with open(os.path.join(root_dir, 'ImageSets', '2017', split + '.txt')) as f:
        seqs = f.readlines()
    seqs = list(map(lambda elem: elem.strip(), seqs))

    seq_obj_dict = {}

    for i, seq in enumerate(seqs):
        print(f"i:{i}, seq:{seq}")
        name_labels = np.sort(os.listdir(os.path.join(root_dir, 'Annotations/480p/', seq)))
        _mask = np.stack([np.array(Image.open(os.path.join(root_dir, 'Annotations/480p/', seq, name_label)))
                          for name_label in name_labels], axis=0)
        _mask[_mask == 255] = 0
        obj_ids = np.unique(_mask)
        num_objs = len(obj_ids)

        frame_dict = {}
        frame_dict['query'] = [name_label for name_label in name_labels]
        frame_dict['supp'] = [name_label for i, name_label in enumerate(name_labels)
                              if len(np.unique(_mask[i])) == num_objs]
        seq_obj_dict[seq] = frame_dict

    with open(seq_obj_list_file, 'w') as outfile:
        json.dump(seq_obj_dict, outfile)

    print("preprocess finish")
    return seq_obj_dict


class DAVIS_MO(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, train_mode=True, split='train', resolution='480p', epoch=0, seq=None,
                 num_sample_frames=0, transform=None, convert_to_single_instance=False):
        self.root = root
        self.train_mode = train_mode
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        self.transform = transform
        self.epoch = epoch
        self.convert_to_single_instance = convert_to_single_instance
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, '2017', split+'.txt')

        self.seq_obj_dict = check_preprocess(root_dir=root, split=split)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        if seq is not None:
            for _video in seq:
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)
        else:
            with open(os.path.join(_imset_f), "r") as lines:
                for line in lines:
                    if not self.convert_to_single_instance:
                        _video = line.rstrip('\n')
                        self.videos.append(_video)
                        self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                        _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                        self.num_objects[_video] = np.max(_mask)
                        self.shape[_video] = np.shape(_mask)
                        _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                        self.size_480p[_video] = np.shape(_mask480)
                    else:
                        _video = line.rstrip('\n')
                        _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                        _mask480[_mask480 == 255] = 0
                        num_objects = np.max(_mask480)
                        for o in range(1, num_objects+1):
                            self.videos.append(_video + f'_{o}')
                            self.num_objects[_video] = 1
                            self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                            self.shape[_video] = np.shape(_mask480)
                            self.size_480p[_video] = np.shape(_mask480)


        self.num_train_frames = num_sample_frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        if self.convert_to_single_instance:
            video, obj_id = video.split('_')[0], int(video.split('_')[1])
        else:
            obj_id = None

        info = {}
        if self.train_mode:
            info['name'] = video + f'_{obj_id}' if self.convert_to_single_instance else video
            info['num_frames'] = len(self.seq_obj_dict[video]['supp'])
            while True:
                if info['num_frames'] < self.num_train_frames:
                    video = random.sample(self.videos, 1)[0]
                    if self.convert_to_single_instance:
                        video, obj_id = video.split('_')[0], int(video.split('_')[1])
                    else:
                        obj_id = None
                    info['name'] = video + f'_{obj_id}' if self.convert_to_single_instance else video
                    info['num_frames'] = len(self.seq_obj_dict[video]['supp'])
                else:
                    break
            info['size_480p'] = self.size_480p[video]
            max_N_skip_frames = min(int(4 + self.epoch / 20), 8)
            N_skip_frames = random.randint(min(int(info['num_frames'] / self.num_train_frames) - 1, 4),
                                           min(int(info['num_frames'] / self.num_train_frames) - 1, max_N_skip_frames))
            subseq = np.linspace(0, (self.num_train_frames * (N_skip_frames + 1)) - 1, num=self.num_train_frames).astype(int)
            if subseq[-1] < info['num_frames']:
                add_index = random.randint(0, info['num_frames'] - subseq[-1] - 1)
                subseq += add_index

            supp_index = [self.seq_obj_dict[video]['supp'][i] for i in subseq]

            fnames = [os.path.join(self.image_dir, video, f"{i.split('.')[0]}.jpg") for i in supp_index]
            frames = np.stack([np.array(Image.open(fname).convert('RGB'), dtype=np.uint8) for fname in fnames], axis=0)

            masks = np.empty((self.num_train_frames,) + self.shape[video], dtype=np.uint8)
            for f in range(self.num_train_frames):
                try:
                    mask_file = os.path.join(self.mask_dir, video, f'{supp_index[f]}')
                    mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                    if self.convert_to_single_instance:
                        assert obj_id is not None
                        mask = (mask == obj_id).astype(np.uint8)
                    else:
                        mask[mask == 255] = 0

                    masks[f] = mask
                except:
                    masks[f] = 0
        else:
            info['name'] = video
            info['num_frames'] = len(self.seq_obj_dict[video]['query'])
            info['size_480p'] = self.size_480p[video]

            fnames = [os.path.join(self.image_dir, video, f"{i.split('.')[0]}.jpg") for i in self.seq_obj_dict[video]['query']]
            frames = np.stack([np.array(Image.open(fname).convert('RGB'), dtype=np.uint8) for fname in fnames], axis=0)

            masks = np.empty((info['num_frames'],) + self.shape[video], dtype=np.uint8)
            for f in range(info['num_frames']):
                try:
                    mask_file = os.path.join(self.mask_dir, video, f"{self.seq_obj_dict[video]['query'][f]}")
                    mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                    mask[mask == 255] = 0
                    masks[f] = mask
                except:
                    masks[f] = 0
            N_skip_frames = 0
        sample = {}
        sample['frames'] = frames
        sample['masks'] = masks
        sample['video_name'] = video + f'_{obj_id}' if self.convert_to_single_instance else video
        sample['N_skip_frames'] = N_skip_frames

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
