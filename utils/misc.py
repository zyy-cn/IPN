import shutil
import os

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_IPN_checkpoint(model_I, model_P, ckpt_dir, epoch=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    if epoch is None:
        dict_network = model_I.state_dict()
        torch.save(dict_network, os.path.join(ckpt_dir, f'I.pth'))
        dict_network = model_P.state_dict()
        torch.save(dict_network, os.path.join(ckpt_dir, f'P.pth'))
    else:
        dict_network = model_I.state_dict()
        torch.save(dict_network, os.path.join(ckpt_dir, f'I_{epoch}.pth'))
        dict_network = model_P.state_dict()
        torch.save(dict_network, os.path.join(ckpt_dir, f'P_{epoch}.pth'))
