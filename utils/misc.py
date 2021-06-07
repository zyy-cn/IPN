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


def save_IPN_checkpoint(model_I, model_P, epoch, ckpt_dir, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    dict_network = model_I.state_dict()
    torch.save(dict_network, os.path.join(ckpt_dir, f'I_latest.pth'))
    dict_network = model_P.state_dict()
    torch.save(dict_network, os.path.join(ckpt_dir, f'P_latest.pth'))

    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, f'I_latest.pth'),
                        os.path.join(ckpt_dir, f'I_best.pth'))
        shutil.copyfile(os.path.join(ckpt_dir, f'P_latest.pth'),
                        os.path.join(ckpt_dir, f'P_best.pth'))
