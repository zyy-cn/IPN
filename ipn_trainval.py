import os
import pickle
import random
import numpy as np
from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision.transforms import Compose

from davisinteractive.dataset.davis import Davis
from davisinteractive import utils as interactive_utils
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
from davisinteractive.evaluation.service import ROBOT_DEFAULT_PARAMETERS
from davisinteractive.metrics import batched_jaccard, batched_f_measure

from utils.misc import save_IPN_checkpoint, AverageMeter
from utils.select_next_frame import select_next_frame


from dataset.dataset_ivs import DAVIS_MO
import dataset.transforms_ivs_train as tr
from model import model as ivs_model

ex = Experiment()
ex.add_config('./configs/ipn.yaml')

cudnn.benchmark = False
cudnn.deterministic = True

robot = InteractiveScribblesRobot(**ROBOT_DEFAULT_PARAMETERS)

GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


@ex.capture
def train(cfg, model, optimizer, scheduler, history, epoch, device, _log):
    losses = AverageMeter()

    # prepare data
    train_transform = Compose([
        tr.Resize(),
        tr.RandomCrop(),
        tr.RandomAffine(),
        tr.RandomContrast(),
        tr.AdditiveNoise(),
        tr.RandomMirror(),
    ])

    dataset_train = DAVIS_MO(root=cfg.dataset.root_dir_davis, train_mode=True, split='train', epoch=epoch,
                             num_sample_frames=8, transform=train_transform, convert_to_single_instance=True)
    dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    for index, sample in enumerate(dataloader_train):
        Fs = sample['frames']
        GTs = sample['masks']
        video_name = sample['video_name']
        N_skip_frames = sample['N_skip_frames']
        assert Fs.shape[0] == 1 and GTs.shape[0] == 1
        Fs = Fs[0].to(device)
        GTs = GTs[0].to(device)
        video_name = video_name[0]
        N_skip_frames = N_skip_frames[0]

        num_frames = len(GTs)

        model.model_I.zero_grad()
        model.model_P.zero_grad()
        variables = model.init_variables(frames=Fs, masks=GTs, device=device)
        metric = None
        if N_skip_frames < 5:
            nb_interactions = 1
        elif N_skip_frames < 7:
            nb_interactions = 2
        else:
            nb_interactions = 3
        for n in range(nb_interactions):
            target_frames = random.randint(0, num_frames-1) if n == 0 else metric.mean(1).argmin()
            masks = np.zeros_like(GTs.cpu().numpy()) if n == 0 else variables['masks'].cpu().numpy()
            scribbles = robot.interact('', masks, GTs.cpu().numpy(), frame=target_frames)
            scribbles['annotated_frame'] = target_frames
            variables['scribbles'] = scribbles

            model.Run(variables)

            loss = variables['loss']
            losses.update(loss.item())
            # Backward
            loss.backward()
            for param in list(model.model_I.parameters()) + list(model.model_P.parameters()):
                if hasattr(param.grad, 'data'):
                    param.grad.data.clamp_(-1, 1)
            optimizer.step()

            if nb_interactions > 1:
                metric = 0.5 * batched_jaccard(GTs.cpu().numpy(), variables['masks'], average_over_objects=False) + \
                         0.5 * batched_f_measure(GTs.cpu().numpy(), variables['masks'], average_over_objects=False)
                # clear the grad
                model.model_I.zero_grad()
                model.model_P.zero_grad()
                variables['mask_objs'] = variables['mask_objs'].data
                variables['ref'] = [variables['ref'][o].data for o in range(len(variables['ref']))]

        _log.info(f"Epoch: [{epoch:2d}][{index + 1:3d}/{len(dataloader_train):3d}]\t"
                  f"Loss: {losses.val:.20f} ({losses.avg:.20f})\t"
                  f"video_name: {video_name}")

    _log.info(f"* Epoch: [{epoch:3d}]\tLoss: {losses.avg:.6f}")

    if scheduler is not None:
        scheduler.step(epoch)

    history['train']['loss'].append(losses.avg)


@ex.capture
def eval_17(cfg, model, device, max_nb_interactions, scribble_list, history, epoch=0, seq=None,
            metric_type='J_AND_F', _log=None):
    nb_objects = AverageMeter()
    davis = Davis(davis_root=cfg.dataset.root_dir_davis)
    dataset_val = DAVIS_MO(root=cfg.dataset.root_dir_davis, train_mode=False, split='val', seq=seq)
    dataloader_val = data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    metric_meter = [AverageMeter() for _ in range(max_nb_interactions)]
    global_auc, global_metric_list = None, None
    for index, sample in enumerate(dataloader_val):
        Fs = sample['frames']
        GTs = sample['masks']
        video_name = sample['video_name']
        assert Fs.shape[0] == 1 and GTs.shape[0] == 1
        Fs = Fs[0].to(device)
        GTs = GTs[0].to(device)
        video_name = video_name[0]
        num_objs = len(torch.unique(GTs)) - 1
        nb_objects.update(num_objs)

        for s in scribble_list:
            metric = None
            variables = model.init_variables(frames=Fs, masks=GTs, device=device)
            metric_list_n = []
            for n in range(max_nb_interactions):
                if n == 0:
                    scribbles = davis.load_scribble(video_name, s)
                else:
                    assert metric is not None
                    target_frames = select_next_frame(metric.mean(1), metric='worst',
                                                      prev_frames=variables['prev_targets'])           # worst
                    scribbles = robot.interact('', variables['masks'].cpu().numpy(), GTs.cpu().numpy(), frame=target_frames)

                scribbles['annotated_frame'] = interactive_utils.scribbles.annotated_frames(scribbles)[0]
                variables['scribbles'] = scribbles

                model.Run(variables)

                if metric_type == 'J':
                    metric = batched_jaccard(GTs.cpu().numpy(), variables['masks'], average_over_objects=False)
                elif metric_type == 'F':
                    metric = batched_f_measure(GTs.cpu().numpy(), variables['masks'], average_over_objects=False)
                elif metric_type == 'J_AND_F':
                    metric = 0.5 * batched_jaccard(GTs.cpu().numpy(), variables['masks'], average_over_objects=False) + \
                             0.5 * batched_f_measure(GTs.cpu().numpy(), variables['masks'], average_over_objects=False)
                else:
                    raise NotImplementedError("metric must specific to 'J', 'F' or 'J_AND_F'")


                for o in range(metric.shape[1]):
                    metric_meter[n].update(metric[:, o].mean())
                metric_list_n.append(metric.mean())

            # local
            metric_list_n = [0] + metric_list_n
            auc = np.trapz(metric_list_n) / (len(metric_list_n)-1)

            # global
            global_metric_list = [0] + [metric_n.avg for metric_n in metric_meter]
            global_auc = np.trapz(global_metric_list) / (len(global_metric_list) - 1)

            _log.info(f"video: [{index + 1:3d}/{len(dataloader_val):3d}][{s}]\t"
                      f"curve: {[f'{metric_i*100:.2f}' for metric_i in metric_list_n[1:]]} \t"
                      f"auc: {auc * 100:.2f} ({global_auc * 100:.2f})\t"
                      f"video_name: {video_name}")

    _log.info(f"* Epoch: [{epoch:3d}]\t"
              f"curve:\t{[f'{metric_i*100:.2f}' for metric_i in global_metric_list]}\t"
              f"auc: {global_auc * 100:.2f}")

    history['val']['auc'].append(global_auc)


@ex.automain
def main(_run, _log):
    cfg = edict(_run.config)

    # set random seeds
    set_seed(cfg.seed)

    # Network Builders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ivs_model(load_pretrain=False)

    ckpt_dir = os.path.join('experiments', str(_run._id)) if _run._id else os.path.join('experiments', 'public')

    # Set up optimizers
    params_dict = [dict(params=list(model.model_I.parameters()) + list(model.model_P.parameters()))]
    optimizer = torch.optim.Adam(params_dict, lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"total params to optimize: {len(params_dict[0]['params'])}")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400], gamma=0.1, last_epoch=-1)

    history = {'train': {'loss': [], 'iou': []},
               'val': {'auc': [], 'best_auc': 0}}

    for epoch in range(1, cfg.num_epochs+1):

        _log.info(f"Epoch: {epoch}, current learning rate: {scheduler.get_lr()[0]}")

        train(cfg, model, optimizer, scheduler, history, epoch, device)

        if epoch % cfg.eval_interval == 0:
            with torch.no_grad():
                eval_17(cfg, model, device=device, max_nb_interactions=8, scribble_list=[1], history=history, epoch=epoch,
                        seq=None)

                save_IPN_checkpoint(model.model_I, model.model_P, epoch, ckpt_dir, is_best=False)
                pickle.dump(history, open(os.path.join(ckpt_dir, 'history.pkl'), 'wb'))

