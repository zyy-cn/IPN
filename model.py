from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

# general libs
import numpy as np
import copy

# my libs
from utils_ipn import ToCudaPN, Dilate_mask, load_UnDP, Get_weight
from interaction_net import Inet
from propagation_net import Pnet

# davis
from davisinteractive.utils.scribbles import scribbles2mask


class model():
    def __init__(self, load_pretrain=True):
        self.model_I = Inet()
        self.model_P = Pnet()
        if torch.cuda.is_available():
            print('Using GPU')
            self.model_I = nn.DataParallel(self.model_I)
            self.model_P = nn.DataParallel(self.model_P)
            self.model_I.cuda()
            self.model_P.cuda()
            if load_pretrain:
                self.model_I.load_state_dict(torch.load('I_e290.pth'))
                self.model_P.load_state_dict(torch.load('P_e290.pth'))
        else:
            print('Using CPU')
            if load_pretrain:
                self.model_P.load_state_dict(load_UnDP('P_e290.pth'))
                self.model_I.load_state_dict(load_UnDP('I_e290.pth'))

        self.model_I.eval() # turn-off BN
        self.model_P.eval() # turn-off BN


    def init_variables(self, frames, masks, device='cuda'):
        num_frames, height, width = frames.shape[:3]
        if type(frames) is np.ndarray:
            frames = torch.Tensor(frames)
        if type(masks) is np.ndarray:
            masks = torch.Tensor(masks)
        num_objs = len(torch.unique(masks)) - 1
        all_F = torch.unsqueeze(frames.permute((3, 0, 1, 2)).float() / 255., dim=0).to(device) # 1,3,t,h,w
        all_M = torch.unsqueeze(masks.float(), dim=0).long().to(device) # 1,t,h,w
        prev_targets = []
        variables = {}
        variables['all_F'] = all_F
        variables['all_M'] = all_M
        variables['mask_objs'] = -1 * torch.ones(num_objs, num_frames, height, width).to(device)  # o,t,h,w
        variables['prev_targets'] = prev_targets
        variables['probs'] = None
        variables['info'] = {}
        variables['info']['num_frames'] = num_frames
        variables['info']['num_objs'] = num_objs
        variables['info']['height'] = height
        variables['info']['width'] = width
        variables['info']['device'] = device
        variables['ref'] = [None for _ in range(num_objs)]

        return variables

    def Run(self, variables, optimizer=None):
        all_F = variables['all_F']
        num_objects = variables['info']['num_objs']
        num_frames = variables['info']['num_frames']
        height = variables['info']['height']
        width = variables['info']['width']
        prev_targets = variables['prev_targets']
        scribbles = variables['scribbles']
        target = scribbles['annotated_frame']

        loss = 0
        masks = torch.zeros(num_objects, num_frames, height, width)
        for n_obj in range(1, num_objects+1):

            if optimizer is not None:
                self.model_I.zero_grad()
                self.model_P.zero_grad()

            # variables for current obj
            all_E_n = variables['mask_objs'][n_obj-1:n_obj].data if variables['mask_objs'][n_obj-1:n_obj] is not None \
                else variables['mask_objs'][n_obj-1:n_obj]
            a_ref = variables['ref'][n_obj-1]
            prev_E_n = all_E_n.clone()
            all_M_n = (variables['all_M'] == n_obj).long()

            # divide scribbles for current object
            n_scribble = copy.deepcopy(scribbles)
            n_scribble['scribbles'][target] = []
            for p in scribbles['scribbles'][target]:
                if p['object_id'] == n_obj:
                    n_scribble['scribbles'][target].append(copy.deepcopy(p))
                    n_scribble['scribbles'][target][-1]['object_id'] = 1
                else:
                    if p['object_id'] == 0:
                        n_scribble['scribbles'][target].append(copy.deepcopy(p))

            scribble_mask = scribbles2mask(n_scribble, (height, width))[target]
            scribble_mask_N = (prev_E_n[0, target].cpu() > 0.5) & (torch.tensor(scribble_mask) == 0)
            scribble_mask[scribble_mask == 0] = -1
            scribble_mask[scribble_mask_N] = 0
            scribble_mask = Dilate_mask(scribble_mask, 1)

            # interaction
            tar_P, tar_N = ToCudaPN(scribble_mask)
            all_E_n[:, target], batch_CE, ref = self.model_I(all_F[:, :, target], all_E_n[:, target],
                                                             # tar_P, tar_N, all_M_n[:, target], [1, 0, 0, 0, 0])  # [batch, 256,512,2]
                                                             tar_P, tar_N, all_M_n[:, target], [1, 1, 1, 1, 1])  # [batch, 256,512,2]
            loss += batch_CE

            # propagation
            left_end, right_end, weight = Get_weight(target, prev_targets, num_frames, at_least=-1)

            # Prop_forward
            next_a_ref = None
            for n in range(target+1, right_end+1):  #[1,2,...,N-1]
                all_E_n[:,n], batch_CE, next_a_ref = self.model_P(ref, a_ref, all_F[:,:,n], prev_E_n[:,n],
                                                                  # all_E_n[:,n-1], all_M_n[:, n], [1,0,0,0,0])
                                                                  all_E_n[:,n-1], all_M_n[:, n], [1,1,1,1,1], next_a_ref)
                loss += batch_CE

            # Prop_backward
            for n in reversed(range(left_end, target)):
                all_E_n[:,n], batch_CE, next_a_ref = self.model_P(ref, a_ref, all_F[:,:,n], prev_E_n[:,n],
                                                                  all_E_n[:,n+1], all_M_n[:, n], [1,1,1,1,1], next_a_ref)
                loss += batch_CE

            for f in range(num_frames):
                # all_E_n[:, :, f] = weight[f] * all_E_n[:, :, f] + (1 - weight[f]) * prev_E_n[:, :, f]
                all_E_n[:, f, :, :] = weight[f] * all_E_n[:, f, :, :] + (1 - weight[f]) * prev_E_n[:, f, :, :]

            masks[n_obj - 1] = all_E_n[0]
            variables['ref'][n_obj - 1] = next_a_ref


            # Backward
            if optimizer is not None:
                loss.backward(retain_graph=True)
                for param in list(self.model_I.parameters()) + list(self.model_P.parameters()):
                    if hasattr(param.grad, 'data'):
                        param.grad.data.clamp_(-1, 1)
                optimizer.step()


        loss /= num_objects

        em = torch.zeros(1, num_objects + 1, num_frames, height, width).to(masks.device)
        em[0, 0, :, :, :] = torch.prod(1 - masks, dim=0)  # bg prob
        em[0, 1:num_objects + 1, :, :] = masks  # obj prob
        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        all_E = torch.log((em / (1 - em)))

        all_E = F.softmax(all_E, dim=1)
        final_masks = all_E[0].max(0)[1].float()

        variables['prev_targets'].append(target)
        variables['masks'] = final_masks
        variables['mask_objs'] = masks
        variables['probs'] = all_E
        variables['loss'] = loss

