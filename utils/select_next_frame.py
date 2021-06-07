import numpy as np
import torch.nn.functional as F
import torch

def select_next_frame(frame_value, metric='min', prev_frames=None):

    nb_frames = len(frame_value)

    if metric == 'random': return int(np.random.randint(nb_frames, size=1))

    if metric == 'uniform':
        assert prev_frames is not None

    if metric == 'prob':
        temp_prob = np.random.rand()
        prob = F.softmax(torch.Tensor(frame_value), 0)
        k = 0
        while (temp_prob > 0):
            temp_prob = temp_prob - prob[k]
            k += 1
        frame_to_annotate = (k - 1)

        return frame_to_annotate

    if metric == 'max': frame_value = -frame_value

    if prev_frames is not None:
        value_idx = frame_value.argsort()
        i = 0
        while i < nb_frames and value_idx[i] in prev_frames: i += 1
        if i == nb_frames: return frame_value.argmin() # All the frames have been annotated
        frame_to_annotate = value_idx[i]

    else:
        frame_to_annotate = frame_value.argmin()

    return frame_to_annotate