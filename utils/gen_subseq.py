import numpy as np

def gen_subseq(subseq_style, first_frame, n_frame, len_subseq):
    assert n_frame >= len_subseq
    if subseq_style == 'equal':
        start = 0
        end = n_frame - 1
        if (end - start + 1) < len_subseq + 1:
            subseq = np.array(range(len_subseq))
        else:
            assert (end - start + 1) >= len_subseq + 1
            subseq = np.linspace(start, n_frame-1, num=len_subseq + 1).astype(int)
            while True:
                if first_frame not in list(subseq):
                    subseq += 1
                else:
                    break
            if first_frame != subseq[-1]:
                subseq = list(subseq[:-1])
            else:
                subseq = list(subseq[1:])
    else:
        raise NotImplementedError

    return subseq