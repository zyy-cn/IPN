import os
import pickle

import matplotlib.pyplot as plt
plt.switch_backend('agg')


if __name__ == '__main__':
    ckpt_dir = 'experiments'

    if not os.path.isdir(os.path.join(ckpt_dir, 'figures')):
        os.makedirs(os.path.join(ckpt_dir, 'figures'))

    for id in range(24, 26):
        if not os.path.exists(os.path.join(ckpt_dir, str(id), 'history.pkl')):
            print(f'History of experiment {id} does not exist.')
            # continue

        history = pickle.load(open(os.path.join(ckpt_dir, str(id), 'history.pkl'), 'rb'))

        plt.figure(figsize=(12, 18))
        plt.subplot(211)
        plt.plot(history['train']['loss'], linewidth=3)
        plt.title('train loss')
        plt.grid()

        plt.subplot(212)
        plt.plot(history['val']['auc'], linewidth=3)
        plt.title('val auc')
        plt.grid()

        plt.savefig(os.path.join(ckpt_dir, 'figures', str(id) + '.png'), format='png')
