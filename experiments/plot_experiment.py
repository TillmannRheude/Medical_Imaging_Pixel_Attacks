from experiments import load_raw
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np

root = 'experiments_raw'
count_root = 'count_error_experiments_raw'
location_root = 'location_acc_experiments_raw'
target_dir = 'experiment_results'

def count(experiment):
    metric_dict = load_raw(f'{count_root}/{experiment}')

    fig, ax = plt.subplots()

    auc = []
    acc = []
    xs = []
    ax.set_ylim(0,1)
    for k, metrics in metric_dict.items():
        xs.append(k)
        acc.append(metrics[1])
        auc.append(metrics[0])
        if k >= 100: break

    ax.plot(xs, acc, label="Accuracy")
    ax.plot(xs, auc, label="Area Under Curve")
    ax.set_xlabel("Attacked Pixels")
    ax.set_ylabel("value")
    ax.legend()
    fig.suptitle(' '.join(experiment[:-4].split('_')) + ' attack')
    fig.savefig(f'{target_dir}/count_{experiment[:-4]}.pdf')
    fig.savefig(f'{target_dir}/count_{experiment[:-4]}.png')


def location(experiment):

    heatmap = load_raw(f'{location_root}/{experiment}')

    fig, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.imshow(heatmap, cmap='hot', interpolation='nearest',)
    im = ax2.imshow(heatmap, cmap='hot', interpolation='nearest')
    fig.colorbar(im)

    ax[0,0].set_ylabel("Area under curve")
    ax[1,0].set_ylabel("Accuracy")
    fig.suptitle(' '.join(experiment[:-4].split('_')) + ' attack')
    fig.savefig(f'{target_dir}/location_{experiment[:-4]}.pdf')
    fig.savefig(f'{target_dir}/location_{experiment[:-4]}.png')


if __name__ == '__main__':
    counts = map(os.path.basename, glob(f'{count_root}/*.pth'))
    locations = map(os.path.basename, glob(f'{location_root}/*.pth'))

    for c in counts:
        count(c)
    for l in locations:
        location(l)


