from experiments import load_raw
import matplotlib.pyplot as plt
from glob import glob
import os

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
    for k, metrics in metric_dict.items():
        xs.append(k)
        auc.append(metrics[0])
        acc.append(metrics[1])

    ax.plot(xs, acc, label="Accuracy")
    ax.plot(xs, auc, label="Area Under Curve")
    ax.set_xlabel("Attacked Pixels")
    ax.set_ylabel("value")
    ax.legend()
    fig.suptitle(' '.join(experiment[:-4].split('_')) + ' attack')
    plt.show()
    fig.savefig(f'{target_dir}/count_{experiment[:-4]}.pdf')
    fig.savefig(f'{target_dir}/count_{experiment[:-4]}.png')



def location(experiment):

    heatmap = load_raw(f'{location_root}/{experiment}')

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    fig.colorbar(im)

    fig.suptitle(' '.join(experiment[:-4].split('_')) + ' attack')
    fig.savefig(f'{target_dir}/location_{experiment[:-4]}.pdf')
    fig.savefig(f'{target_dir}/location_{experiment[:-4]}.png')


    plt.show()
if __name__ == '__main__':
    counts = map(os.path.basename, glob(f'{count_root}/*.pth'))
    locations = map(os.path.basename, glob(f'{location_root}/*.pth'))

    for c in counts:
        count(c)
    for l in locations:
        location(l)


