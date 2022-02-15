from experiments.experiments import load_raw
import matplotlib.pyplot as plt

root = 'experiments_raw'
target_dir = 'experiment_results'

def plot_experiment(experiment):
    heatmap = load_raw(f'{root}/{experiment}')
    PATH = ''
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.savefig(PATH)
    plt.show()


if __name__ == '__main__':
    "resnet18_octmnist_zero_one.pth"


