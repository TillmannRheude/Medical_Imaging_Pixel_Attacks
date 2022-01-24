import os

import torch
from medmnist import INFO, Evaluator
from models import create_resnet
import torchvision.transforms as transforms

from utils import load_mnist

# evaluation

def evaluate(model, data_loader, split, data_flag, dev="cpu"):
    info = INFO[data_flag]
    task = info['task']

    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(dev)

            outputs = model(inputs)

            outputs = outputs.to("cpu")

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print('%s  acc: %.3f  auc:%.3f' % (split, *metrics))



if __name__ == "__main__":
    ## load model
    data_flag = 'bloodmnist'
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    download = True
    num_workers = 2
    NUM_EPOCHS = 1
    BATCH_SIZE = 128
    lr = 0.001

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'trained_models/', data_flag + '.pth')

    dataset = load_mnist(data_flag, BATCH_SIZE, download, num_workers, data_transform)
    train_loader = dataset["train_loader"]
    test_loader = dataset["test_loader"]
    train_loader_at_eval = dataset["train_loader_at_eval"]

    model = create_resnet(data_flag)
    model.to(dev)
    model.load_state_dict(torch.load(PATH))

    evaluate(model, train_loader_at_eval, "train", data_flag, dev=dev)
    evaluate(model, test_loader, "test", data_flag, dev=dev)
