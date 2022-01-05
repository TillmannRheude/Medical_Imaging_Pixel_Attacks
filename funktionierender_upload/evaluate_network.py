import torch
from medmnist import INFO, Evaluator

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)

# evaluation

def test(model, data_loader, split, data_flag):
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