import os
import shutil
import torch


def save(log_dir, state_dict, is_best):
    checkpoint_path = os.path.join(log_dir, 'checkpoint.pt')
    torch.save(state_dict, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(log_dir, 'best_model.pt')
        shutil.copyfile(checkpoint_path, best_model_path)


class AverageMeter(object):
    """Computes and stores the average and current value
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
