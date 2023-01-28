from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, path):
        self.writer = SummaryWriter(log_dir=path)
    
    def log(self, path, value, idx):
        self.writer.add_scalar(path, value, idx)