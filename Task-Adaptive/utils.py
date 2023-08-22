class AverageMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count
