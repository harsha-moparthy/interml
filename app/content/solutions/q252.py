"""Decay a learning rate exponentially by gamma each step."""


class ExponentialLRScheduler:

    def __init__(self, initial_lr, gamma):
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        current_lr = self.initial_lr * self.gamma**epoch
        return round(current_lr, 4)
