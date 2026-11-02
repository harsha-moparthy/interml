"""Decay a learning rate by gamma at fixed step intervals."""


class StepLRScheduler:

    def __init__(self, initial_lr, step_size, gamma):
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        num_decays = epoch // self.step_size
        current_lr = self.initial_lr * self.gamma**num_decays
        return round(current_lr, 4)
