import math

import torch


class MomentumSGD(torch.optim.Optimizer):
    """Dense-gradient SGD with classical momentum; no dampening or Nesterov variant."""

    def __init__(self, params, lr=0.01, momentum=0.9):
        if lr < 0 or not 0 <= momentum < 1:
            raise ValueError("invalid learning rate or momentum")
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise ValueError("only dense gradients are supported")
                state = self.state[parameter]
                if "velocity" not in state:
                    state["velocity"] = gradient.clone()
                else:
                    state["velocity"].mul_(group["momentum"]).add_(gradient)
                parameter.add_(state["velocity"], alpha=-group["lr"])
        return loss


class AdamW(torch.optim.Optimizer):
    """Dense AdamW with per-parameter moments, bias correction, and decoupled decay."""

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        if lr < 0 or eps <= 0 or weight_decay < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("invalid optimizer settings")
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise ValueError("only dense gradients are supported")
                state = self.state[parameter]
                if not state:
                    state.update(
                        step=0,
                        mean=torch.zeros_like(parameter),
                        variance=torch.zeros_like(parameter),
                    )
                state["step"] += 1
                mean, variance = state["mean"], state["variance"]
                mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                corrected_mean = mean / (1 - beta1 ** state["step"])
                corrected_std = variance.sqrt() / math.sqrt(1 - beta2 ** state["step"])
                # Decay the parameter directly; it does not enter either moment.
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(
                    corrected_mean, corrected_std.add(group["eps"]), value=-group["lr"]
                )
        return loss
