import torch
import torch.nn.functional as F


class GatedFeedForward(torch.nn.Module):
    """SwiGLU or GeGLU: down(activation(gate(x)) * up(x)).

    Input/output: (..., model_dim). hidden_dim sets the intermediate capacity.
    The two input projections are fused; cost is O(tokens*model_dim*hidden_dim).
    """

    def __init__(self, model_dim, hidden_dim, *, activation="swiglu"):
        super().__init__()
        if min(model_dim, hidden_dim) <= 0 or activation not in ("swiglu", "geglu"):
            raise ValueError("positive dimensions and swiglu/geglu activation required")
        self.activation = activation
        self.gate_up = torch.nn.Linear(model_dim, 2 * hidden_dim, bias=False)
        self.down = torch.nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        gate = F.silu(gate) if self.activation == "swiglu" else F.gelu(gate)
        return self.down(gate * up)
