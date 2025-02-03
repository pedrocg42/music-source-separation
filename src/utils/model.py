import torch
from torch.utils.flop_counter import FlopCounterMode


def get_flops(model: torch.nn.Module, inp: torch.Tensor, with_backward=False) -> int:
    model.eval()
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    with FlopCounterMode(model, display=True) as flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)

    total_flops = flop_counter.get_total_flops()
    return total_flops


def get_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
