from torch import nn


class LossFactory:
    @staticmethod
    def build(name: str) -> nn.Module:
        match name:
            case "l1":
                criteria = nn.L1Loss()
            case _:
                raise ValueError(f"Not supported criteria {name}")
        return criteria
