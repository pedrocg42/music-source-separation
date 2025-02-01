import torch


def cosine_loss() -> torch.nn.CosineSimilarity:
    """
    Create a cosine similarity-based loss function.

    Returns:
    torch.nn.CosineSimilarity: Cosine similarity function with an embedded loss function.
    """
    cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    """
    (1 - cosine) to have a loss between 2 and 0.
    Minimizing it will do the signals be close together
    """

    def loss_fn(y_pred, y_true):
        return 1 - cosine(y_pred, y_true)

    return loss_fn
