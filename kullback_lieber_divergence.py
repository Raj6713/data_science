import torch


class KLDivergence:
    def __init__(
        self, reduction: str = "mean", log_input: bool = False, eps: float = 1e-10
    ):
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction
        self.log_input = log_input
        self.eps = eps

    def __call__(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        P = P + self.eps
        if self.log_input:
            log_Q = Q
        else:
            log_Q = torch.log(Q + self.eps)
        log_P = torch.log(P)
        kl = (P * (log_P - log_Q)).sum(dim=1)

        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        else:
            return kl


if __name__ == "__main__":
    P = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float32)
    Q = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]], dtype=torch.float32)
    kl_loss = KLDivergence(reduction="mean", log_input=False)
    loss = kl_loss(P, Q)
    print("KL Divergence", loss.item())
