import numpy as np
import torch
import torch.nn.functional as F


def compute_D2(
    D1: torch.Tensor,
    sigma_factor: float,
    filter_size: int = 16,
    stride: int = 1,
) -> tuple:
    """
    Smooth D1 with a diagonal binomial filter to enforce temporal continuity.

    The filter rewards transitions where neighboring frames also match well,
    not just the single pair (i, j).

    Args:
        D1: pairwise distance matrix [N, N]
        sigma_factor: Gaussian bandwidth scale
        filter_size: size of the binomial smoothing kernel
        stride: convolution stride (used for Classic++ variant)

    Returns:
        D2, P2, sigma, binomial_filter
    """
    coeffs = (np.poly1d([0.5, 0.5]) ** (filter_size - 1)).coeffs
    binomial_filter = torch.tensor(np.diag(coeffs), dtype=torch.float32)

    D2 = D1.float().unsqueeze(0).unsqueeze(0)          # [1, 1, N, N]
    kernel = binomial_filter.unsqueeze(0).unsqueeze(0)  # [1, 1, fs, fs]

    D2 = F.conv2d(D2, kernel, stride=stride)
    D2 = D2.squeeze(0).squeeze(0)                       # [N', N']

    non_zero_count = (D2 != 0).sum().item()
    if non_zero_count == 0:
        non_zero_count = 1
    sigma = sigma_factor * (D2.sum() / non_zero_count)

    P2 = torch.exp(-D2 / sigma)
    P2 = torch.cat((P2[1:, :], P2[-1, :].unsqueeze(0)), dim=0)
    P2 = P2 / (P2.sum(1, keepdim=True) + 1e-8)

    return D2, P2, sigma, binomial_filter
