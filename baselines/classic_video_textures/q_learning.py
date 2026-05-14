import copy
import torch


def q_learning(
    D2: torch.Tensor,
    sigma_factor: float,
    p: float = 0.7,
    alpha: float = 0.997,
    thresholding: float = 0.75,
    max_iters: int = 50,
) -> tuple:
    """
    Refine transition costs using Q-learning to prefer transitions that
    lead to low-cost futures, not just low one-step cost.

    Args:
        D2: smoothed distance matrix [N, N]
        sigma_factor: Gaussian bandwidth scale
        p: cost exponent
        alpha: future discount factor
        thresholding: fraction of max probability to zero out (sparsify P)
        max_iters: cap on iterations to prevent runaway loops

    Returns:
        D3, P3, P3_new (thresholded), sigma
    """
    D2 = D2.float()
    D3 = D2 ** p
    D3_new = D3.clone()

    eps = float("inf")
    itr = 0
    while eps > 1e-2 and itr < max_iters:
        D3_old = D3_new.clone()

        # Off-diagonal min future cost for each row
        N = D3.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool)
        for i in range(N - 1, 0, -1):
            off_diag = D3_old[mask.reshape(N, N)[i]].reshape(-1)
            min_future = off_diag.min()
            D3_new[i] = D3[i] + alpha * min_future

        eps = ((D3_new - D3_old) ** 2).mean().item()
        itr += 1
        print(f"  Q-learning iter {itr}, eps={eps:.5f}")

    non_zero_count = (D3_new != 0).sum().item()
    if non_zero_count == 0:
        non_zero_count = 1
    sigma = sigma_factor * (D3_new.sum() / non_zero_count)

    P3 = torch.exp(-D3_new / sigma)
    P3 = torch.cat((P3[1:, :], P3[-1, :].unsqueeze(0)), dim=0)
    P3 = P3 / (P3.sum(1, keepdim=True) + 1e-8)

    P3_new = P3.clone()
    for i in range(len(P3_new)):
        threshold = P3_new[i].max() - thresholding * P3_new[i].max()
        P3_new[i][P3_new[i] < threshold] = 0.0
    # Re-normalize after thresholding
    row_sums = P3_new.sum(1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    P3_new = P3_new / row_sums

    print(f"  Avg non-zero transitions per row: {(P3_new > 0).float().sum(1).mean():.1f}")

    return D3_new, P3, P3_new, sigma
