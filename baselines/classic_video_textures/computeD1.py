import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_D1(
    frames: torch.Tensor,
    sigma_factor: float,
    feats: str = "RGB",
    audio: np.ndarray = None,
    sr: int = 0,
    fps: int = 30,
    slow: bool = True,
    batch_size: int = 64,
) -> tuple:
    """
    Compute pairwise frame distance matrix D1 and transition probability P1.

    Args:
        frames: tensor of shape [N, C, H, W]
        sigma_factor: controls bandwidth of Gaussian kernel
        feats: 'RGB' or 'ResNet'
        slow: if True, use batched computation to save memory

    Returns:
        D1, P1, sigma
    """
    frames = frames.to(DEVICE)

    if feats == "RGB":
        if not slow:
            A = frames.unsqueeze(0).expand(frames.shape[0], -1, -1, -1, -1)
            A = A.reshape(frames.shape[0], frames.shape[0], -1)
            B = frames.unsqueeze(1).expand(-1, frames.shape[0], -1, -1, -1)
            B = B.reshape(frames.shape[0], frames.shape[0], -1)
            D1 = torch.norm(A - B, dim=2)
        else:
            N = len(frames)
            D1 = torch.zeros((N, N), device=DEVICE)
            for i in range(0, N, batch_size):
                batch_A = frames[i: i + batch_size]
                for j in range(0, N, batch_size):
                    batch_B = frames[j: j + batch_size]
                    fa = batch_A.reshape(len(batch_A), -1).unsqueeze(1)
                    fb = batch_B.reshape(len(batch_B), -1).unsqueeze(0)
                    D1[i: i + len(batch_A), j: j + len(batch_B)] = torch.norm(fa - fb, dim=2)

    elif feats == "ResNet":
        resnet = models.resnet18(pretrained=True).to(DEVICE)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        resnet.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        N = len(frames)
        all_feats = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = frames[i: i + batch_size].float() / 255.0
                batch = torch.stack([normalize(f) for f in batch])
                f = resnet(batch).squeeze(-1).squeeze(-1)
                all_feats.append(f.cpu())
        all_feats = torch.cat(all_feats, dim=0)  # [N, 512]

        fa = F.normalize(all_feats, dim=1).unsqueeze(1)  # [N, 1, 512]
        fb = F.normalize(all_feats, dim=1).unsqueeze(0)  # [1, N, 512]
        D1 = torch.norm(fa - fb, dim=2).to(DEVICE)

    else:
        raise ValueError(f"Unknown feats type: {feats}. Use 'RGB' or 'ResNet'.")

    non_zero_count = (D1 != 0).sum().item()
    if non_zero_count == 0:
        non_zero_count = 1
    sigma = sigma_factor * (D1.sum() / non_zero_count)

    P1 = torch.exp(-D1 / sigma)
    P1 = torch.cat((P1[1:, :], P1[-1, :].unsqueeze(0)), dim=0)
    P1 = P1 / (P1.sum(1, keepdim=True) + 1e-8)

    return D1.cpu(), P1.cpu(), sigma.cpu()
