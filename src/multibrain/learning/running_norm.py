"""Running observation normalization (PLAN.md §7: 観測は移動統計で正規化).

Welford's online mean/variance per feature. ``normalize`` maps to roughly
unit scale and clamps to +/-clip (10 per spec). ``freeze`` stops updates
so evaluation runs on fixed statistics. Non-finite inputs are replaced
before they can poison the statistics or the normalized output.
"""

import torch
from torch import nn


class RunningNorm(nn.Module):
    """Per-feature running mean/variance; buffers ride state_dict."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        # small initial count so the first real batch dominates
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = float(clip)
        self.frozen = False

    def update(self, x: torch.Tensor) -> None:
        """Fold one batch (..., dim) into the running statistics."""
        if self.frozen:
            return
        with torch.no_grad():
            x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0,
                                 neginf=0.0)
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            n = x.shape[0]
            if n == 0:
                return
            bm = x.mean(dim=0)
            bv = x.var(dim=0, unbiased=False)
            delta = bm - self.mean
            tot = self.count + n
            m2 = (self.var * self.count + bv * n
                  + delta * delta * self.count * n / tot)
            self.mean += delta * n / tot
            self.var = m2 / tot
            self.count.fill_(tot)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """clamp((x - mean) / std, -clip, +clip); NaN/inf -> 0."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = (self.var + 1e-8).sqrt()
        return ((x - self.mean) / std).clamp(-self.clip, self.clip)

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False
