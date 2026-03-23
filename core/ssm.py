"""
Minimal CPU-compatible Selective State Space Model (Mamba-inspired).

This implementation reproduces the core Mamba selective scan mechanism in pure
PyTorch — no CUDA kernels, no mamba-ssm dependency. Runs on any machine.

It is intentionally simplified for the playground:
- Sequential scan (not parallel) → correct but slower than the CUDA version
- No conv1d input projection → simpler code, same representational capacity for short sequences
- Standard LayerNorm residual → standard practice

For production use on GPU, replace with mamba-ssm which has optimized CUDA kernels.
The interface is identical: same forward() signature, same output shape.

Architecture:
    SelectiveSSMBlock   — one Mamba-style block (project → SSM → gate → project)
    StackedSSM          — N blocks stacked with residuals
    SSMClassifier       — StackedSSM + pooling + linear head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """
    One Mamba-style selective SSM block.

    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        # Input expansion + gate
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # SSM parameters
        # A: initialized as range [1, d_state] — standard Mamba init
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(A_init)

        # Selective parameters (depend on input x)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Skip connection
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        residual = x

        # Expand and split into content + gate
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        x_content, z = xz.chunk(2, dim=-1)            # each (B, L, d_inner)
        x_content = F.silu(x_content)

        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_content)                # (B, L, 2*d_state + 1)
        dt_raw, B_mat, C_mat = x_dbl.split(
            [1, self.d_state, self.d_state], dim=-1
        )
        # dt: controls how much the state updates (input-dependent time step)
        dt = F.softplus(self.dt_proj(dt_raw))          # (B, L, d_inner)

        # Fixed SSM matrix A (negative to ensure stability)
        A = -torch.exp(self.A_log)                     # (d_state,)

        # Sequential selective scan (CPU-friendly)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            dt_t = dt[:, t, :]                         # (B, d_inner)
            B_t = B_mat[:, t, :]                       # (B, d_state)
            C_t = C_mat[:, t, :]                       # (B, d_state)
            u_t = x_content[:, t, :]                   # (B, d_inner)

            # Discretize: zero-order hold
            dA = torch.exp(dt_t.unsqueeze(-1) * A)     # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, d_inner, d_state)

            # State update: h = A*h + B*u
            h = h * dA + dB * u_t.unsqueeze(-1)        # (B, d_inner, d_state)

            # Output: y = C*h + D*u
            y_t = (h * C_t.unsqueeze(1)).sum(-1) + self.D * u_t  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                     # (B, L, d_inner)

        # Multiplicative gate (SiLU)
        y = y * F.silu(z)

        # Project back and apply residual
        y = self.out_proj(y)
        return self.norm(y + residual)


class StackedSSM(nn.Module):
    """N SelectiveSSMBlocks stacked."""

    def __init__(self, d_model: int, n_layers: int = 2, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            SelectiveSSMBlock(d_model, d_state, expand)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SSMClassifier(nn.Module):
    """
    Sequence classifier using a stacked SSM.

    Uses the last token's representation for classification (causal model —
    the final state summarizes all previous events).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_classes: int,
        n_layers: int = 2,
        d_state: int = 16,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.ssm = StackedSSM(d_model, n_layers, d_state)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, n_classes) — logits
        """
        x = self.input_proj(x)          # project to d_model
        x = self.ssm(x)                 # process sequence
        x = x[:, -1, :]                 # last token = summary of full sequence
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        return self.forward(x).argmax(dim=-1)


class SSMAnomalyDetector(nn.Module):
    """
    Per-timestep anomaly scorer.

    Outputs a scalar anomaly score at each position in the sequence.
    Score > threshold = anomaly detected at that timestep.
    """

    def __init__(self, input_dim: int, d_model: int, n_layers: int = 2, d_state: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.ssm = StackedSSM(d_model, n_layers, d_state)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len) — anomaly scores [0, 1]
        """
        x = self.input_proj(x)
        x = self.ssm(x)
        return self.scorer(x).squeeze(-1)


class SSMRouter(nn.Module):
    """
    Learned routing model for agent-pool.

    Takes a sequence of past routing events and predicts which worker
    to select for the next request (as a worker index).
    """

    def __init__(self, event_dim: int, d_model: int, n_workers: int, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(event_dim, d_model)
        self.ssm = StackedSSM(d_model, n_layers, d_state=16)
        self.router_head = nn.Linear(d_model, n_workers)
        self.n_workers = n_workers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, event_dim)
        returns: (batch, n_workers) — logits over workers
        """
        x = self.input_proj(x)
        x = self.ssm(x)
        return self.router_head(x[:, -1, :])

    def route(self, x: torch.Tensor, n_workers: int | None = None) -> int:
        """Return the index of the recommended worker."""
        with torch.no_grad():
            logits = self.forward(x)
            return int(logits[0].argmax().item())


class SSMQualityPredictor(nn.Module):
    """
    Prediction-based worker router.

    Instead of imitating an oracle (SSMRouter), this model predicts the
    composite quality score [0, 1] for each worker from event history, then
    routes to the worker with the highest predicted quality.

    Training uses masked MSE: only the observed worker's quality is supervised
    per step, so random/round-robin data collection gives unbiased coverage of
    all workers without requiring counterfactual rollouts.

    This design answers a different question than imitation learning:
      SSMRouter:         "what would cost-aware do here?"
      SSMQualityPredictor: "which worker will actually perform best here?"
    """

    def __init__(self, event_dim: int, d_model: int, n_workers: int, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(event_dim, d_model)
        self.ssm = StackedSSM(d_model, n_layers, d_state=16)
        self.quality_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_workers),
            nn.Sigmoid(),
        )
        self.n_workers = n_workers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, event_dim)
        returns: (batch, n_workers) — predicted quality scores in [0, 1]
        """
        x = self.input_proj(x)
        x = self.ssm(x)
        return self.quality_head(x[:, -1, :])

    def route(self, x: torch.Tensor) -> int:
        """Return the index of the worker with the highest predicted quality."""
        with torch.no_grad():
            quality = self.forward(x)
            return int(quality[0].argmax().item())
