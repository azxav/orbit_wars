from __future__ import annotations

import torch
from torch import nn

from .config import BCModelConfig


class EntityBCPolicy(nn.Module):
    def __init__(self, config: BCModelConfig):
        super().__init__()
        self.config = config
        h = int(config.hidden_size)
        self.planet_encoder = nn.Linear(config.planet_feature_dim, h)
        self.global_encoder = nn.Linear(config.global_feature_dim, h)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(config.num_heads),
            dim_feedforward=int(config.mlp_size),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(config.num_layers))
        self.target_pair = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1))
        self.noop_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.noop_target_context = nn.Parameter(torch.zeros(h))
        self.amount_head = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, config.amount_bins))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        planets = batch["planet_features"]
        global_features = batch["global_features"]
        bsz, pmax, _ = planets.shape
        planet_tokens = self.planet_encoder(planets)
        global_token = self.global_encoder(global_features).unsqueeze(1)
        encoded = self.encoder(torch.cat([global_token, planet_tokens], dim=1))
        planet_ctx = encoded[:, 1 : pmax + 1]
        source_idx = batch["source_slot"].clamp(0, pmax - 1)
        source_ctx = planet_ctx[torch.arange(bsz, device=planets.device), source_idx]
        source_expanded = source_ctx.unsqueeze(1).expand(-1, pmax, -1)
        target_logits = self.target_pair(torch.cat([source_expanded, planet_ctx], dim=-1)).squeeze(-1)
        noop_logit = self.noop_head(source_ctx)
        target_logits = torch.cat([target_logits, noop_logit], dim=1)
        teacher_target = batch.get("target_label")
        if teacher_target is None:
            teacher_target = target_logits.argmax(dim=1)
        target_idx = teacher_target.clamp(0, pmax - 1)
        target_ctx = planet_ctx[torch.arange(bsz, device=planets.device), target_idx]
        noop_ctx = self.noop_target_context.unsqueeze(0).expand(bsz, -1)
        target_ctx = torch.where((teacher_target == self.config.noop_target_slot).unsqueeze(1), noop_ctx, target_ctx)
        amount_logits = self.amount_head(torch.cat([source_ctx, target_ctx], dim=1))
        return {"target_logits": target_logits, "amount_logits": amount_logits}

