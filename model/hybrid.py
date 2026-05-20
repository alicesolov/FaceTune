"""Hybrid FFT+CLIP binary classifier for AI image detection.

Architecture:
  - FFT branch:  ResNet-50 backbone (without FC) → 2048-dim embedding
  - CLIP branch: frozen ViT-B/32 image encoder  →  512-dim embedding
  - Fusion:      concat(2560) → Linear(512) → ReLU → Dropout → Linear(2)

Training phases:
  Phase 1: freeze both backbones, train fusion only  (lr=1e-3, 3 epochs)
  Phase 2: unfreeze ResNet-50 layer4, keep CLIP frozen (lr=3e-5, 5 epochs)
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchvision.models import ResNet50_Weights, resnet50
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision required for HybridDetector") from exc

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers required for HybridDetector (pip install transformers)") from exc

FFT_EMBED_DIM = 2048
CLIP_EMBED_DIM = 512
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class HybridDetector(nn.Module):
    """Dual-branch AI image detector combining spectral and semantic features."""

    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL_NAME,
        fusion_hidden: int = 512,
        dropout: float = 0.3,
        freeze_fft: bool = True,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()

        # ── FFT branch (ResNet-50 without FC) ──────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove only the final FC; keep avgpool → output (B, 2048, 1, 1)
        self.fft_encoder = nn.Sequential(*list(backbone.children())[:-1])

        # ── CLIP branch ────────────────────────────────────────────────────
        self.clip_encoder = CLIPVisionModel.from_pretrained(clip_model_name)

        # ── Fusion MLP ────────────────────────────────────────────────────
        total_dim = FFT_EMBED_DIM + CLIP_EMBED_DIM
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 2),
        )

        if freeze_fft:
            self._set_grad(self.fft_encoder, False)
        if freeze_clip:
            self._set_grad(self.clip_encoder, False)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        fft_tensor: torch.Tensor,
        clip_pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Return 2-logit tensor (B, 2).

        Args:
            fft_tensor:         (B, 3, 256, 256) FFT-preprocessed images.
            clip_pixel_values:  (B, 3, 224, 224) images normalised for CLIP.
        """
        fft_feat = self.fft_encoder(fft_tensor).flatten(1)       # (B, 2048)
        clip_feat = self.clip_encoder(
            pixel_values=clip_pixel_values
        ).pooler_output                                            # (B, 512)
        x = torch.cat([fft_feat, clip_feat], dim=1)              # (B, 2560)
        return self.fusion(x)                                     # (B, 2)

    # ── Phase control ─────────────────────────────────────────────────────

    def unfreeze_fft_last_block(self) -> None:
        """Unfreeze ResNet-50 layer4 for phase-2 fine-tuning."""
        # children: conv1 bn1 relu maxpool layer1 layer2 layer3 layer4 avgpool
        layer4 = list(self.fft_encoder.children())[-2]
        self._set_grad(layer4, True)

    def freeze_all_backbones(self) -> None:
        self._set_grad(self.fft_encoder, False)
        self._set_grad(self.clip_encoder, False)

    # ── Optimizer helpers ─────────────────────────────────────────────────

    def get_parameter_groups(
        self,
        lr_fusion: float = 1e-3,
        lr_backbone: float = 3e-5,
    ) -> list[dict]:
        """Return per-group lr dicts for torch.optim."""
        groups: list[dict] = [
            {"params": list(self.fusion.parameters()), "lr": lr_fusion}
        ]
        fft_trainable = [p for p in self.fft_encoder.parameters() if p.requires_grad]
        clip_trainable = [p for p in self.clip_encoder.parameters() if p.requires_grad]
        if fft_trainable:
            groups.append({"params": fft_trainable, "lr": lr_backbone})
        if clip_trainable:
            groups.append({"params": clip_trainable, "lr": lr_backbone})
        return groups

    def embedding_shapes(
        self,
        device: torch.device | None = None,
    ) -> dict[str, tuple]:
        """Return embedding shapes for a dummy forward pass (useful for debugging)."""
        dev = device or next(self.parameters()).device
        dummy_fft = torch.zeros(1, 3, 256, 256, device=dev)
        dummy_clip = torch.zeros(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            fft_f = self.fft_encoder(dummy_fft).flatten(1)
            clip_f = self.clip_encoder(pixel_values=dummy_clip).pooler_output
            logits = self.fusion(torch.cat([fft_f, clip_f], dim=1))
        return {
            "fft_embedding": tuple(fft_f.shape),
            "clip_embedding": tuple(clip_f.shape),
            "logits": tuple(logits.shape),
        }

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _set_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad


def load_hybrid_weights(model: HybridDetector, weights_path: str) -> HybridDetector:
    """Load a saved hybrid checkpoint into an existing HybridDetector."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model
