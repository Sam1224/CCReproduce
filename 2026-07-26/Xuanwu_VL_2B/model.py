import torch
import torch.nn as nn
import torch.nn.functional as functional


class VisualEncoder(nn.Module):
    """Tiny CNN standing in for a general multimodal visual foundation encoder."""

    def __init__(self, hidden_dim: int = 64, fine_dim: int = 6) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projector = nn.Sequential(nn.Flatten(), nn.Linear(48, hidden_dim), nn.ReLU())
        self.fine_head = nn.Linear(hidden_dim, fine_dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.projector(self.backbone(image))
        fine_logits = self.fine_head(feature)
        return feature, fine_logits


class TokenEncoder(nn.Module):
    """Shared light text/OCR encoder with PAD masking."""

    def __init__(self, vocab_size: int, embed_dim: int = 48, hidden_dim: int = 64, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(self.pad_id)
        embedded = self.embedding(tokens)
        encoded, _ = self.gru(embedded)
        encoded = encoded * mask.unsqueeze(-1).float()
        pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return self.proj(pooled)


class GatedFusion(nn.Module):
    """Fine-grained multimodal fusion with modality gates.

    The gate learns when to trust visual evidence, user text, or OCR, which mimics the
    paper story of evolving a general multimodal base into an industrial moderation
    foundation with detailed perception and robust text-in-image handling.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

    def forward(self, visual: torch.Tensor, text: torch.Tensor, ocr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([visual, text, ocr], dim=1)
        gate_logits = self.gate(torch.cat([visual, text, ocr], dim=-1))
        weights = torch.softmax(gate_logits, dim=-1)
        gated = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.mixer(torch.cat([visual, text, ocr, gated], dim=-1))
        return fused, weights


class XuanwuVL2BToy(nn.Module):
    """Lightweight content-governance VLM toy model.

    Outputs include coarse policy logits, fine-grained evidence logits, OCR-clean vs
    OCR-adversarial consistency tensors, and a compact deployment embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        fine_dim: int = 6,
        hidden_dim: int = 64,
        embed_dim: int = 48,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder(hidden_dim=hidden_dim, fine_dim=fine_dim)
        self.text_encoder = TokenEncoder(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, pad_id=pad_id)
        self.ocr_encoder = TokenEncoder(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, pad_id=pad_id)
        self.fusion = GatedFusion(hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.deploy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode(self, image: torch.Tensor, text_tokens: torch.Tensor, ocr_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        visual, fine_logits = self.visual_encoder(image)
        text = self.text_encoder(text_tokens)
        ocr = self.ocr_encoder(ocr_tokens)
        fused, modality_weights = self.fusion(visual, text, ocr)
        logits = self.classifier(fused)
        deploy_logits = self.deploy_head(fused.detach() + 0.0)
        return {
            "visual": visual,
            "text": text,
            "ocr": ocr,
            "fused": fused,
            "logits": logits,
            "deploy_logits": deploy_logits,
            "fine_logits": fine_logits,
            "modality_weights": modality_weights,
        }

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        ocr_tokens: torch.Tensor,
        adv_ocr_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        clean = self.encode(image, text_tokens, ocr_tokens)
        if adv_ocr_tokens is not None:
            adv = self.encode(image, text_tokens, adv_ocr_tokens)
            clean["adv_logits"] = adv["logits"]
            clean["adv_fused"] = adv["fused"]
            clean["adv_modality_weights"] = adv["modality_weights"]
        return clean


def stage_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, fine_labels: torch.Tensor, stage: str = "mid") -> tuple[torch.Tensor, dict[str, float]]:
    """Three-stage training interface.

    - pre: align general multimodal features and warm up coarse policy classes.
    - mid: emphasize fine-grained evidence perception and normal classification.
    - post: add adversarial OCR consistency and deploy-head distillation.
    """

    ce = functional.cross_entropy(outputs["logits"], labels)
    fine = functional.binary_cross_entropy_with_logits(outputs["fine_logits"], fine_labels)
    align = 1.0 - functional.cosine_similarity(outputs["visual"], outputs["ocr"], dim=-1).mean()
    deploy = functional.kl_div(
        functional.log_softmax(outputs["deploy_logits"], dim=-1),
        functional.softmax(outputs["logits"].detach(), dim=-1),
        reduction="batchmean",
    )
    consistency = torch.zeros((), device=labels.device)
    if "adv_logits" in outputs:
        clean_log_prob = functional.log_softmax(outputs["logits"], dim=-1)
        adv_prob = functional.softmax(outputs["adv_logits"].detach(), dim=-1)
        consistency = functional.kl_div(clean_log_prob, adv_prob, reduction="batchmean")

    if stage == "pre":
        loss = 0.55 * ce + 0.35 * align + 0.10 * fine
    elif stage == "mid":
        loss = ce + 0.65 * fine + 0.10 * align
    elif stage == "post":
        loss = ce + 0.45 * fine + 0.55 * consistency + 0.25 * deploy
    else:
        raise ValueError(f"unknown stage: {stage}")

    logs = {
        "loss": float(loss.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "fine": float(fine.detach().cpu()),
        "align": float(align.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "deploy": float(deploy.detach().cpu()),
    }
    return loss, logs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    model = XuanwuVL2BToy(vocab_size=32)
    image = torch.randn(2, 3, 32, 32)
    tokens = torch.ones(2, 8, dtype=torch.long)
    result = model(image, tokens, tokens, tokens)
    print(result["logits"].shape, result["fine_logits"].shape, count_parameters(model))
