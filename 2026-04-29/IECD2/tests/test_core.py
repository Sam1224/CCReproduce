import torch

from iecd2.core import IECDConfig, iecd_fuse_logits


def test_gate_is_half_when_distributions_match() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    result = iecd_fuse_logits(logits, logits, config=IECDConfig(eta=-3.0))
    assert torch.allclose(result.symmetric_kl, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(result.gate_g, torch.tensor([0.5]), atol=1e-6)


def test_gate_prefers_evidence_when_disagreement_high_and_eta_negative() -> None:
    logits_i = torch.tensor([[10.0, 0.0, 0.0]])
    logits_e = torch.tensor([[0.0, 10.0, 0.0]])
    result = iecd_fuse_logits(logits_i, logits_e, config=IECDConfig(eta=-10.0))

    # Big disagreement -> g ~ 0 -> fused distribution close to evidence.
    assert float(result.gate_g.item()) < 0.05

    p_e = torch.softmax(logits_e / 0.9, dim=-1)
    assert torch.allclose(result.fused_probs, p_e, atol=1e-3)
