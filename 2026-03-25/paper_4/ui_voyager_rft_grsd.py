import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Paper: UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience
Summary: Two-stage training: Rejection Fine-Tuning (RFT) + Group Relative Self-Distillation (GRSD).
Core: Use failed trajectories as signal by extracting dense step-level supervision from successful peers.

Note:
This is a toy reproduction of the training *mechanics* with a minimal policy network.
"""


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=256, act_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.GELU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def rejection_fine_tuning_loss(logits, actions, trajectory_score, threshold=0.5):
    """Keep only good trajectories (simple RFT gate)."""

    keep = trajectory_score >= threshold
    if keep.sum() == 0:
        return logits.sum() * 0

    logits = logits[keep]
    actions = actions[keep]
    return F.cross_entropy(logits, actions)


def grsd_step_distill_loss(success_logits, fail_logits, temperature=1.0):
    """Dense step supervision: match failed-step policy to successful policy at shared states."""

    p = F.log_softmax(fail_logits / temperature, dim=-1)
    q = F.softmax(success_logits / temperature, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * (temperature ** 2)


class UIVoyager(nn.Module):
    def __init__(self, obs_dim=256, act_dim=64):
        super().__init__()
        self.policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim)

    def forward(self, obs):
        return self.policy(obs)


if __name__ == "__main__":
    torch.manual_seed(0)

    bsz, obs_dim, act_dim = 8, 256, 64
    obs = torch.randn(bsz, obs_dim)
    actions = torch.randint(0, act_dim, (bsz,))

    model = UIVoyager(obs_dim=obs_dim, act_dim=act_dim)
    logits = model(obs)

    traj_score = torch.rand(bsz)
    rft_loss = rejection_fine_tuning_loss(logits, actions, traj_score, threshold=0.6)

    success = torch.randn(bsz, act_dim)
    fail = torch.randn(bsz, act_dim)
    grsd_loss = grsd_step_distill_loss(success, fail)

    print("rft_loss:", float(rft_loss))
    print("grsd_loss:", float(grsd_loss))
    print("UI-Voyager reproduction structure complete.")
