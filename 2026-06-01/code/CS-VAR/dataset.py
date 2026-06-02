"""
Toy dataset for CS-VAR live streaming risk assessment.
Each sample represents a single live streaming session with behavioral events.
"""
import torch
from torch.utils.data import Dataset, DataLoader


class ToySessionDataset(Dataset):
    """
    Simulates live streaming sessions with behavioral event sequences.

    In production:
      event_dim features might include:
        - normalized timestamps of actions
        - one-hot encoded action types (comment, gift, product click, host action)
        - interaction counts per time window
        - text embedding of key comments
    """

    def __init__(
        self,
        num_sessions: int = 500,
        event_dim: int = 64,
        seq_len: int = 64,
        num_risk_levels: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        torch.manual_seed(seed)
        # Simulate session event sequences
        self.events = torch.randn(num_sessions, seq_len, event_dim)
        self.risk_labels = torch.randint(0, num_risk_levels, (num_sessions,))
        # Binary: high-risk vs. low/medium
        self.is_high_risk = (self.risk_labels == 2).long()
        self.session_ids = [f"session_{i:06d}" for i in range(num_sessions)]

    def __len__(self):
        return len(self.risk_labels)

    def __getitem__(self, idx):
        return {
            "events": self.events[idx],
            "risk_label": self.risk_labels[idx],
            "is_high_risk": self.is_high_risk[idx],
            "session_id": self.session_ids[idx],
        }


def get_dataloaders(
    batch_size: int = 32,
    event_dim: int = 64,
    seq_len: int = 64,
    num_risk_levels: int = 3,
):
    train_ds = ToySessionDataset(500, event_dim, seq_len, num_risk_levels, seed=42)
    val_ds = ToySessionDataset(100, event_dim, seq_len, num_risk_levels, seed=99)
    test_ds = ToySessionDataset(100, event_dim, seq_len, num_risk_levels, seed=77)
    history_ds = ToySessionDataset(1000, event_dim, seq_len, num_risk_levels, seed=7)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        DataLoader(history_ds, batch_size=batch_size),
    )
