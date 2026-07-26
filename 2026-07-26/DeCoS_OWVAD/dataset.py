import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class DefinitionBank:
    normal_embedding: torch.Tensor
    anomaly_embeddings: torch.Tensor
    shared_direction: torch.Tensor
    class_directions: torch.Tensor


@dataclass
class ToySample:
    visual_features: torch.Tensor
    frame_labels: torch.Tensor
    event_spans: torch.Tensor
    is_multi_event: torch.Tensor


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def build_definition_bank(num_classes: int = 4, feature_dim: int = 64, seed: int = 7) -> DefinitionBank:
    generator = torch.Generator().manual_seed(seed)
    normal_embedding = _normalize(torch.randn(feature_dim, generator=generator))
    anomaly_embeddings = _normalize(torch.randn(num_classes, feature_dim, generator=generator))
    shared_direction = _normalize(anomaly_embeddings.sum(dim=0))
    class_directions = _normalize(anomaly_embeddings + 0.35 * shared_direction.unsqueeze(0))
    return DefinitionBank(
        normal_embedding=normal_embedding,
        anomaly_embeddings=anomaly_embeddings,
        shared_direction=shared_direction,
        class_directions=class_directions,
    )


class DeCoSToyDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 256,
        seq_len: int = 64,
        feature_dim: int = 64,
        num_classes: int = 4,
        definition_bank: DefinitionBank | None = None,
        multi_event_ratio: float = 0.4,
        beta: float = 2.0,
        gamma: float = 0.45,
        noise_scale: float = 0.18,
        seed: int = 13,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.definition_bank = definition_bank or build_definition_bank(num_classes, feature_dim, seed)
        self.multi_event_ratio = multi_event_ratio
        self.beta = beta
        self.gamma = gamma
        self.noise_scale = noise_scale
        self.samples = self._build_samples(seed)

    def _make_normal_frame(self) -> torch.Tensor:
        return self.definition_bank.normal_embedding + self.noise_scale * torch.randn(self.feature_dim)

    def _make_anomaly_frame(self, class_id: int) -> torch.Tensor:
        anomaly_vector = (
            self.definition_bank.normal_embedding
            + self.beta * self.definition_bank.shared_direction
            + self.gamma * self.definition_bank.class_directions[class_id]
            + self.noise_scale * torch.randn(self.feature_dim)
        )
        return anomaly_vector

    def _place_event(self, features: torch.Tensor, labels: torch.Tensor, start: int, length: int, class_id: int) -> None:
        for index in range(start, start + length):
            features[index] = self._make_anomaly_frame(class_id)
            labels[index] = class_id + 1

    def _build_samples(self, seed: int) -> list[ToySample]:
        rng = random.Random(seed)
        samples: list[ToySample] = []
        event_length = max(8, self.seq_len // 4)
        gap = max(6, self.seq_len // 8)

        for _ in range(self.num_samples):
            features = torch.stack([self._make_normal_frame() for _ in range(self.seq_len)])
            labels = torch.zeros(self.seq_len, dtype=torch.long)
            spans = torch.full((2, 3), -1, dtype=torch.long)

            is_multi_event = rng.random() < self.multi_event_ratio
            first_class = rng.randrange(self.num_classes)
            first_start = rng.randint(4, self.seq_len - event_length - 4)
            self._place_event(features, labels, first_start, event_length, first_class)
            spans[0] = torch.tensor([first_start, first_start + event_length, first_class + 1], dtype=torch.long)

            if is_multi_event:
                second_class_candidates = [value for value in range(self.num_classes) if value != first_class]
                second_class = rng.choice(second_class_candidates)
                second_min = min(self.seq_len - event_length - 2, first_start + event_length + gap)
                second_max = self.seq_len - event_length - 2
                if second_min > second_max:
                    second_min = max(2, first_start - event_length - gap)
                    second_max = max(second_min, first_start - gap)
                second_start = rng.randint(max(2, second_min), max(2, second_max))
                self._place_event(features, labels, second_start, event_length, second_class)
                spans[1] = torch.tensor([second_start, second_start + event_length, second_class + 1], dtype=torch.long)

            features = _normalize(features)
            samples.append(
                ToySample(
                    visual_features=features,
                    frame_labels=labels,
                    event_spans=spans,
                    is_multi_event=torch.tensor(int(is_multi_event), dtype=torch.long),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "visual_features": sample.visual_features.clone(),
            "frame_labels": sample.frame_labels.clone(),
            "event_spans": sample.event_spans.clone(),
            "is_multi_event": sample.is_multi_event.clone(),
        }


if __name__ == "__main__":
    dataset = DeCoSToyDataset(num_samples=4)
    batch = dataset[0]
    print(batch["visual_features"].shape)
    print(batch["frame_labels"].unique(sorted=True))
    print(batch["event_spans"])
