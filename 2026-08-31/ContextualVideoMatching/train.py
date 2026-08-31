import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from dataset import MatchingDataset, get_mock_data
from model import ContextualVideoMatcher, contrastive_training_loss


def collate(batch):
    article_tokens, video_tokens = zip(*batch)
    article_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in article_tokens]
    video_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in video_tokens]
    return article_tensors, video_tensors


def encode_batch(model, token_lists):
    return model.encode_texts([tokens.tolist() for tokens in token_lists])


def train(epochs: int = 10) -> None:
    samples, _, _ = get_mock_data()
    loader = DataLoader(MatchingDataset(samples), batch_size=3, shuffle=True, collate_fn=collate)
    model = ContextualVideoMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        total_loss = 0.0
        for article_tokens, video_tokens in loader:
            article_vectors = encode_batch(model, article_tokens)
            video_vectors = encode_batch(model, video_tokens)
            loss = contrastive_training_loss(article_vectors, video_vectors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} loss={total_loss / max(len(loader), 1):.4f}")
    torch.save(model.state_dict(), "contextual_video_matcher.pt")


if __name__ == "__main__":
    train()
