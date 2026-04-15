import torch
import torch.optim as optim
from model import MOON

def train():
    model = MOON()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss() # Placeholder for contrastive/generative loss
    
    # Toy Data
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (8, 32))
    attention_mask = torch.ones(8, 32)
    labels = torch.randn(8, 768)
    
    model.train()
    for epoch in range(2):
        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
