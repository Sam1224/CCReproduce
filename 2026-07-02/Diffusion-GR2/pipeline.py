import torch
from model import DiffusionGR2

def inference_pipeline():
    """
    Simulates the block-diffusion decoding process.
    """
    model = DiffusionGR2(vocab_size=1000, hidden_dim=256)
    model.eval()
    
    # Initial random noise
    x = torch.randint(0, 1000, (1, 60)) # reasoning + ranking span
    num_steps = 5 # small number of denoising steps
    
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            timesteps = torch.tensor([t / num_steps])
            logits_res, logits_rank = model(x[:, :50], timesteps)
            
            # Simplified denoising: pick most confident tokens
            pred_res = torch.argmax(logits_res, dim=-1)
            pred_rank = torch.argmax(logits_rank, dim=-1)
            
            x = torch.cat([pred_res, pred_rank], dim=-1)
            
    print("Final Ranking Permutation Sample:", x[0, -10:].tolist())

if __name__ == "__main__":
    inference_pipeline()
