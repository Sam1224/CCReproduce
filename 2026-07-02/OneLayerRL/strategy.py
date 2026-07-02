import torch
import torch.nn as nn

class LayerWiseRL:
    """
    Implements the finding that training a single transformer layer matches full-parameter RL.
    """
    def __init__(self, model):
        self.model = model

    def freeze_all_but_one(self, layer_index):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze specific layer (usually middle layers 40-60% depth)
        target_layer = self.model.layers[layer_index]
        for param in target_layer.parameters():
            param.requires_grad = True
        
        print(f"Layer {layer_index} unfrozen for RL optimization.")

def run_rl_training(model, optimizer, reward_fn):
    # Standard GRPO/PPO loop but only gradients for one layer
    pass

if __name__ == "__main__":
    # Example usage:
    # model = Qwen3ForCausalLM.from_pretrained(...)
    # strategy = LayerWiseRL(model)
    # strategy.freeze_all_but_one(layer_index=16) # Middle layer for 32-layer model
    print("Layer-wise RL training strategy initialized.")
