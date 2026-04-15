import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MOON(nn.Module):
    def __init__(self, vision_model_name='clip-vit-base-patch32', text_model_name='bert-base-uncased'):
        super().__init__()
        self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        # Generative MLLM representation head
        self.projection = nn.Linear(768 * 2, 768)
        self.generative_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=8), num_layers=6
        )

    def forward(self, images, input_ids, attention_mask):
        img_features = self.vision_encoder(images).pooler_output
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        fused_features = torch.cat([img_features, text_features], dim=-1)
        repr_features = self.projection(fused_features)
        
        # In a real setup, target sequence would be provided for generation
        tgt = torch.zeros_like(repr_features).unsqueeze(0)
        out = self.generative_decoder(tgt, repr_features.unsqueeze(0))
        return out.squeeze(0)
