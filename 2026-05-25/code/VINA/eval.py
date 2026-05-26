"""
VINA Evaluation Script

Evaluates the trained VINA detector on:
  1. Image-only benchmark (standard AIGC image detection)
  2. Video-frame benchmark (frames extracted from AI-generated videos)
  3. Cross-modal gap analysis (performance delta between image and video frames)

Usage:
    python eval.py --ckpt checkpoints/vina_best.pt
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import ToyAIGCDataset
from model import VINADetector


@torch.no_grad()
def evaluate(model, loader, device, modality='image'):
    """Evaluate detector on specified modality (image or video frame)."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    for images, video_frames, labels in loader:
        if modality == 'image':
            x = images.to(device)
        else:
            x = video_frames.to(device)
        labels_np = labels.numpy()

        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels_np.tolist())
        all_preds.extend(preds.tolist())

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    return {'AUC': auc, 'ACC': acc}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test dataset
    test_ds = ToyAIGCDataset(n_samples=1000, image_size=64, mode='val')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Load model
    model = VINADetector(embed_dim=128, proj_dim=64).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # Evaluate on images (standard AIGC image detection benchmark)
    img_metrics = evaluate(model, test_loader, device, modality='image')
    # Evaluate on video frames (cross-modal benchmark)
    vid_metrics = evaluate(model, test_loader, device, modality='video')

    print("\n" + "="*50)
    print("VINA Evaluation Results")
    print("="*50)
    print(f"Image detection:       AUC={img_metrics['AUC']:.4f}  ACC={img_metrics['ACC']:.4f}")
    print(f"Video frame detection: AUC={vid_metrics['AUC']:.4f}  ACC={vid_metrics['ACC']:.4f}")
    cross_modal_gap = img_metrics['AUC'] - vid_metrics['AUC']
    print(f"Cross-modal gap (AUC): {cross_modal_gap:+.4f}  (negative = video better than image)")
    print("="*50)
    print("\nNote: In the paper, VINA eliminates the cross-modal gap that baseline detectors")
    print("(trained on images only) exhibit when evaluated on video frames.")
    print("Baseline image-only detectors typically show AUC drop of 5-15% on video frames.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/vina_best.pt')
    args = parser.parse_args()
    main(args)
