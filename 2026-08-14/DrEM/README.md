# DrEM toy reproduction

This folder contains a toy but runnable PyTorch reproduction of **DrEM: Dual-Side Robust Ensemble Ranking from Noisy User Preference Predictions in Video Recommendation**.

## What is implemented

- synthetic user-item pair generation with multi-task pxtr predictions and heteroscedastic pxtr noise;
- a rank tower that fuses user, item, and pxtr features into pairwise scores;
- supervision-side robust weighting via estimated preference flip probability;
- feature-side perturbation consistency by sampling pxtr noise and enforcing stable ranking outputs;
- training and evaluation scripts with pair accuracy and toy GAUC.

## What is simplified

- upstream pxtr generation is simulated rather than trained by a separate multi-task network;
- the risk-correction term is implemented as a weighted pairwise BCE proxy instead of the paper's full industrial objective;
- serving-time system details and large-scale A/B platform hooks are omitted.

## Run

```bash
python train.py
python test.py --checkpoint drem_toy.pt
```
