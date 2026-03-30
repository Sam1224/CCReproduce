# Geometry-aware Spatial Reasoning (toy reproduction)

Toy reproduction skeleton for:

- **Make Geometry Matter for Spatial Reasoning** (arXiv:2603.26639)

## Idea captured

We build a toy spatial reasoning task on 2D points:

- Input: coordinates of points A/B/C.
- Question: which point is closest to the origin? / is A inside a circle? etc.
- Compare a baseline MLP vs a geometry-aware model that explicitly uses distances.

## Quickstart

```bash
cd ccreproduce_repo/2026-03-27/geometry_spatial_reasoning
python3 -m pip install -r requirements.txt
python3 train.py --epochs 15
python3 test.py
```
