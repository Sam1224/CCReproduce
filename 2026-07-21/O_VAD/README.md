# O-VAD (toy reproduction)

This toy reproduction follows the paper's `ground -> track -> reason` idea with a simplified industrial video pipeline.

## Design
- `data.py`: synthesizes industrial process clips with multiple objects, stage-wise normal state evolution, and injected anomaly segments.
- `model.py`: tracks each object with a recurrent tracker, predicts object states over time, localizes anomalous frames, and reasons about anomaly type/object.
- `train.py`: jointly optimizes video anomaly detection, anomalous frame localization, state tracking, and anomaly object/type prediction.
- `test.py`: reports quantitative metrics and prints a structured anomaly report.

## Run
```bash
python3 train.py --epochs 5
python3 test.py --checkpoint checkpoints/o_vad.pt
```

## Notes
- The implementation is training-based for the toy setup, while the original paper is training-free.
- The module decomposition still mirrors the paper's emphasis on object-centric tracking and temporal reasoning.
