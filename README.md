# RadarLiDAR-LongRange: Range-Adaptive Sensor Fusion for Highway 3D Detection

## Project Overview

Camera and short-range LiDAR perception often collapse past ~150m. Highway trucking needs reliable perception hundreds of meters out due to longer stopping distances. Most fusion systems trust all sensors equally regardless of range, even though only long-range FMCW LiDAR and 4D radar maintain efficacy at distance.

This project implements a fusion detector that shifts trust toward the sensors that still work as range increases. It uses native Doppler/velocity signals (from FMCW LiDAR and 4D Radar) as a trust mechanism, demonstrating improved accuracy in the 150-400m band compared to standard fixed-weight baselines.

## Findings & Evaluation
*(This section will be populated in Phase 5 with the headline AP vs. Range chart and a summary of our findings).*

## Repo Structure

- `data/`: Downloaded scenes from the dataset.
- `detectors/`: Per-sensor classical clustering detectors (`aeva.py`, `conti542.py`, `ouster.py`).
- `fusion/`: Fusion implementations (baseline vs. adaptive). *(In Progress)*
- `eval/`: Evaluation metrics and chart generation. *(In Progress)*
- `configs/`: Thresholds and weights for range buckets. *(In Progress)*

## Interview Talking Points
*(This section will be populated in Phase 5).*

## Attribution & License

> "TruckDrive, provided by Torc Robotics, Inc., available at torc-ai.github.io/TruckDrive, used under the Torc Robotics Non-Commercial License v1.0."
