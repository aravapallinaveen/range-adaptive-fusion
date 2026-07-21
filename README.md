# RadarLiDAR-LongRange: Range-Adaptive Sensor Fusion for Highway 3D Detection

## Project Overview

Camera and short-range LiDAR perception often collapse past ~150m. Highway trucking needs reliable perception hundreds of meters out due to longer stopping distances. Most fusion systems trust all sensors equally regardless of range, even though only long-range FMCW LiDAR and 4D radar maintain efficacy at distance.

This project implements a fusion detector that shifts trust toward the sensors that still work as range increases. It uses native Doppler/velocity signals (from FMCW LiDAR and 4D Radar) as a trust mechanism, demonstrating improved accuracy in the 150-400m band compared to standard fixed-weight baselines.

## Findings & Evaluation

As we move from the short-range band (0-100m) into the long-range bands (100-400m), the point density of traditional LiDAR decays rapidly. Our evaluation on the Torc Robotics TruckDrive dataset demonstrates that by adaptively shifting fusion weights toward FMCW LiDAR (Aeva) and 4D Radar (Conti542) at further distances, we are able to maintain a higher recall and precision for moving targets. 

*(Note: The absolute AP values are low due to the use of classical DBSCAN geometric clustering instead of deep-learning detectors, but the relative robustness at long ranges is maintained).*

![Average Precision vs Range Bucket](ap_vs_range.png)

### Fusion Demo

Below is a side-by-side comparison. The left panel shows the baseline fixed-weight fusion, which often drops detections at long distances. The right panel demonstrates the range-adaptive fusion preserving tracking by relying on the radar and FMCW LiDAR returns.

![Fusion Demo Comparison](demo.gif)

## Repo Structure

- `data/`: Downloaded scenes from the dataset.
- `detectors/`: Per-sensor classical clustering detectors (`aeva.py`, `conti542.py`, `ouster.py`).
- `fusion/`: Fusion implementations (`baseline.py` vs. `adaptive.py`).
- `eval/`: Evaluation metrics (`ap_by_range.py`), chart generation, and demo scripts (`demo_video.py`, `make_gif.py`).
- `configs/`: Thresholds and weights for range buckets (`fusion_config.py`).

## Interview Talking Points

- **The Problem:** Modern autonomous trucking requires >200m perception to safely brake, but camera/LiDAR fusion usually collapses at ~150m because most pipelines use fixed fusion weights.
- **The Approach:** Implemented a range-adaptive late-fusion pipeline that dynamically shifts sensor trust based on the target's distance. At >200m, it relies heavily on 4D Radar and FMCW LiDAR velocity returns rather than geometry.
- **The Tools:** Leveraged classical clustering (DBSCAN via Open3D) as an interpretable proxy for bounding box proposals, and built a custom evaluation script to calculate AP across range buckets.
- **The Result:** The adaptive pipeline preserves target tracking through the 150-400m blindspot where the static-weight baseline fails.

## Attribution & License

> "TruckDrive, provided by Torc Robotics, Inc., available at torc-ai.github.io/TruckDrive, used under the Torc Robotics Non-Commercial License v1.0."
