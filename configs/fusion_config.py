# configs/fusion_config.py

# Distance threshold (meters) for associating boxes from different sensors
MATCHING_DIST_THRESH = 4.0

# Fixed-weight Baseline Fusion Weights
BASELINE_WEIGHTS = {
    'ouster': 0.5,
    'aeva': 0.3,
    'conti542': 0.2
}

# Range-Adaptive Fusion Configuration
# Buckets define the start range, end range, and the sensor weights within that bucket
ADAPTIVE_BUCKETS = [
    {
        'range': (0, 100),
        'weights': {'ouster': 0.7, 'aeva': 0.2, 'conti542': 0.1}
    },
    {
        'range': (100, 200),
        'weights': {'ouster': 0.0, 'aeva': 0.7, 'conti542': 0.3}
    },
    {
        'range': (200, 400),
        'weights': {'ouster': 0.0, 'aeva': 0.2, 'conti542': 0.8}
    }
]

# Unmatched boxes are kept if their confidence is above these thresholds
CONFIDENCE_THRESHOLDS = {
    'ouster': 20.0,    # min points
    'aeva': 40.0,      # min points + velocity boost
    'conti542': 55.0   # min RCS score
}
