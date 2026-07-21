import os
from dataclasses import dataclass
import numpy as np

@dataclass
class Box3D:
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    confidence: float = 1.0
    sensor_source: str = "unknown"

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'z': self.z,
            'l': self.l, 'w': self.w, 'h': self.h,
            'yaw': self.yaw, 'vx': self.vx, 'vy': self.vy, 'vz': self.vz,
            'confidence': self.confidence, 'sensor_source': self.sensor_source
        }

def fit_box_2d(points):
    """Fit a simple AABB bounding box to points."""
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    
    center = (min_pt + max_pt) / 2
    l = max_pt[0] - min_pt[0]
    w = max_pt[1] - min_pt[1]
    h = max_pt[2] - min_pt[2]
    
    # ensure minimum dimensions
    l = max(l, 0.5)
    w = max(w, 0.5)
    h = max(h, 0.5)
    
    return center[0], center[1], center[2], l, w, h, 0.0
