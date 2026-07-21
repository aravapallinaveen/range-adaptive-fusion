import numpy as np
import open3d as o3d
from detectors.utils import Box3D, fit_box_2d

def detect_ouster(points, eps=1.0, min_points=10, voxel_size=0.5):
    if points is None or len(points) == 0:
        return []
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    ds_points = np.asarray(pcd.points)
    if len(ds_points) == 0:
        return []
        
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    boxes = []
    max_label = labels.max()
    for i in range(max_label + 1):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) < min_points:
            continue
            
        cluster_pts = ds_points[cluster_idx]
        x, y, z, l, w, h, yaw = fit_box_2d(cluster_pts)
        
        # Filter implausible sizes (e.g. too big > 30m, too small < 1m area)
        if l > 30 or w > 10 or (l * w) < 1.0:
            continue
            
        boxes.append(Box3D(
            x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw,
            confidence=float(len(cluster_idx)), 
            sensor_source='ouster'
        ).to_dict())
        
    return boxes
