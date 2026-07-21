import numpy as np
import open3d as o3d
from detectors.utils import Box3D

def detect_conti542(points, rcs_threshold=-10.0, eps=3.0, min_points=1):
    # Conti542 schema has 33 columns.
    # indices: 0:x, 1:y, 2:z, 3:rangerate, 4:rcs, 5:amplitude
    # 27:vx, 28:vy, 29:vz
    if points is None or len(points) == 0:
        return []
        
    # filter by rcs
    valid_mask = points[:, 4] > rcs_threshold
    valid_points = points[valid_mask]
    
    if len(valid_points) == 0:
        return []
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    boxes = []
    max_label = labels.max()
    
    for i in range(-1, max_label + 1):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) == 0:
            continue
            
        cluster_pts = valid_points[cluster_idx]
        
        # Center of cluster
        center = np.mean(cluster_pts[:, :3], axis=0)
        x, y, z = center[0], center[1], center[2]
        
        # Fixed box size heuristic for vehicles
        l, w, h = 4.5, 2.0, 2.0
        yaw = 0.0
        
        mean_vx = np.mean(cluster_pts[:, 27])
        mean_vy = np.mean(cluster_pts[:, 28])
        mean_vz = np.mean(cluster_pts[:, 29])
        
        mean_rcs = np.mean(cluster_pts[:, 4])
        
        boxes.append(Box3D(
            x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw,
            vx=mean_vx, vy=mean_vy, vz=mean_vz,
            confidence=mean_rcs + 50.0,
            sensor_source='conti542'
        ).to_dict())
        
    return boxes
