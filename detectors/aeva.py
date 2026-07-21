import numpy as np
import open3d as o3d
from detectors.utils import Box3D, fit_box_2d

def detect_aeva(points, eps=1.5, min_points=10, voxel_size=0.5):
    # Aeva points have 11 columns: x, y, z, intensity, velocity, reflectivity, time_offset_ns, sensor_id, vx, vy, vz
    if points is None or len(points) == 0:
        return []
        
    # To keep features during downsampling in Open3D, it's a bit tricky. 
    # We can just do a fast voxel downsampling manually, or just use larger eps/min_points without downsampling,
    # or just downsample xyz and then nearest neighbor to find velocity. 
    # A simpler way in python: random subsampling if it's too big.
    
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        points = points[idx]
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    boxes = []
    max_label = labels.max()
    for i in range(max_label + 1):
        cluster_idx = np.where(labels == i)[0]
        if len(cluster_idx) < min_points:
            continue
            
        cluster_pts = points[cluster_idx]
        x, y, z, l, w, h, yaw = fit_box_2d(cluster_pts[:, :3])
        
        if l > 30 or w > 10 or (l * w) < 1.0:
            continue
            
        # compute average velocity
        mean_vx = np.mean(cluster_pts[:, 8])
        mean_vy = np.mean(cluster_pts[:, 9])
        mean_vz = np.mean(cluster_pts[:, 10])
        
        # boost confidence if moving
        vel_mag = np.sqrt(mean_vx**2 + mean_vy**2)
        confidence = float(len(cluster_idx)) + vel_mag * 10.0
            
        boxes.append(Box3D(
            x=x, y=y, z=z, l=l, w=w, h=h, yaw=yaw,
            vx=mean_vx, vy=mean_vy, vz=mean_vz,
            confidence=confidence,
            sensor_source='aeva'
        ).to_dict())
        
    return boxes
