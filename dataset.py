import os
import json
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

class TruckDriveDataset:
    def __init__(self, dataroot, scene):
        self.dataroot = dataroot
        self.scene = scene
        self.scene_path = os.path.join(dataroot, scene)
        
        # Load calibration
        with open(os.path.join(self.scene_path, 'calib_tf_tree_full.json'), 'r') as f:
            self.calib = json.load(f)
            
        # Extract velodyne to cab transform (cab is parent)
        velo_cal = self.calib.get('cab_velodyne')
        if velo_cal:
            t = velo_cal['transform']['translation']
            r = velo_cal['transform']['rotation']
            self.T_cab_velodyne = self._make_transform(t, r)
            self.T_velodyne_cab = np.linalg.inv(self.T_cab_velodyne)
        else:
            self.T_velodyne_cab = np.eye(4)
            
        # Ouster sensors
        self.ouster_sensors = ['forward_center', 'sideward_left', 'sideward_right']
        self.ouster_transforms = {}
        for s in self.ouster_sensors:
            cal = self.calib.get(f'cab_lidar_ouster_{s}')
            if cal:
                t = cal['transform']['translation']
                r = cal['transform']['rotation']
                T_cab_sensor = self._make_transform(t, r)
                self.ouster_transforms[s] = self.T_velodyne_cab @ T_cab_sensor
            else:
                self.ouster_transforms[s] = np.eye(4)
                
        # Find sync IDs from bounding boxes
        self.frames = []
        bboxes_dir = os.path.join(self.scene_path, 'bounding_boxes')
        for f in sorted(glob.glob(os.path.join(bboxes_dir, '*.json'))):
            basename = os.path.basename(f)
            sync_id = basename.split('_')[0]
            self.frames.append({
                'sync_id': sync_id,
                'bbox_path': f
            })
            
    def _make_transform(self, t, r):
        mat = np.eye(4)
        mat[:3, 3] = [t['x'], t['y'], t['z']]
        mat[:3, :3] = R.from_quat([r['x'], r['y'], r['z'], r['w']]).as_matrix()
        return mat
        
    def _transform_points(self, points, T):
        ones = np.ones((points.shape[0], 1))
        homo_points = np.hstack([points, ones])
        transformed = (T @ homo_points.T).T
        return transformed[:, :3]
        
    def _transform_vectors(self, vectors, T):
        rot = T[:3, :3]
        return (rot @ vectors.T).T

    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        frame = self.frames[idx]
        sync_id = frame['sync_id']
        
        data = {
            'sync_id': sync_id,
            'boxes': [],
            'aeva': None,
            'conti542': None,
            'ouster': []
        }
        
        # Load bounding boxes
        with open(frame['bbox_path'], 'r') as f:
            boxes = json.load(f)
            for b in boxes:
                if b.get('x', -1000) != -1000: # filter invalid
                    data['boxes'].append(b)
                    
        # Load Ouster
        ouster_points_list = []
        for s in self.ouster_sensors:
            paths = glob.glob(os.path.join(self.scene_path, 'ouster', s, 'points', f"{sync_id}_*.bin"))
            if paths:
                pts = np.fromfile(paths[0], dtype=np.float32).reshape(-1, 7)
                xyz = pts[:, :3]
                xyz_trans = self._transform_points(xyz, self.ouster_transforms[s])
                pts[:, :3] = xyz_trans
                ouster_points_list.append(pts)
        if ouster_points_list:
            data['ouster'] = np.vstack(ouster_points_list)
            
        # Load Aeva
        aeva_paths = glob.glob(os.path.join(self.scene_path, 'aeva', 'joint_lidars', 'points', f"{sync_id}_*.bin"))
        if aeva_paths:
            pts = np.fromfile(aeva_paths[0], dtype=np.float64).reshape(-1, 11)
            xyz = pts[:, :3]
            xyz_trans = self._transform_points(xyz, self.T_velodyne_cab)
            pts[:, :3] = xyz_trans
            
            # transform velocity
            vel = pts[:, 8:11]
            vel_trans = self._transform_vectors(vel, self.T_velodyne_cab)
            pts[:, 8:11] = vel_trans
            data['aeva'] = pts
            
        # Load Conti542
        conti_paths = glob.glob(os.path.join(self.scene_path, 'conti542', 'joint_radars', 'detections', f"{sync_id}_*.bin"))
        if conti_paths:
            pts = np.fromfile(conti_paths[0], dtype=np.float64).reshape(-1, 33)
            xyz = pts[:, :3]
            xyz_trans = self._transform_points(xyz, self.T_velodyne_cab)
            pts[:, :3] = xyz_trans
            
            # transform velocity
            vel = pts[:, 27:30]
            vel_trans = self._transform_vectors(vel, self.T_velodyne_cab)
            pts[:, 27:30] = vel_trans
            data['conti542'] = pts
            
        return data

if __name__ == '__main__':
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    print(f"Loaded {len(dataset)} frames.")
    frame_data = dataset[0]
    print("Boxes:", len(frame_data['boxes']))
    if frame_data['aeva'] is not None:
        print("Aeva pts:", frame_data['aeva'].shape)
    if frame_data['conti542'] is not None:
        print("Conti pts:", frame_data['conti542'].shape)
    if len(frame_data['ouster']) > 0:
        print("Ouster pts:", frame_data['ouster'].shape)
