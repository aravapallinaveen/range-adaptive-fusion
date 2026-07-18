import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from dataset import TruckDriveDataset

def draw_box(ax, x, y, l, w, yaw, color='r'):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    
    corners = np.array([
        [l/2, w/2],
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])
    
    rotated = (R @ corners.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    
    poly = patches.Polygon(rotated, closed=True, fill=False, edgecolor=color, linewidth=1.5)
    ax.add_patch(poly)

def visualize_frame(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Plot Ouster (near field)
    if len(data['ouster']) > 0:
        o_pts = data['ouster']
        ax.scatter(o_pts[:, 0], o_pts[:, 1], c='blue', s=0.5, alpha=0.5, label='Ouster')
        
    # Plot Aeva
    if data['aeva'] is not None:
        a_pts = data['aeva']
        ax.scatter(a_pts[:, 0], a_pts[:, 1], c='teal', s=0.5, alpha=0.5, label='Aeva')
        
    # Plot Conti542
    if data['conti542'] is not None:
        c_pts = data['conti542']
        ax.scatter(c_pts[:, 0], c_pts[:, 1], c='coral', s=2.0, alpha=0.8, label='Conti542')
        
    # Plot boxes
    for b in data['boxes']:
        # Bounding box is in velodyne coordinates
        draw_box(ax, b['x'], b['y'], b['l'], b['w'], b['yaw'], color='red')
        
    ax.set_xlim(-50, 200)
    ax.set_ylim(-50, 50)
    ax.set_title(f"Frame {data['sync_id']}")
    ax.legend(loc='upper right')
    
    out_file = f"viz_{data['sync_id']}.png"
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved visualization to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='data/TruckDrive')
    parser.add_argument('--scene', type=str, default='scene_28_1')
    args = parser.parse_args()
    
    dataset = TruckDriveDataset(args.dataroot, args.scene)
    print(f"Loaded dataset with {len(dataset)} frames")
    
    # Visualize first 3 frames
    for i in range(min(3, len(dataset))):
        visualize_frame(dataset[i])
