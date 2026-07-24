import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from dataset import TruckDriveDataset

def draw_box_3d(ax, x, y, z, l, w, h, yaw, color='red'):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    
    # 2D corners relative to center
    corners_2d = np.array([
        [l/2, w/2],
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])
    
    # Rotate and translate
    corners_2d = (R @ corners_2d.T).T
    corners_2d[:, 0] += x
    corners_2d[:, 1] += y
    
    # 3D corners (bottom and top)
    z_bottom = z - h/2
    z_top = z + h/2
    
    for i in range(4):
        p1 = corners_2d[i]
        p2 = corners_2d[(i+1)%4]
        
        # Bottom edges
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z_bottom, z_bottom], color=color, linewidth=2)
        # Top edges
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z_top, z_top], color=color, linewidth=2)
        # Vertical edges
        ax.plot([p1[0], p1[0]], [p1[1], p1[1]], [z_bottom, z_top], color=color, linewidth=2)

def main():
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        return
        
    data = dataset[0] # First frame
    
    # Create dark-themed 3D plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#111111')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111111')
    
    # Plot Ouster (near field)
    if len(data['ouster']) > 0:
        o_pts = data['ouster']
        ax.scatter(o_pts[:, 0], o_pts[:, 1], o_pts[:, 2], c='#4169E1', s=0.1, alpha=0.3, label='Ouster (Short LiDAR)')
        
    # Plot Aeva
    if data['aeva'] is not None:
        a_pts = data['aeva']
        ax.scatter(a_pts[:, 0], a_pts[:, 1], a_pts[:, 2], c='#00FFFF', s=0.2, alpha=0.6, label='Aeva (FMCW LiDAR)')
        
    # Plot Conti542
    if data['conti542'] is not None:
        c_pts = data['conti542']
        ax.scatter(c_pts[:, 0], c_pts[:, 1], c_pts[:, 2], c='#FF4500', s=5.0, alpha=1.0, label='Conti542 (4D Radar)')
        
    # Plot boxes
    for b in data['boxes']:
        draw_box_3d(ax, b['x'], b['y'], b['z'], b['l'], b['w'], b['h'], b['yaw'], color='#FF1493')
        
    # Set view limits and angle
    ax.set_xlim(0, 150)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-5, 10)
    
    ax.view_init(elev=25, azim=-115)
    
    # Remove grid and panes for cleaner look
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    plt.legend(loc='upper right', frameon=True, facecolor='black', edgecolor='white')
    
    out_file = "demo_3d.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved stunning 3D visualization to {out_file}")

if __name__ == '__main__':
    main()
