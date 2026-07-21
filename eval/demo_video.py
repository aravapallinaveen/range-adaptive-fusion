import os
import matplotlib.pyplot as plt
from dataset import TruckDriveDataset
from detectors.ouster import detect_ouster
from detectors.aeva import detect_aeva
from detectors.conti542 import detect_conti542
from fusion.baseline import baseline_fusion
from fusion.adaptive import adaptive_fusion
import matplotlib.patches as patches
import numpy as np

def draw_box(ax, x, y, l, w, yaw, color='r'):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([
        [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
    ])
    rotated = (R @ corners.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    poly = patches.Polygon(rotated, closed=True, fill=False, edgecolor=color, linewidth=2.0)
    ax.add_patch(poly)

def render_frame(data, base_preds, adapt_preds, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlim(-50, 400)
        ax.set_ylim(-100, 100)
        
        if data['aeva'] is not None:
            pts = data['aeva']
            idx = np.random.choice(len(pts), min(10000, len(pts)), replace=False)
            ax.scatter(pts[idx, 0], pts[idx, 1], c='lightgray', s=0.5, alpha=0.5)
            
    ax1.set_title("Fixed-Weight Baseline (Phase 3)")
    ax2.set_title("Range-Adaptive Fusion (Phase 4)")
    
    for b in base_preds:
        draw_box(ax1, b['x'], b['y'], b['l'], b['w'], b['yaw'], color='red')
        
    for b in adapt_preds:
        color = 'gray'
        if 'ouster' in b['sensor_source']: color = 'blue'
        if 'aeva' in b['sensor_source']: color = 'teal'
        if 'conti542' in b['sensor_source']: color = 'coral'
        
        if 'fused' in b['sensor_source']:
            d = np.sqrt(b['x']**2 + b['y']**2)
            if d < 100: color = 'blue'
            elif d < 200: color = 'teal'
            else: color = 'coral'
            
        draw_box(ax2, b['x'], b['y'], b['l'], b['w'], b['yaw'], color=color)
        
    for gt in data['boxes']:
        draw_box(ax1, gt['x'], gt['y'], gt['l'], gt['w'], gt['yaw'], color='black')
        draw_box(ax2, gt['x'], gt['y'], gt['l'], gt['w'], gt['yaw'], color='black')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()

def main():
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        return
        
    out_dir = 'demo_frames'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    num_frames = min(10, len(dataset))
    for i in range(num_frames):
        frame = dataset[i]
        
        o_boxes = detect_ouster(frame['ouster']) if len(frame['ouster']) > 0 else []
        a_boxes = detect_aeva(frame['aeva']) if frame['aeva'] is not None else []
        c_boxes = detect_conti542(frame['conti542']) if frame['conti542'] is not None else []
        
        base_preds = baseline_fusion(o_boxes, a_boxes, c_boxes)
        adapt_preds = adaptive_fusion(o_boxes, a_boxes, c_boxes)
        
        out_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        render_frame(frame, base_preds, adapt_preds, out_path)
        print(f"Rendered {out_path}")
        
if __name__ == '__main__':
    main()
