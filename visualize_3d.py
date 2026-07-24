"""
Generate publication-quality BEV (Bird's Eye View) visualizations
of the TruckDrive sensor suite with lane markings and bounding boxes.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from dataset import TruckDriveDataset


def draw_box_bev(ax, x, y, l, w, yaw, color='#FF1493', lw=1.8, label=None):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l/2, w/2],[l/2, -w/2],[-l/2, -w/2],[-l/2, w/2]])
    rotated = (R @ corners.T).T + np.array([x, y])
    poly = patches.Polygon(rotated, closed=True, fill=False,
                           edgecolor=color, linewidth=lw, zorder=10, label=label)
    ax.add_patch(poly)
    # heading arrow
    head = R @ np.array([l/2, 0]) + np.array([x, y])
    ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=lw, zorder=10)


def load_lane_lines(scene_path, sync_id):
    """Load lane lines from JSON for the given frame."""
    pattern = os.path.join(scene_path, 'lane_lines', f'{sync_id}_*.json')
    paths = glob.glob(pattern)
    if not paths:
        return [], []
    with open(paths[0]) as f:
        data = json.load(f)
    lines = [x for x in data if x['obj_class'] == 'lane_line']
    segments = [x for x in data if x['obj_class'] == 'lane_segment']
    return lines, segments


def generate_hero_bev(dataset, frame_idx=0):
    """Generate a stunning BEV showing all sensors, lanes, and boxes."""
    data = dataset[frame_idx]
    scene_path = dataset.scene_path

    # ── dark theme ──
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # ── 1. Ouster points colored by intensity ──
    if len(data['ouster']) > 0:
        o = data['ouster']
        intensity = o[:, 3]
        intensity_norm = np.clip(intensity / np.percentile(intensity, 99), 0, 1)
        ax.scatter(o[:, 0], o[:, 1], c=intensity_norm, cmap='Blues',
                   s=0.15, alpha=0.45, zorder=2, rasterized=True)

    # ── 2. Aeva points colored by radial velocity ──
    if data['aeva'] is not None:
        a = data['aeva']
        vr = np.sqrt(a[:, 8]**2 + a[:, 9]**2 + a[:, 10]**2)
        vr_norm = np.clip(vr / max(np.percentile(vr, 99), 0.01), 0, 1)
        ax.scatter(a[:, 0], a[:, 1], c=vr_norm, cmap='cool',
                   s=0.08, alpha=0.5, zorder=3, rasterized=True)

    # ── 3. Conti542 radar points colored by radial velocity ──
    if data['conti542'] is not None:
        c = data['conti542']
        vr_c = np.sqrt(c[:, 27]**2 + c[:, 28]**2 + c[:, 29]**2)
        vr_c_norm = np.clip(vr_c / max(np.percentile(vr_c, 99), 0.01), 0, 1)
        ax.scatter(c[:, 0], c[:, 1], c=vr_c_norm, cmap='autumn',
                   s=6.0, alpha=0.9, zorder=4, marker='D',
                   edgecolors='none', rasterized=True)

    # ── 4. Lane lines ──
    lines, _ = load_lane_lines(scene_path, data['sync_id'])
    for ll in lines:
        pts = np.array(ll['points'])
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color='#444444', linewidth=1.0,
                    linestyle='--', alpha=0.7, zorder=1)

    # ── 5. Bounding boxes ──
    first_box = True
    for b in data['boxes']:
        lbl = 'Ground-Truth Box' if first_box else None
        draw_box_bev(ax, b['x'], b['y'], b['l'], b['w'], b['yaw'],
                     color='#FF1493', lw=2.0, label=lbl)
        first_box = False

    # ── 6. Ego vehicle marker ──
    ego = patches.FancyBboxPatch((-2, -1), 4, 2, boxstyle="round,pad=0.3",
                                  facecolor='#222222', edgecolor='white',
                                  linewidth=1.5, zorder=20)
    ax.add_patch(ego)
    ax.text(0, 0, 'EGO', ha='center', va='center', color='white',
            fontsize=7, fontweight='bold', zorder=21)

    # ── 7. Range rings ──
    for r in [50, 100, 150, 200, 300]:
        circle = plt.Circle((0, 0), r, fill=False, edgecolor='#333333',
                             linewidth=0.5, linestyle=':', zorder=0)
        ax.add_patch(circle)
        ax.text(r * 0.71, r * 0.71, f'{r}m', color='#555555',
                fontsize=7, ha='center', zorder=0)

    # ── styling ──
    ax.set_xlim(-30, 250)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('X (forward) [m]', color='#888888', fontsize=10)
    ax.set_ylabel('Y (lateral) [m]', color='#888888', fontsize=10)
    ax.tick_params(colors='#555555', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    # ── legend ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4169E1',
               markersize=5, linestyle='None', label='Ouster (intensity-colored)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#00CED1',
               markersize=5, linestyle='None', label='Aeva FMCW LiDAR (velocity-colored)'),
        Line2D([0],[0], marker='D', color='w', markerfacecolor='#FF6347',
               markersize=5, linestyle='None', label='Conti542 4D Radar (velocity-colored)'),
        Line2D([0],[0], color='#FF1493', linewidth=2, label='Ground-Truth Boxes'),
        Line2D([0],[0], color='#444444', linewidth=1, linestyle='--', label='Lane Lines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              facecolor='#111111', edgecolor='#333333', labelcolor='#cccccc',
              framealpha=0.9)

    ax.set_title(f'Bird\'s Eye View — Multi-Sensor Fusion (Frame {data["sync_id"]})',
                 color='white', fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig('sensor_bev.png', dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close()
    print('Saved sensor_bev.png')


def generate_sensor_comparison(dataset, frame_idx=0):
    """Generate a 3-panel comparison showing each sensor's contribution."""
    data = dataset[frame_idx]
    scene_path = dataset.scene_path

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor='#0a0a0a')

    sensors = [
        ('Ouster (Short-Range LiDAR)', '#4169E1', 'ouster'),
        ('Aeva (FMCW LiDAR — Long Range)', '#00CED1', 'aeva'),
        ('Conti542 (4D Radar — Long Range)', '#FF6347', 'conti542'),
    ]

    for ax, (title, color, key) in zip(axes, sensors):
        ax.set_facecolor('#0a0a0a')

        # Range rings
        for r in [50, 100, 200, 300]:
            circle = plt.Circle((0, 0), r, fill=False, edgecolor='#222222',
                                linewidth=0.4, linestyle=':', zorder=0)
            ax.add_patch(circle)

        # Lane lines
        lines, _ = load_lane_lines(scene_path, data['sync_id'])
        for ll in lines:
            pts = np.array(ll['points'])
            if len(pts) >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color='#333333', linewidth=0.8,
                        linestyle='--', alpha=0.5, zorder=1)

        # Points
        if key == 'ouster' and len(data['ouster']) > 0:
            o = data['ouster']
            intensity = np.clip(o[:, 3] / np.percentile(o[:, 3], 99), 0, 1)
            ax.scatter(o[:, 0], o[:, 1], c=intensity, cmap='Blues',
                       s=0.1, alpha=0.5, zorder=2, rasterized=True)
        elif key == 'aeva' and data['aeva'] is not None:
            a = data['aeva']
            vr = np.sqrt(a[:, 8]**2 + a[:, 9]**2 + a[:, 10]**2)
            vr_norm = np.clip(vr / max(np.percentile(vr, 99), 0.01), 0, 1)
            ax.scatter(a[:, 0], a[:, 1], c=vr_norm, cmap='cool',
                       s=0.06, alpha=0.5, zorder=2, rasterized=True)
        elif key == 'conti542' and data['conti542'] is not None:
            c = data['conti542']
            vr = np.sqrt(c[:, 27]**2 + c[:, 28]**2 + c[:, 29]**2)
            vr_norm = np.clip(vr / max(np.percentile(vr, 99), 0.01), 0, 1)
            ax.scatter(c[:, 0], c[:, 1], c=vr_norm, cmap='autumn',
                       s=4.0, alpha=0.8, zorder=2, marker='D',
                       edgecolors='none', rasterized=True)

        # Boxes
        for b in data['boxes']:
            draw_box_bev(ax, b['x'], b['y'], b['l'], b['w'], b['yaw'],
                         color='#FF1493', lw=1.5)

        # Ego
        ego = patches.FancyBboxPatch((-1.5, -0.75), 3, 1.5,
                                      boxstyle="round,pad=0.2",
                                      facecolor='#222222', edgecolor='white',
                                      linewidth=1, zorder=20)
        ax.add_patch(ego)

        ax.set_xlim(-20, 200)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.set_title(title, color=color, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors='#444444', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#222222')

    fig.suptitle('Per-Sensor Point Cloud Coverage',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('sensor_comparison.png', dpi=300, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    plt.close()
    print('Saved sensor_comparison.png')


if __name__ == '__main__':
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        print('No frames found.')
        exit(1)
    generate_hero_bev(dataset, frame_idx=0)
    generate_sensor_comparison(dataset, frame_idx=0)
