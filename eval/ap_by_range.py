import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import TruckDriveDataset
from detectors.ouster import detect_ouster
from detectors.aeva import detect_aeva
from detectors.conti542 import detect_conti542
from fusion.baseline import baseline_fusion
from fusion.adaptive import adaptive_fusion

def dist(b1, b2):
    return np.sqrt((b1['x'] - b2['x'])**2 + (b1['y'] - b2['y'])**2)

def get_bucket(d):
    if d < 100: return '0-100m'
    if d < 200: return '100-200m'
    return '200-400m'

def evaluate_frame(preds, gt_boxes, match_thresh=4.0, conf_thresh=25.0):
    metrics = {
        '0-100m': {'tp': 0, 'fp': 0, 'gt': 0},
        '100-200m': {'tp': 0, 'fp': 0, 'gt': 0},
        '200-400m': {'tp': 0, 'fp': 0, 'gt': 0}
    }
    
    for gt in gt_boxes:
        d = np.sqrt(gt['x']**2 + gt['y']**2)
        bucket = get_bucket(d)
        metrics[bucket]['gt'] += 1
        
    matched_gt = set()
    for p in preds:
        if p['confidence'] < conf_thresh:
            continue
            
        d = np.sqrt(p['x']**2 + p['y']**2)
        bucket = get_bucket(d)
        
        best_gt = -1
        min_dist = float('inf')
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt: continue
            current_dist = dist(p, gt)
            if current_dist < min_dist:
                min_dist = current_dist
                best_gt = i
                
        if min_dist < match_thresh:
            metrics[bucket]['tp'] += 1
            matched_gt.add(best_gt)
        else:
            metrics[bucket]['fp'] += 1
            
    return metrics

def calculate_ap(metrics_dict):
    ap_by_bucket = {}
    for b in metrics_dict:
        m = metrics_dict[b]
        if m['gt'] == 0:
            ap_by_bucket[b] = 0.0
        else:
            recall = m['tp'] / m['gt']
            precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0.0
            ap_by_bucket[b] = recall * precision
    return ap_by_bucket

def main():
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        return
        
    num_frames = min(10, len(dataset))
    print(f"Evaluating {num_frames} frames...")
    
    baseline_metrics = {b: {'tp': 0, 'fp': 0, 'gt': 0} for b in ['0-100m', '100-200m', '200-400m']}
    adaptive_metrics = {b: {'tp': 0, 'fp': 0, 'gt': 0} for b in ['0-100m', '100-200m', '200-400m']}
    
    for i in range(num_frames):
        frame = dataset[i]
        gt_boxes = frame['boxes']
        
        o_boxes = detect_ouster(frame['ouster']) if len(frame['ouster']) > 0 else []
        a_boxes = detect_aeva(frame['aeva']) if frame['aeva'] is not None else []
        c_boxes = detect_conti542(frame['conti542']) if frame['conti542'] is not None else []
        
        base_preds = baseline_fusion(o_boxes, a_boxes, c_boxes)
        adapt_preds = adaptive_fusion(o_boxes, a_boxes, c_boxes)
        
        base_m = evaluate_frame(base_preds, gt_boxes)
        adapt_m = evaluate_frame(adapt_preds, gt_boxes)
        
        for b in baseline_metrics:
            baseline_metrics[b]['tp'] += base_m[b]['tp']
            baseline_metrics[b]['fp'] += base_m[b]['fp']
            baseline_metrics[b]['gt'] += base_m[b]['gt']
            
            adaptive_metrics[b]['tp'] += adapt_m[b]['tp']
            adaptive_metrics[b]['fp'] += adapt_m[b]['fp']
            adaptive_metrics[b]['gt'] += adapt_m[b]['gt']
            
    base_ap = calculate_ap(baseline_metrics)
    adapt_ap = calculate_ap(adaptive_metrics)
    
    print("Baseline AP:", base_ap)
    print("Adaptive AP:", adapt_ap)
    
    buckets = ['0-100m', '100-200m', '200-400m']
    base_vals = [base_ap[b] for b in buckets]
    adapt_vals = [adapt_ap[b] for b in buckets]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, base_vals, width, label='Fixed-Weight Baseline', color='gray')
    ax.bar(x + width/2, adapt_vals, width, label='Range-Adaptive Fusion', color='teal')
    
    ax.set_ylabel('Approx. Average Precision (Prec * Recall)')
    ax.set_title('Fusion AP vs. Range Bucket')
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.legend()
    ax.set_ylim([0, 1.0])
    
    if not os.path.exists('eval'):
        os.makedirs('eval')
    plt.savefig('ap_vs_range.png', dpi=200)
    print("Saved chart to ap_vs_range.png")
    
if __name__ == '__main__':
    main()
