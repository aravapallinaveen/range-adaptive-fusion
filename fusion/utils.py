import numpy as np

def compute_distance(box1, box2):
    return np.sqrt((box1['x'] - box2['x'])**2 + (box1['y'] - box2['y'])**2)

def match_boxes(boxes1, boxes2, dist_thresh=2.0):
    """
    Greedy matching between two lists of boxes based on center distance.
    """
    if not boxes1:
        return [], [], boxes2.copy()
    if not boxes2:
        return [], boxes1.copy(), []
        
    dist_matrix = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            dist_matrix[i, j] = compute_distance(b1, b2)
            
    matched_pairs = []
    matched_i = set()
    matched_j = set()
    
    while True:
        min_dist = np.inf
        best_i, best_j = -1, -1
        
        for i in range(len(boxes1)):
            if i in matched_i: continue
            for j in range(len(boxes2)):
                if j in matched_j: continue
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    best_i, best_j = i, j
                    
        if min_dist > dist_thresh or best_i == -1:
            break
            
        matched_pairs.append((boxes1[best_i], boxes2[best_j]))
        matched_i.add(best_i)
        matched_j.add(best_j)
        
    unmatched_1 = [b for i, b in enumerate(boxes1) if i not in matched_i]
    unmatched_2 = [b for j, b in enumerate(boxes2) if j not in matched_j]
    
    return matched_pairs, unmatched_1, unmatched_2

def fuse_matched_boxes(box_list, weights_dict):
    """
    Fuses a list of matched boxes into a single box using weighted average.
    Scales the confidence of the final box by the sensor trust weights.
    """
    if not box_list:
        return None
        
    total_weight = sum(weights_dict.get(b['sensor_source'], 0.0) for b in box_list)
    
    # If total weight is 0 (e.g. sensor not trusted at this range), return None to drop it
    if total_weight == 0:
        return None
        
    fused_box = {
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'l': 0.0, 'w': 0.0, 'h': 0.0,
        'yaw': 0.0, 'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'confidence': 0.0,
        'sensor_source': 'fused_' + '_'.join(sorted([b['sensor_source'] for b in box_list]))
    }
    
    for b in box_list:
        raw_w = weights_dict.get(b['sensor_source'], 0.0)
        norm_w = raw_w / total_weight
        
        fused_box['x'] += b['x'] * norm_w
        fused_box['y'] += b['y'] * norm_w
        fused_box['z'] += b['z'] * norm_w
        fused_box['l'] += b['l'] * norm_w
        fused_box['w'] += b['w'] * norm_w
        fused_box['h'] += b['h'] * norm_w
        fused_box['yaw'] += b['yaw'] * norm_w 
        
        fused_box['vx'] += b['vx'] * norm_w
        fused_box['vy'] += b['vy'] * norm_w
        fused_box['vz'] += b['vz'] * norm_w
        
        # Scale confidence by raw weight to penalize untrusted sensors
        fused_box['confidence'] += b['confidence'] * raw_w
        
    return fused_box
