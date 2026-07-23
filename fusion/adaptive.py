import numpy as np
from fusion.utils import match_boxes, fuse_matched_boxes
from configs.fusion_config import MATCHING_DIST_THRESH, ADAPTIVE_BUCKETS, CONFIDENCE_THRESHOLDS

def filter_unmatched(boxes, sensor_name):
    thresh = CONFIDENCE_THRESHOLDS.get(sensor_name, 0.0)
    return [b for b in boxes if b['confidence'] >= thresh]

def get_adaptive_weights(x, y):
    d = np.sqrt(x**2 + y**2)
    for bucket in ADAPTIVE_BUCKETS:
        r_min, r_max = bucket['range']
        if r_min <= d < r_max:
            return bucket['weights']
    return ADAPTIVE_BUCKETS[-1]['weights']

def adaptive_fusion(ouster_boxes, aeva_boxes, conti_boxes):
    final_fused_boxes = []
    
    matched_oa, un_o, un_a = match_boxes(ouster_boxes, aeva_boxes, MATCHING_DIST_THRESH)
    
    temp_groups = []
    for o_box, a_box in matched_oa:
        temp_groups.append([o_box, a_box])
        
    un_o_filtered = filter_unmatched(un_o, 'ouster')
    un_a_filtered = filter_unmatched(un_a, 'aeva')
    
    for b in un_o_filtered: temp_groups.append([b])
    for b in un_a_filtered: temp_groups.append([b])
    
    temp_oa_boxes = []
    for g in temp_groups:
        center_x = np.mean([b['x'] for b in g])
        center_y = np.mean([b['y'] for b in g])
        temp_oa_boxes.append({'x': center_x, 'y': center_y, 'group': g})
        
    matched_oac, un_oa_temp, un_c = match_boxes(temp_oa_boxes, conti_boxes, MATCHING_DIST_THRESH)
    
    final_groups = []
    for oa_temp, c_box in matched_oac:
        g = oa_temp['group'].copy()
        g.append(c_box)
        final_groups.append(g)
        
    for oa_temp in un_oa_temp:
        final_groups.append(oa_temp['group'])
        
    un_c_filtered = filter_unmatched(un_c, 'conti542')
    for b in un_c_filtered:
        final_groups.append([b])
        
    for g in final_groups:
        center_x = np.mean([b['x'] for b in g])
        center_y = np.mean([b['y'] for b in g])
        
        weights = get_adaptive_weights(center_x, center_y)
        fused_box = fuse_matched_boxes(g, weights)
        if fused_box is not None:
            final_fused_boxes.append(fused_box)
        
    return final_fused_boxes
