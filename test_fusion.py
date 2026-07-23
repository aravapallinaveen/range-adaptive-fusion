import numpy as np
from dataset import TruckDriveDataset
from detectors.ouster import detect_ouster
from detectors.aeva import detect_aeva
from detectors.conti542 import detect_conti542
from fusion.baseline import baseline_fusion
from fusion.adaptive import adaptive_fusion

def main():
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        print("No frames found!")
        return
        
    frame_data = dataset[0]
    print(f"Testing on Frame {frame_data['sync_id']}")
    
    ouster_boxes = []
    if len(frame_data['ouster']) > 0:
        ouster_boxes = detect_ouster(frame_data['ouster'])
        
    aeva_boxes = []
    if frame_data['aeva'] is not None:
        aeva_boxes = detect_aeva(frame_data['aeva'])
        
    conti_boxes = []
    if frame_data['conti542'] is not None:
        conti_boxes = detect_conti542(frame_data['conti542'])
    
    print(f"Raw Detections: Ouster={len(ouster_boxes)}, Aeva={len(aeva_boxes)}, Conti={len(conti_boxes)}")
    
    baseline_boxes = baseline_fusion(ouster_boxes, aeva_boxes, conti_boxes)
    print(f"Baseline Fused Boxes: {len(baseline_boxes)}")
    if baseline_boxes:
        print(f"Sample Baseline Box Source: {baseline_boxes[0]['sensor_source']}")
    
    adaptive_boxes = adaptive_fusion(ouster_boxes, aeva_boxes, conti_boxes)
    print(f"Adaptive Fused Boxes: {len(adaptive_boxes)}")
    if adaptive_boxes:
        print(f"Sample Adaptive Box Source: {adaptive_boxes[0]['sensor_source']}")
    
if __name__ == '__main__':
    main()
