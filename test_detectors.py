import numpy as np
from dataset import TruckDriveDataset
from detectors.ouster import detect_ouster
from detectors.aeva import detect_aeva
from detectors.conti542 import detect_conti542

def main():
    dataset = TruckDriveDataset('data/TruckDrive', 'scene_28_1')
    if len(dataset) == 0:
        print("No frames in dataset!")
        return
        
    frame_data = dataset[0]
    print(f"Testing on Frame {frame_data['sync_id']}")
    
    if len(frame_data['ouster']) > 0:
        ouster_boxes = detect_ouster(frame_data['ouster'])
        print(f"Ouster found {len(ouster_boxes)} boxes")
        if ouster_boxes:
            print("Sample Ouster box:", ouster_boxes[0])
            
    if frame_data['aeva'] is not None:
        aeva_boxes = detect_aeva(frame_data['aeva'])
        print(f"Aeva found {len(aeva_boxes)} boxes")
        if aeva_boxes:
            print("Sample Aeva box:", aeva_boxes[0])
            
    if frame_data['conti542'] is not None:
        conti_boxes = detect_conti542(frame_data['conti542'])
        print(f"Conti542 found {len(conti_boxes)} boxes")
        if conti_boxes:
            print("Sample Conti542 box:", conti_boxes[0])

if __name__ == '__main__':
    main()
