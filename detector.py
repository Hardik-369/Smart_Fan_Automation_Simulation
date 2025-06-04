import cv2
import numpy as np
from typing import List, Tuple

class MobileNetSSDDetector:
    """Class to handle person detection using MobileNet-SSD model."""
    
    def __init__(self, prototxt_path: str, model_path: str):
        """
        Initialize the detector with MobileNet-SSD model.
        
        Args:
            prototxt_path (str): Path to the .prototxt file.
            model_path (str): Path to the .caffemodel file.
        """
        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5, 
               nms_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in a frame and return bounding boxes.
        
        Args:
            frame (np.ndarray): Input frame for detection.
            confidence_threshold (float): Minimum confidence for detections.
            nms_threshold (float): Non-maximum suppression threshold.
        
        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes [x, y, w, h].
        """
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        person_boxes = []
        person_scores = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            class_id = int(detections[0, 0, i, 1])
            
            if class_id == 1 and confidence > confidence_threshold:  # Class ID 1 is 'person' in COCO
                left = int(detections[0, 0, i, 3] * width)
                top = int(detections[0, 0, i, 4] * height)
                right = int(detections[0, 0, i, 5] * width)
                bottom = int(detections[0, 0, i, 6] * height)
                box = [left, top, right - left, bottom - top]
                person_boxes.append(box)
                person_scores.append(float(confidence))
        
        selected_boxes = []
        if person_boxes:
            indices = cv2.dnn.NMSBoxes(person_boxes, person_scores, confidence_threshold, nms_threshold)
            # Ensure indices are properly formatted
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            selected_boxes = [person_boxes[idx] for idx in indices]
        
        return selected_boxes
