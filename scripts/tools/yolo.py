import numpy as np
from ultralytics import YOLO
import cv2
import rospy
from typing import Tuple


class YOLODetector:
    def __init__(self, weight_path: str, conf_th_crosswalk: float, debug_yolo: bool) -> None:
        self._conf_th_crosswalk = conf_th_crosswalk
        self._debug_yolo = debug_yolo
        self._model = YOLO(weight_path)
    
    def _traffic_light_yolo(self, input_img: np.ndarray) -> tuple: 
        max_conf = -1
        max_conf_class = None
        output = None
        # self._model() returns list of
        #   class:ultralytics.engine.results.Results
        yolo_output = self._model(input_img, classes=[15, 16], conf=0, verbose=self._debug_yolo)

        if len(yolo_output[0]) != 0: # Checks if there are any detections
            max_conf = yolo_output[0].boxes[0].conf.item()
            max_conf_class = yolo_output[0].boxes[0].cls.item()
            output = yolo_output[0]

        return output, max_conf, int(max_conf_class) if max_conf_class is not None else -1

    def _crosswalk_and_vehicle_yolo(self, input_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Run YOLO inference once, filter by crosswalk and vehicle classes
        yolo_output = self._model(input_img, classes=[1, 2, 3, 4, 5, 6, 13], conf=self._conf_th_crosswalk, verbose=self._debug_yolo)
            
        # Return blank images if no detections
        if not yolo_output or len(yolo_output[0]) == 0:
            rospy.logwarn("yolo_output is None or empty")
            crosswalk_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
            vehicle_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
            return crosswalk_img, vehicle_img

        # Initialize output images
        crosswalk_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)
        vehicle_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

        # Process each detection by class
        for detection in yolo_output[0]:
            # Ensure there are masks available
            if not hasattr(detection, 'masks') or detection.masks is None:
                rospy.logwarn("No masks in YOLO output")
                continue

            for segment in detection.masks.xy:
                mask = np.zeros_like(crosswalk_img)
                cv2.fillPoly(mask, [segment.astype(np.int32)], 1)  # Fill mask with 1

                # Iterate through the class IDs of each detection box
                for cls_id in detection.boxes.cls:
                    cls_id = int(cls_id.item())  # Convert tensor to int
                    if cls_id == 13:  # Crosswalk class
                        crosswalk_img = cv2.bitwise_or(crosswalk_img, mask)
                    elif cls_id in [1, 2, 3, 4, 5, 6]:  # Vehicle-related classes
                        vehicle_img = cv2.bitwise_or(vehicle_img, mask)

        return crosswalk_img, vehicle_img
