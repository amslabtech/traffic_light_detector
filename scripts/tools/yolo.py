import numpy as np
from ultralytics import YOLO
import cv2
import rospy


class YOLODetector:
    def __init__(self, weight_path: str) -> None:
        self._model = YOLO(weight_path)
    
    def _traffic_light_yolo(self, input_img: np.ndarray) -> tuple: 
        max_conf = -1
        max_conf_class = None
        output = None
        # self._model() returns list of
        #   class:ultralytics.engine.results.Results
        yolo_output = self._model(input_img, classes=[15, 16], conf=0, verbose=True)

        if len(yolo_output[0]) != 0: # Checks if there are any detections
            max_conf = yolo_output[0].boxes[0].conf.item()
            max_conf_class = yolo_output[0].boxes[0].cls.item()
            output = yolo_output[0]

        return output, max_conf, int(max_conf_class)

    def _crosswalk_yolo(self, input_img: np.ndarray) -> np.ndarray:   
        # crosswalk
        yolo_output = self._model(input_img, classes=[13], conf=0.5, verbose=False)
        
        if yolo_output is None or len(yolo_output) == 0:  # Checks if detections are empty
            rospy.logwarn("yolo_output is None")
            return np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)  # Return blank image

        output_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

        if hasattr(yolo_output[0], 'masks') and yolo_output[0].masks is not None:
            for segment in yolo_output[0].masks.xy:
                mask = np.zeros_like(output_img)
                cv2.fillPoly(mask, [segment.astype(np.int32)], 1) # Fill mask with 1
                output_img = cv2.bitwise_or(output_img, mask)
        else:
            rospy.logwarn("No segmentation masks found for detected crosswalk in YOLO output")
            
        return output_img

    def _vehicle_yolo(self, input_img: np.ndarray) -> np.ndarray:
        # bicycle, car, motorbike, bus, train, truck
        yolo_output = self._model(input_img, classes=[1, 2, 3, 4, 5, 6], conf=0.5, verbose=False)
        
        if yolo_output is None or len(yolo_output) == 0: # Checks if detections are empty
            rospy.logwarn("yolo_output is None")
            return np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)  # Return blank image

        output_img = np.zeros((input_img.shape[0], input_img.shape[1]), dtype=np.uint8)

        if hasattr(yolo_output[0], 'masks') and yolo_output[0].masks is not None:
            for segment in yolo_output[0].masks.xy:
                mask = np.zeros_like(output_img)
                cv2.fillPoly(mask, [segment.astype(np.int32)], 1)  # Fill mask with 1
                output_img = cv2.bitwise_or(output_img, mask)
            
        return output_img