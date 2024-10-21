from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, weight_path: str) -> None:
        self._model = YOLO(weight_path)
    
    def _traffic_light_yolo(self, input_img: np.ndarray) -> tuple:
        max_conf = -1
        max_conf_class = None
        output = None
        # self._model() returns list of
        #   class:ultralytics.engine.results.Results
        yolo_output = self._model(input_img, classes=[15, 16], conf=0)

        if len(yolo_output[0]) != 0:
            max_conf = yolo_output[0].boxes[0].conf.item()
            max_conf_class = yolo_output[0].boxes[0].cls.item()
            output = yolo_output[0]

        return output, max_conf, int(max_conf_class)
