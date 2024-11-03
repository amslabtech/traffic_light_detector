import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge


class BoxRecognition:
    def __init__(self, yolo_detector, backlight_correction, param, img_pub, box_pub):
        self._stored_boxes = []
        self._yolo_detecter = yolo_detector
        self._backlight_correction = backlight_correction
        self._param = param
        self._img_pub = img_pub
        self._box_pub = box_pub
    
    def _draw_box(self, img: np.ndarray, box: tuple, color: tuple = (0, 0, 0)) -> np.ndarray:
        # Draws a bounding box on the image with specified thickness and color
        thickness = 4
        x1, y1, x2, y2 = box[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _draw_boxes(self, img: np.ndarray, boxes: list, color: tuple = (0, 0, 0)) -> np.ndarray:
        # Draws multiple bounding boxes
        thickness = 2
        for box in boxes:
            x1, y1, x2, y2 = box[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _visualize_box(self, img: np.ndarray = None) -> None:
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if img is None:
            img = self._input_cvimg

        result_msg = CvBridge().cv2_to_compressed_imgmsg(img)
        self._box_pub.publish(result_msg)

    def _visualize(self, img: np.ndarray = None) -> None:
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if img is None:
            img = self._input_cvimg

        result_msg = CvBridge().cv2_to_compressed_imgmsg(img)
        self._img_pub.publish(result_msg)

    def _contain_yellow_px(self, box: tuple, img: np.ndarray) -> bool:
        # Checks if yellow pixels are within the specified box
        lower_yellow_h = 23
        upper_yellow_h = 30
        x1, y1, x2, y2 = box[0]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        box_region_h = hsv[y1:y2, x1:x2, 0]

        yellow_mask_h = (box_region_h >= lower_yellow_h) & (
            box_region_h <= upper_yellow_h
        )
        yellow_count = np.sum(yellow_mask_h)

        if yellow_count > 5:
            return True
        else:
            return False

    def _within_appropriate_aspect(self, box: tuple) -> bool:
        # Checks if the box aspect ratio is within the specified range
        x1, y1, x2, y2 = box[0]
        h = y2 - y1
        w = x2 - x1

        if 1.55 <= h / w <= 1.85:
            return True
        else:
            return False

    def _store_box(self, yolo_output) -> None:
        # Stores detected boxes that meet specific criteria
        tmp_boxes = []
        valid_box = None

        if len(self._stored_boxes) > 10:
            self._stored_boxes.pop()

        for box in yolo_output.boxes:
            box_xyxy = box.xyxy.to("cpu").detach().numpy().astype(int)
            # print(box_xyxy[0])
            if self._within_appropriate_aspect(
                box_xyxy
            ) and self._contain_yellow_px(box_xyxy, yolo_output.orig_img):
                tmp_boxes.append((box_xyxy[0], box.conf.item()))  # tuple

        # ==== DEBUG ====
        stored_boxes = self._draw_boxes(
            img=yolo_output.orig_img, boxes=self._stored_boxes
        )
        self._visualize_box(stored_boxes)
        # ==== DEBUG ====

        if len(tmp_boxes) > 0:
            valid_box = max(tmp_boxes, key=lambda x: x[1])
            # print("VALID BOX:", valid_box)
            self._stored_boxes.append(valid_box)

        self._stored_boxes.sort(key=lambda x: x[1], reverse=True)

    def _brightness_judge(self, yolo_output) -> tuple:
        # Determines the signal color based on brightness within a detected box
        signal = None
        output = None

        if len(self._stored_boxes) > 0:

            valid_box = self._stored_boxes[0]

            x1, y1, x2, y2 = valid_box[0]
            h = y2 - y1

            hsv = cv2.cvtColor(yolo_output.orig_img, cv2.COLOR_BGR2HSV)
            upper_hsv = hsv[y1 : y1 + h // 2, x1:x2, :]
            lower_hsv = hsv[y1 + h // 2 : y2, x1:x2, :]

            upper_brightness = np.mean(upper_hsv[:, :, 2])
            lower_brightness = np.mean(lower_hsv[:, :, 2])

            if self._contain_yellow_px(valid_box, yolo_output.orig_img):
                if upper_brightness < lower_brightness:
                    color = (0, 255, 0)  # バウンディングボックスの色 (BGR形式)
                    signal = "signal_blue"
                else:
                    color = (0, 0, 255)
                    signal = "signal_red"
            else:
                color = (0, 0, 0)
                signal = "unknown"

            output = self._draw_box(yolo_output.orig_img, valid_box, color)
        else:
            output = yolo_output.orig_img
            rospy.logerr("NO VALID BOX")

        return signal, output
    
    def _judge_signal(self, input_cvimg: np.ndarray, count) -> str:
        # Evaluates the traffic signal status based on YOLO output and brightness
        signal = None
        visualize_cvimg = None
        input_img = self._backlight_correction._preprocess(input_cvimg, self._param)
        yolo_output, max_conf, max_conf_class = self._yolo_detecter._traffic_light_yolo(input_img)

        if max_conf_class is not None:

            self._store_box(yolo_output)

            if (
                max_conf_class == 16
                and max_conf > self._param.confidence_threshold_blue
            ) or (
                max_conf_class == 15
                and max_conf > self._param.confidence_threshold_red
            ):

                signal = yolo_output.names.get(max_conf_class)
                visualize_cvimg = yolo_output[0].plot()
                count.to_start_brightness_judge = 0

            elif (
                count.to_start_brightness_judge
                < self._param.start_brightness_judge_threshold
            ):
                count.to_start_brightness_judge += 1
                visualize_cvimg = yolo_output[0].plot()
                rospy.logwarn("UNDER THRESHOLD")
            else:
                signal, visualize_cvimg = self._brightness_judge(yolo_output)
                rospy.logwarn("BRIGHTNESS JUDGE")
        else:
            visualize_cvimg = input_img
            rospy.logerr("NOT WORKING")
        self._visualize(visualize_cvimg)
        return signal