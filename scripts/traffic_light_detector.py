#!/usr/bin/python3

from dataclasses import dataclass

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse
from ultralytics import YOLO


@dataclass(frozen=True)
class Param:
    hz: int
    conf_threshold_blue: float
    conf_threshold_red: float
    count_threshold_blue: int
    count_threshold_red: int
    start_brightness_judge_threshold: int
    do_preprocess: bool
    weight_path: str
    debug: bool

    def _print(self):
        rospy.loginfo(f"hz: {self.hz}")
        rospy.loginfo(f"conf_threshold_blue: {self.conf_threshold_blue}")
        rospy.loginfo(f"conf_threshold_red: {self.conf_threshold_red}")
        rospy.loginfo(f"count_threshold_blue: {self.count_threshold_blue}")
        rospy.loginfo(f"count_threshold_red: {self.count_threshold_red}")
        rospy.loginfo(
            f"start_brightness_judge_threshold: {self.start_brightness_judge_threshold}"
        )
        rospy.loginfo(f"do_preprocess: {self.do_preprocess}")
        rospy.loginfo(f"weight_path: {self.weight_path}")
        rospy.loginfo(f"debug: {self.debug}")


@dataclass
class Count:
    blue: int = 0
    red: int = 0
    to_start_brightness_judge: int = 0


class TrafficlightDetector:
    def __init__(self):
        rospy.init_node("traffic_light_detector")

        self._img_pub = rospy.Publisher(
            "/yolo_result/image_raw/compressed", CompressedImage, queue_size=10
        )
        self._box_pub = rospy.Publisher(
            "/yolo_box/image_raw/compressed", CompressedImage, queue_size=10
        )
        self._img_sub = rospy.Subscriber(
            "/CompressedImage", CompressedImage, self._image_callback
        )
        self._request_server = rospy.Service(
            "~request", SetBool, self._request_callback
        )
        self._task_stop_client = rospy.ServiceProxy("/task/stop", SetBool)

        self._load_param()
        self._count = Count()
        self._request_flag = self._param.debug
        self._input_cvimg = None
        self._stored_boxes = []
        self._model = YOLO(self._param.weight_path)

        # cuda setting
        torch.cuda.set_device(0)
        self._print_cuda_status()

        # wait for services
        if not self._param.debug:
            rospy.logwarn("waiting for services")
            rospy.wait_for_service("/task/stop")

    def _load_param(self):
        self._param = Param(
            hz=rospy.get_param("~hz", 10),
            conf_threshold_blue=rospy.get_param("~conf_threshold_blue", 0.3),
            conf_threshold_red=rospy.get_param("~conf_threshold_red", 0.3),
            count_threshold_blue=rospy.get_param("~count_threshold_blue", 20),
            count_threshold_red=rospy.get_param("~count_threshold_red", 50),
            start_brightness_judge_threshold=rospy.get_param(
                "~do_brightness_judge_couont", 10
            ),
            do_preprocess=rospy.get_param("~do_preprocess", True),
            weight_path=rospy.get_param("~weight_path", ""),
            debug=rospy.get_param("~debug", False),
        )
        self._param._print()

    def _print_cuda_status(self):
        rospy.loginfo(
            f"torch.cuda.is_available(): {torch.cuda.is_available()}"
        )
        rospy.loginfo(
            f"torch.cuda.current_device(): {torch.cuda.current_device()}"
        )
        rospy.loginfo(
            f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
        )

    def _request_callback(self, req: SetBool):
        self._request_flag = req.data
        res: SetBoolResponse = SetBoolResponse(success=True)
        if self._request_flag:
            res.message = "Traffic light detection started."
        else:
            res.message = "Traffic light detection stopped."
        return res

    def _image_callback(self, msg: CompressedImage):
        if self._request_flag and len(msg.data) != 0:
            self._input_cvimg = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self._img_pub.publish(msg)

    def _visualize_box(self, img=None):
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if img is None:
            img = self._input_cvimg

        result_msg = CvBridge().cv2_to_compressed_imgmsg(img)
        self._box_pub.publish(result_msg)

    def _visualize(self, img=None):
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if img is None:
            img = self._input_cvimg

        result_msg = CvBridge().cv2_to_compressed_imgmsg(img)
        self._img_pub.publish(result_msg)

    def _backlight_correction(self):
        # グレースケール変換
        gray_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2GRAY)

        # ガンマ補正
        # gamma = 1.2
        gamma = 1.0
        # gamma = 2.0
        # gamma = 2.2
        corrected_gamma = np.power(gray_image / 255.0, 1.0 / gamma) * 255
        corrected_gamma = corrected_gamma.astype(np.uint8)

        # マルチスケールCLAHE
        # clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
        corrected1 = clahe1.apply(corrected_gamma)
        corrected2 = clahe2.apply(corrected_gamma)
        corrected_gray = cv2.addWeighted(corrected1, 0.5, corrected2, 0.5, 0)

        # 色情報を保持するための変換
        yuv_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = corrected_gray
        corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        return corrected_image

    def _detect_backlight(self):
        # 画像をグレースケールに変換
        gray_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2GRAY)

        # ハイライト領域の検出
        _, highlight_thresh = cv2.threshold(
            gray_image, 200, 255, cv2.THRESH_BINARY
        )

        # シャドウ領域の検出
        _, shadow_thresh = cv2.threshold(
            gray_image, 50, 255, cv2.THRESH_BINARY_INV
        )

        # 画像の中央部分を取得
        h, w = self._input_cvimg.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(h, w) // 4
        roi = gray_image[
            center_y - roi_size : center_y + roi_size,
            center_x - roi_size : center_x + roi_size,
        ]

        # ROI内のハイライト・シャドウ領域をカウント
        num_highlight_pixels_roi = np.count_nonzero(
            highlight_thresh[
                center_y - roi_size : center_y + roi_size,
                center_x - roi_size : center_x + roi_size,
            ]
        )
        num_shadow_pixels_roi = np.count_nonzero(
            shadow_thresh[
                center_y - roi_size : center_y + roi_size,
                center_x - roi_size : center_x + roi_size,
            ]
        )

        # ヒストグラムを計算
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        highlight_hist_peak = np.sum(hist[200:])
        shadow_hist_peak = np.sum(hist[:50])

        # 画像のサイズ
        total_pixels = self._input_cvimg.shape[0] * self._input_cvimg.shape[1]

        # ハイライト・シャドウ領域の割合を計算
        highlight_ratio = num_highlight_pixels_roi / (
            roi_size * 2 * roi_size * 2
        )
        shadow_ratio = num_shadow_pixels_roi / (roi_size * 2 * roi_size * 2)

        # ヒストグラムからの割合を計算
        highlight_hist_ratio = highlight_hist_peak / total_pixels
        shadow_hist_ratio = shadow_hist_peak / total_pixels

        # コントラストを計算
        contrast = np.std(roi)

        # 逆光判定の条件を調整
        if (
            (highlight_ratio > 0.1 and shadow_ratio > 0.1) or contrast < 20
        ) or (highlight_hist_ratio > 0.1 and shadow_hist_ratio > 0.1):
            return True
        else:
            return False

    def _draw_box(self, img, box, color=(0, 0, 0)):
        thickness = 4  # バウンディングボックスの線の太さ
        x1, y1, x2, y2 = box[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _draw_boxes(self, img, boxes, color=(0, 0, 0)):
        thickness = 2  # バウンディングボックスの線の太さ
        for box in boxes:
            x1, y1, x2, y2 = box[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def _contain_yellow_px(self, box, img):

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

    def _within_appropriate_aspect(self, box):
        x1, y1, x2, y2 = box[0]
        h = y2 - y1
        w = x2 - x1

        if 1.55 <= h / w <= 1.85:
            return True
        else:
            return False

    def _store_box(self, yolo_output):

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

    def _brightness_judge(self, yolo_output):

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

    def _yolo(self, input_img):

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

    def _preprocess(self):
        if self._detect_backlight() and self._param.do_preprocess:
            return self._backlight_correction()
        else:
            return self._input_cvimg

    def _judge_signal(self) -> str:

        signal = None
        visualize_cvimg = None
        input_img = self._preprocess()
        yolo_output, max_conf, max_conf_class = self._yolo(input_img)

        if max_conf_class is not None:

            self._store_box(yolo_output)

            if (
                max_conf_class == 16
                and max_conf > self._param.conf_threshold_blue
            ) or (
                max_conf_class == 15
                and max_conf > self._param.conf_threshold_red
            ):

                signal = yolo_output.names.get(max_conf_class)
                visualize_cvimg = yolo_output[0].plot()
                self._count.to_start_brightness_judge = 0

            elif (
                self._count.to_start_brightness_judge
                < self._param.start_brightness_judge_threshold
            ):
                self._count.to_start_brightness_judge += 1
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

    def _run(self, _):
        # initialize when the task type is not traffic light
        if not self._request_flag:
            self._count.blue = 0
            self._count.red = 0
        # publish flag if a blue is detected above a threshold value after
        #   a red is detected above a threshold value
        elif self._input_cvimg is not None:
            signal = self._judge_signal()

            if signal == "signal_red":
                self._count.red += 1
            elif (
                signal == "signal_blue"
                and self._count.red > self._param.count_threshold_red
            ):
                self._count.blue += 1

            if self._count.blue > self._param.count_threshold_blue:
                if self._param.debug:
                    rospy.logwarn("cross traffic light")
                    self._request_flag = False
                    return
                while not rospy.is_shutdown():
                    try:
                        resp = self._task_stop_client(False)
                        rospy.logwarn(resp.message)
                        self._request_flag = False
                        break
                    except rospy.ServiceException as e:
                        rospy.logwarn(e)

    def __call__(self):
        duration = int(1.0 / self._param.hz * 1e9)
        rospy.Timer(rospy.Duration(nsecs=duration), self._run)
        rospy.spin()


if __name__ == "__main__":
    TrafficlightDetector()()
