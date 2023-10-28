#!/usr/bin/python3
import rospy
import os
import cv2
import ultralytics
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import numpy as np

class TrafficlightDetector:

    def __init__(self):
        ### ros settings ###
        rospy.init_node("traffic_light_detector")
        self._pub_img = rospy.Publisher('/yolo_result/image_raw/compressed', CompressedImage, queue_size=10)
        self._pub_box = rospy.Publisher('/yolo_box/image_raw/compressed', CompressedImage, queue_size=10)
        self._pub_flag = rospy.Publisher('/cross_traffic_light_flag', Bool, queue_size=1)
        self._sub_img = rospy.Subscriber('/CompressedImage', CompressedImage, self._image_callback)
        self._sub_exe_flag = rospy.Subscriber('/request_detect_traffic_light', Bool, self._exec_flag_callback)
        ### ros params ###
        self._conf_threshold_blue = rospy.get_param('~conf_threshold_blue', 0.3)
        self._conf_threshold_red = rospy.get_param('~conf_threshold_red', 0.3)
        self._min_conf = rospy.get_param('~min_conf', 1e-7)
        self._count_threshold_blue = rospy.get_param('~count_threshold_blue', 20)
        self._count_threshold_red = rospy.get_param('~count_threshold_red', 50)
        self._aspect_ratio_threshold = rospy.get_param('~aspect_ratio_threshold', 2.0)
        self._hz = rospy.get_param('~hz', 10)
        ### basic setting ###
        self._bridge = CvBridge()
        self._exec_flag = False
        self._result_msg = CompressedImage()
        self._callback_flag = False
        ### device setting ###
        torch.cuda.set_device(0)
        ### yolo weights ###
        weight_list = ["vidvip_yolov8n_2023-05-19.pt", "vidvipo_yolov8x_2023-05-19.pt", "yolov8n.pt"]
        self._model=YOLO(os.path.join( "weights", weight_list[1]))
        ### basic config ###
        self._count_blue=0
        self._count_red = 0

    def _exec_flag_callback(self, msg: Bool):
        self._exec_flag = msg.data

    def _image_callback(self, msg: CompressedImage):
        if(self._exec_flag):
            self._input_cvimg = self._bridge.compressed_imgmsg_to_cv2(msg)
        else:
            self._pub_img.publish(msg)
        self._callback_flag = True

    def _visualize_box(self, img=None):
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if(img is None):
            img=self._input_cvimg

        result_msg = self._bridge.cv2_to_compressed_imgmsg(img)
        self._pub_box.publish(result_msg)

    def _visualize(self, img=None):
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        if(img is None):
            img=self._input_cvimg

        result_msg = self._bridge.cv2_to_compressed_imgmsg(img)
        self._pub_img.publish(result_msg)


    def _valid_boxes_judge(self, boxes):
            # HSVに変換
        hsv = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2HSV)
        valid_boxes = []

        print(type(boxes[0]))
        return "a", "a"

        # for box in boxes:
        #     box = box.xywh.to('cpu').detach().numpy().astype(int)
        #     x, y, w, h = box[0]
        #     # 縦横比の確認
        #     if 1.8 <= h/w <= 2.2:
        #         upper_hsv = hsv[y:y+h//3, x:x+w]
        #         lower_hsv = hsv[y+2*h//3:y+h, x:x+w]
        #         upper_red = cv2.inRange(upper_hsv, (0, 50, 50), (10, 255, 255))
        #         lower_blue = cv2.inRange(lower_hsv, (110, 50, 50), (130, 255, 255))
        #         if np.sum(upper_red) > 0 and np.sum(lower_blue) > 0:
        #             valid_boxes.append(box)
        # return valid_boxes

    def _yolo(self, input_img):

        max_conf = -1
        max_conf_class = None
        yolo_result = self._model(input_img, classes=[15, 16], conf=self._min_conf) #self._model() returns list of class:ultralytics.engine.results.Results

        for box in yolo_result[0].boxes:
            if box.conf > max_conf:
                max_conf = box.conf
                max_conf_class = int(box.cls.item())

        if(max_conf_class is not None):
            if((max_conf_class==16 and max_conf > self._conf_threshold_blue) or
               (max_conf_class==15 and max_conf > self._conf_threshold_red)):
                return yolo_result[0], yolo_result[0].names.get(max_conf_class)
            else:
                return yolo_result[0], 'under_threshold'
        else:
            return None, 'no_boxes'


    def _backlight_correction(self):
        # 1. グレースケール変換
        gray_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2GRAY)

        # 2. ガンマ補正
        gamma = 1.2
        corrected_gamma = np.power(gray_image / 255.0, 1.0 / gamma) * 255
        corrected_gamma = corrected_gamma.astype(np.uint8)

        # 3. マルチスケールCLAHE
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        corrected1 = clahe1.apply(corrected_gamma)
        corrected2 = clahe2.apply(corrected_gamma)
        corrected_gray = cv2.addWeighted(corrected1, 0.5, corrected2, 0.5, 0)

        # 4. 色情報を保持するための変換
        yuv_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = corrected_gray
        corrected_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

        return corrected_image

    def _detect_backlight(self):
        # 画像をグレースケールに変換
        gray_image = cv2.cvtColor(self._input_cvimg, cv2.COLOR_BGR2GRAY)

        # ハイライト領域の検出
        _, highlight_thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        # シャドウ領域の検出
        _, shadow_thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)

        # 画像の中央部分を取得
        h, w = self._input_cvimg.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(h, w) // 4
        roi = gray_image[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]

        # ROI内のハイライト・シャドウ領域をカウント
        num_highlight_pixels_roi = np.count_nonzero(highlight_thresh[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size])
        num_shadow_pixels_roi = np.count_nonzero(shadow_thresh[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size])

        # ヒストグラムを計算
        hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])
        highlight_hist_peak = np.sum(hist[200:])
        shadow_hist_peak = np.sum(hist[:50])

        # 画像のサイズ
        total_pixels = self._input_cvimg.shape[0] * self._input_cvimg.shape[1]

        # ハイライト・シャドウ領域の割合を計算
        highlight_ratio = num_highlight_pixels_roi / (roi_size * 2 * roi_size * 2)
        shadow_ratio = num_shadow_pixels_roi / (roi_size * 2 * roi_size * 2)

        # ヒストグラムからの割合を計算
        highlight_hist_ratio = highlight_hist_peak / total_pixels
        shadow_hist_ratio = shadow_hist_peak / total_pixels

        # コントラストを計算
        contrast = np.std(roi)

        # 逆光判定の条件を調整
        if ((highlight_ratio > 0.1 and shadow_ratio > 0.1) or contrast < 20) or (highlight_hist_ratio > 0.1 and shadow_hist_ratio > 0.1):
            return True
        else:
            return False

    def _preprocess(self):
        if self._detect_backlight():
            rospy.logwarn('BLACK LIGHT IS DETECTED')
            # self._visualize_box(self._backlight_correction())
            return self._backlight_correction()
        else:
            return self._input_cvimg

    def _judge_singal(self) -> str:

        input_img = self._preprocess()
        yolo_result, judge_from_yolo = self._yolo(input_img)

        if(judge_from_yolo == 'no_boxes'):
            self._visualize()
            rospy.logerr("NOT WORKING")
            return 'not_working'
        elif(judge_from_yolo == 'under_threshold'):
            valid_box, judge_from_position_of_yellow_pixel = self._valid_boxes_judge(yolo_result.boxes)
            self._visualize(yolo_result[0].plot())
            return judge_from_position_of_yellow_pixel
        else:
            self._visualize(yolo_result[0].plot())
            # self._visualize(yolo_result[0].plot())
            return judge_from_yolo

    def _run(self, _):
        cross_traffic_light_flag = False
        ### initialize when the task type is not traffic light
        if(not self._callback_flag):
            pass
        if(not self._exec_flag):
            self._count_blue=0
            self._count_red=0
        ### publish flag if a blue is detected above a threshold value after a red is detected above a threshold value
        else:
            signal = self._judge_singal()

            if(signal == "signal_red"):
                self._count_red += 1
            elif(signal == "signal_blue" and self._count_red > self._count_threshold_red):
                self._count_blue += 1

            if(self._count_blue > self._count_threshold_blue):
                cross_traffic_light_flag = True
                self._count_red = 0
                self._count_blue = 0
        self._pub_flag.publish(cross_traffic_light_flag)
        self._callback_flag = False

    def __call__(self):
            duration = int(1.0 / self._hz * 1e9)
            rospy.Timer(rospy.Duration(nsecs=duration), self._run)
            rospy.spin()

if __name__=="__main__":
    TrafficlightDetector()()


