#!/usr/bin/python3

from dataclasses import dataclass

import cv2
import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse

from tools.yolo import YOLODetector
from tools.backlight_correction import BacklightCorrection
from tools.box_recognition import BoxRecognition


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
        
        self._yolo_detecter = YOLODetector(weight_path=self._param.weight_path)
        self._backlight_correction = BacklightCorrection()
        self._box_reconginition = BoxRecognition(self._yolo_detecter, self._backlight_correction, self._img_pub, self._box_pub)

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
                
    def _run(self, _):
        # initialize when the task type is not traffic light
        if not self._request_flag:
            self._count.blue = 0
            self._count.red = 0
        # publish flag if a blue is detected above a threshold value after
        #   a red is detected above a threshold value
        elif self._input_cvimg is not None:
            signal = self._box_reconginition._judge_signal(input_cvimg=self._input_cvimg, param=self._param, count=self._count)

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
