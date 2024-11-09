#!/usr/bin/python3

from dataclasses import dataclass

import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse

from tools.yolo import YOLODetector
from tools.backlight_correction import BacklightCorrection
from tools.box_recognition import BoxRecognition
from tools.crosswalk_detector import CrosswalkDetector


@dataclass(frozen=True)
class Param:
    hz: int
    confidence_th_blue: float
    confidence_th_red: float
    confidence_th_crosswalk: float
    count_th_blue_while_red_detected: int
    count_th_red_while_red_detected: int
    count_th_blue_while_blue_detected: int
    count_th_red_while_blue_detected: int
    count_th_crosswalk: int
    count_th_no_vehicle: int
    count_th_allowed_time: int
    start_brightness_judge_threshold: int
    do_preprocess: bool
    weight_path: str
    weight_path_seg: str
    debug: bool

    def _print(self):
        # Print the parameters for debugging
        rospy.loginfo(f"hz: {self.hz}")
        rospy.loginfo(f"confidence_th_blue: {self.confidence_th_blue}")
        rospy.loginfo(f"confidence_th_red: {self.confidence_th_red}")
        rospy.loginfo(f"confidence_th_crosswalk: {self.confidence_th_crosswalk}")
        rospy.loginfo(f"count_th_blue_while_red_detected: {self.count_th_blue_while_red_detected}")
        rospy.loginfo(f"count_th_red_while_red_detected: {self.count_th_red_while_red_detected}")
        rospy.loginfo(f"count_th_blue_while_blue_detected: {self.count_th_blue_while_blue_detected}")
        rospy.loginfo(f"count_th_red_while_blue_detected: {self.count_th_red_while_blue_detected}")
        rospy.loginfo(f"count_th_crosswalk: {self.count_th_crosswalk}")
        rospy.loginfo(f"count_th_no_vehicle: {self.count_th_no_vehicle}")
        rospy.loginfo(f"count_th_allowed_time: {self.count_th_allowed_time}")
        rospy.loginfo(
            f"start_brightness_judge_threshold: {self.start_brightness_judge_threshold}"
        )
        rospy.loginfo(f"do_preprocess: {self.do_preprocess}")
        rospy.loginfo(f"weight_path: {self.weight_path}")
        rospy.loginfo(f"weight_path_seg: {self.weight_path_seg}")
        rospy.loginfo(f"debug: {self.debug}")


@dataclass
class Count:
    blue: int = 0
    red: int = 0
    to_start_brightness_judge: int = 0
    no_vehicle_on_crosswalk: int = 0
    allowed_time: int = 0


class TrafficlightDetector:
    def __init__(self) -> None:
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
        self._is_detecting_blue = True
        self._can_proceed = False

        self._yolo_traffic_light = YOLODetector(weight_path=self._param.weight_path, conf_th_crosswalk=self._param.confidence_th_crosswalk)
        self._yolo_crosswalk = YOLODetector(weight_path=self._param.weight_path_seg, conf_th_crosswalk=self._param.confidence_th_crosswalk)
        self._backlight_correction = BacklightCorrection()
        self._box_recognition = BoxRecognition(
            self._yolo_traffic_light, self._backlight_correction,
            self._param, self._img_pub, self._box_pub
        )
        self._crosswalk_detector = CrosswalkDetector(self._yolo_crosswalk, self._param)

        # cuda setting
        torch.cuda.set_device(0)
        self._print_cuda_status()

        # wait for services
        if not self._param.debug:
            rospy.logwarn("waiting for services")
            rospy.wait_for_service("/task/stop")

    def _load_param(self) -> None:
        self._param = Param(
            hz=rospy.get_param("~hz", 10),
            confidence_th_blue=rospy.get_param("~confidence_th_blue", 0.3),
            confidence_th_red=rospy.get_param("~confidence_th_red", 0.3),
            confidence_th_crosswalk=rospy.get_param("~confidence_th_crosswalk", 0.5),
            count_th_blue_while_red_detected=rospy.get_param("~count_th_blue_while_red_detected", 20),
            count_th_red_while_red_detected=rospy.get_param("~count_th_red_while_red_detected", 30),
            count_th_blue_while_blue_detected=rospy.get_param("~count_th_blue_while_blue_detected", 20),
            count_th_red_while_blue_detected=rospy.get_param("~count_th_red_while_blue_detected", 30),
            count_th_crosswalk=rospy.get_param("~count_th_crosswalk", 40),
            count_th_no_vehicle=rospy.get_param("~count_th_no_vehicle", 5),
            count_th_allowed_time=rospy.get_param("~count_th_allowed_time", 30),
            start_brightness_judge_threshold=rospy.get_param(
                "~do_brightness_judge_count", 10
            ),
            do_preprocess=rospy.get_param("~do_preprocess", True),
            weight_path=rospy.get_param("~weight_path", ""),
            weight_path_seg=rospy.get_param("~weight_path_seg", ""),
            debug=rospy.get_param("~debug", False),
        )
        self._param._print()

    def _print_cuda_status(self) -> None:
        rospy.loginfo(
            f"torch.cuda.is_available(): {torch.cuda.is_available()}"
        )
        rospy.loginfo(
            f"torch.cuda.current_device(): {torch.cuda.current_device()}"
        )
        rospy.loginfo(
            f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
        )

    def _request_callback(self, req: SetBool) -> SetBoolResponse:
        self._request_flag = req.data
        res: SetBoolResponse = SetBoolResponse(success=True)
        if self._request_flag:
            res.message = "Traffic light detection started."
        else:
            res.message = "Traffic light detection stopped."
        return res

    def _image_callback(self, msg: CompressedImage) -> None:
        if self._request_flag and len(msg.data) != 0:
            self._input_cvimg = CvBridge().compressed_imgmsg_to_cv2(msg)
        else:
            self._img_pub.publish(msg)

    def _reset_count_traffic_light(self) -> None:
        self._count.blue = 0
        self._count.red = 0
        self._count.allowed_time = 0

    def _run(self, _) -> None:
        # initialize when the task type is not traffic light
        if not self._request_flag:
            self._reset_count_traffic_light()
            self._count.to_start_brightness_judge = 0
            self._count.no_vehicle_on_crosswalk = 0
            self._crosswalk_detector.reset_buffer()
            self._can_proceed = False
            self._is_detecting_blue = True

        # publish flag if a blue is detected above a threshold value after
        #   a red is detected above a threshold value
        elif self._input_cvimg is not None:
            signal = self._box_recognition._judge_signal(input_cvimg=self._input_cvimg, count=self._count)

            # traffic light: red -> blue
            if self._is_detecting_blue is True:
                if signal == "signal_red":
                    self._count.red += 1
                elif (
                    signal == "signal_blue"
                    and self._count.red > self._param.count_th_red_while_red_detected
                ):
                    self._count.blue += 1
            # traffic light: blue -> red
            else:
                if signal == "signal_blue":
                    self._count.blue += 1
                elif (
                    signal == "signal_red"
                    and self._count.blue > self._param.count_th_blue_while_blue_detected
                ):
                    self._count.red += 1

            # Check for crosswalk overlap
            crosswalk_th_img = self._crosswalk_detector._cumulative_crosswalk(input_cvimg=self._input_cvimg)

            # Check if the vehicle is not on the crosswalk
            if self._crosswalk_detector._check_overlap_with_crosswalk(input_cvimg=self._input_cvimg, thresholded_img=crosswalk_th_img) is False:
                self._count.no_vehicle_on_crosswalk += 1
            else:
                self._count.no_vehicle_on_crosswalk = 0
                rospy.logwarn("Vehicle on the crosswalk")

            if self._is_detecting_blue is True:
                if self._count.blue > self._param.count_th_blue_while_red_detected: # DETECT BLUE(red->blue)
                    self._is_detecting_blue = False
                    self._reset_count_traffic_light() # Reset traffic light count for new detection cycle
                    rospy.loginfo("DETECT Blue")
            else:
                if self._count.red > self._param.count_th_red_while_blue_detected: # DETECT RED(blue->red)
                    self._is_detecting_blue = True
                    self._reset_count_traffic_light() # Reset traffic light count for new detection cycle
                    rospy.loginfo("DETECT Red")

            if self._is_detecting_blue is False: # In case of Blue signal
                if self._count.no_vehicle_on_crosswalk > self._param.count_th_no_vehicle:
                    if self._count.allowed_time < self._param.count_th_allowed_time:
                        self._can_proceed = True
                    else:
                        rospy.logwarn("Allowed time exceeded. Not proceeding.")
                else:
                    self._count.allowed_time += 1

            if self._can_proceed:
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

    def __call__(self) -> None:
        duration = int(1.0 / self._param.hz * 1e9)
        rospy.Timer(rospy.Duration(nsecs=duration), self._run)
        rospy.spin()


if __name__ == "__main__":
    TrafficlightDetector()()
