#!/usr/bin/python3
import rospy
import os
import cv2
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
        self._min_conf = rospy.get_param('~min_conf', 1e-5)
        self._count_threshold_blue = rospy.get_param('~count_threshold_blue', 20)
        self._count_threshold_red = rospy.get_param('~count_threshold_red', 50)
        self._aspect_ratio_threshold = rospy.get_param('~aspect_ratio_threshold', 2.0)
        self._hz = rospy.get_param('~hz', 10)
        ### basic setting ###
        self._bridge = CvBridge()
        self._exec_flag = False
        self._result_msg = CompressedImage()
        # self._callback_flag = False
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

    def _visualize(self, img):
        # cv_img = img.to('cpu').detach().numpy().astype(int)

        print(type(img))
        result_msg = self._bridge.cv2_to_compressed_imgmsg(img)
        self._pub_img.publish(result_msg)


    def _pixel_judge(self):
        pass

    def _yolo(self) -> str:

        max_conf = -1
        max_conf_class = None
        yolo_infer = self._model(self._input_cvimg, classes=[15, 16], conf=self._min_conf) #self._model() returns list of class:ultralytics.engine.results.Results

        for box in yolo_infer[0].boxes:
            if box.conf > max_conf:
                max_conf = box.conf
                max_conf_class = int(box.cls.item())

        print("Type:", type(self._input_cvimg), type(yolo_infer[0].plot()))
        if(max_conf_class is not None):
            if((max_conf_class==16 and max_conf > self._conf_threshold_blue) or
               (max_conf_class==15 and max_conf > self._conf_threshold_red)):
                self._visualize(yolo_infer[0].plot())
                return yolo_infer[0].names.get(max_conf_class)
            else:
                self._visualize(self._input_cvimg)
                return 'not_working'


        # if max_conf_class is not None:
        #     print("Max Confidence Class:", max_conf_class)
        # else:
        #     print("No boxes with confidence > max_conf found.")


        # if(max_conf_class )        print("Max Conf Class:", max_conf_class)



            # if(int(box.cls.item()) == 16 and box.conf > self.conf_threshold_blue):
            #     contain_blue = True
            # elif(int(box.cls.item()) == 15 and box.conf > self.conf_threshold_red):
            #     contain_red = True
            # # if there is a single bounding box, it is cut from the image
            # if(contain_blue + contain_red == 1):
            #     box_array = box.xyxy
            #     box_numpy = box_array.to('cpu').detach().numpy().astype(int)
            #     x1, y1, x2, y2 = box_numpy[0]
            #     box_img = cv_image[y1:y2, x1:x2]
            #     # output_box = self.bridge.cv2_to_compressed_imgmsg(box_img)
            #     # self.pub_box.publish(output_box)
            #     infer_box = self.model(box_img, classes=[15, 16], conf=self.conf_threshold_blue)
            #     output_box = self.bridge.cv2_to_compressed_imgmsg(infer_box[0].plot())
            #     self.pub_box.publish(output_box)
        #
        # if(not contain_blue):
        #     self.count_red += contain_red
        #
        # if(not contain_red and self.count_red > self.count_threshold_red):
        #     self.count_blue += contain_blue
        #     if(self.count_blue > self.count_threshold_blue):
        #         cross_traffic_light_flag = True
        #         self.count_blue=0
        #         self.count_red=0
        #
        # self.pub_flag.publish(cross_traffic_light_flag)
        # self.result_msg = self.bridge.cv2_to_compressed_imgmsg(infer_result[0].plot())
        # self.pub_img.publish(self.result_msg)



    def _judge_singal(self) -> str:
        judge_from_yolo = self._yolo()

        print("yolo result:",judge_from_yolo)
        # if(judge_from_yolo == 'not_working'):
        #     judge_from_position_of_yellow_pixel = self._pixel_judge()



        # if(not self._contain_blue and not self._contain_red):
        #     judge_from_position_of_yellow_pixel = self._pixel_judge()
        #
        #
        return "Red"

    def _run(self, _):
        cross_traffic_light_flag = False
        ### if there is no execution flag, initialize the count
        if(not self._exec_flag):
            self._count_blue=0
            self._count_red=0
        ### publish flag if a blue is detected above a threshold value after a red is detected above a threshold value
        else:
            signal = self._judge_singal()
            if(signal == "red"):
                self._count_red += 1
            elif(signal == "blue" and self.count_red > self._count_threshold_red):
                self._count_blue += 1

            if(self._count_blue > self._count_threshold_blue):
                cross_traffic_light_flag = True
                self._count_red = 0
                self._count_blue = 0
        self._pub_flag.publish(cross_traffic_light_flag)



            #
            #
            # infer_result = self.model(self._input_cvimg, classes=[15, 16], conf=self.conf_threshold_blue)
            # # infer_result[0].boxes : class 'ultralytics.engine.results.Boxes'
            # for box in infer_result[0].boxes:
            #     if(int(box.cls.item()) == 16 and box.conf > self.conf_threshold_blue):
            #         contain_blue = True
            #     elif(int(box.cls.item()) == 15 and box.conf > self.conf_threshold_red):
            #         contain_red = True
            #     # if there is a single bounding box, it is cut from the image
            #     if(contain_blue + contain_red == 1):
            #         box_array = box.xyxy
            #         box_numpy = box_array.to('cpu').detach().numpy().astype(int)
            #         x1, y1, x2, y2 = box_numpy[0]
            #         box_img = cv_image[y1:y2, x1:x2]
            #         # output_box = self.bridge.cv2_to_compressed_imgmsg(box_img)
            #         # self.pub_box.publish(output_box)
            #         infer_box = self.model(box_img, classes=[15, 16], conf=self.conf_threshold_blue)
            #         output_box = self.bridge.cv2_to_compressed_imgmsg(infer_box[0].plot())
            #         self.pub_box.publish(output_box)
            #
            # if(not contain_blue):
            #     self.count_red += contain_red
            #
            # if(not contain_red and self.count_red > self.count_threshold_red):
            #     self.count_blue += contain_blue
            #     if(self.count_blue > self.count_threshold_blue):
            #         cross_traffic_light_flag = True
            #         self.count_blue=0
            #         self.count_red=0
            #
            # self.pub_flag.publish(cross_traffic_light_flag)
            # self.result_msg = self.bridge.cv2_to_compressed_imgmsg(infer_result[0].plot())
            # self.pub_img.publish(self.result_msg)
            #

    def __call__(self):
            duration = int(1.0 / self._hz * 1e9)
            rospy.Timer(rospy.Duration(nsecs=duration), self._run)
            rospy.spin()

if __name__=="__main__":
    TrafficlightDetector()()
    rospy.spin()


