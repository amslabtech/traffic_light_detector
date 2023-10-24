#!/usr/bin/python3
import rospy
import os
import cv2
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch

class TrafficlightDetector:
    def __init__(self):

        ### device setting ###
        torch.cuda.set_device(0)
        ### ros settings ###
        #node
        self.node_name="traffic_light_detector"
        rospy.init_node(self.node_name)
        #subscriber
        image_sub = rospy.Subscriber('/CompressedImage', CompressedImage, self.image_callback)
        exec_flag_sub = rospy.Subscriber('/request_detect_traffic_light', Bool, self.exec_flag_callback)
        #publisher
        self.pub_img = rospy.Publisher('/yolo_result/image_raw/compressed', CompressedImage, queue_size=10)
        self.pub_box = rospy.Publisher('/yolo_box/image_raw/compressed', CompressedImage, queue_size=10)
        self.pub_flag = rospy.Publisher('/cross_traffic_light_flag', Bool, queue_size=1)
        ### basic config ###
        self.bridge = CvBridge()
        self.exec_flag:bool = False
        self.count_blue:int = 0
        self.count_red:int = 0
        self.result_msg = CompressedImage()
        self.result_msg.format = "jpeg"

        weight_list = ["vidvip_yolov8n_2023-05-19.pt", "vidvipo_yolov8x_2023-05-19.pt", "yolov8n.pt"]
        self.model=YOLO(os.path.join( "weights", weight_list[1]))

    def get_param(self):
        self.conf_threshold_blue = rospy.get_param('~conf_threshold_blue', 0.3)
        self.conf_threshold_red = rospy.get_param('~conf_threshold_red', 0.3)
        self.count_threshold_blue = rospy.get_param('~count_threshold_blue', 20)
        self.count_threshold_red = rospy.get_param('~count_threshold_red', 50)

    def exec_flag_callback(self, msg: Bool):
        self.exec_flag = msg.data

    def image_callback(self, msg: CompressedImage):
        cross_traffic_light_flag = False
        contain_blue = False
        contain_red = False
        self.result_msg.header = msg.header
        ### if there is no execution flag, initialize the count and publish the received image as is
        if(not self.exec_flag):
            self.count_blue=0
            self.count_red=0
            self.pub_img.publish(msg)
        ### publish flag if a green light is detected above a threshold value after a red light is detected above a threshold value
        else:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
            ### conf of the inference result to be output on the screen should be the threshold value of the blue light, which is a low value.
            # infer_result : list of class 'ultralytics.yolo.engine.results.Results'. len=1
            infer_result = self.model(cv_image, classes=[15, 16], conf=self.conf_threshold_blue)
            # infer_result[0].boxes : class 'ultralytics.engine.results.Boxes'
            for box in infer_result[0].boxes:
                if(int(box.cls.item()) == 16 and box.conf > self.conf_threshold_blue):
                    contain_blue = True
                elif(int(box.cls.item()) == 15 and box.conf > self.conf_threshold_red):
                    contain_red = True
                # if there is a single bounding box, it is cut from the image
                if(contain_blue + contain_red == 1):
                    box_array = box.xyxy
                    box_numpy = box_array.to('cpu').detach().numpy().astype(int)
                    x1, y1, x2, y2 = box_numpy[0]
                    box_img = cv_image[y1:y2, x1:x2]
                    # output_box = self.bridge.cv2_to_compressed_imgmsg(box_img)
                    # self.pub_box.publish(output_box)
                    infer_box = self.model(box_img, classes=[15, 16], conf=self.conf_threshold_blue)
                    output_box = self.bridge.cv2_to_compressed_imgmsg(infer_box[0].plot())
                    self.pub_box.publish(output_box)

            if(not contain_blue):
                self.count_red += contain_red

            if(not contain_red and self.count_red > self.count_threshold_red):
                self.count_blue += contain_blue
                if(self.count_blue > self.count_threshold_blue):
                    cross_traffic_light_flag = True
                    self.count_blue=0
                    self.count_red=0

            self.pub_flag.publish(cross_traffic_light_flag)
            self.result_msg = self.bridge.cv2_to_compressed_imgmsg(infer_result[0].plot())
            self.pub_img.publish(self.result_msg)

if __name__=="__main__":
    traffic_right_detector=TrafficlightDetector()

    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        traffic_right_detector.get_param()
        rospy.spin()
        rate.sleep()

