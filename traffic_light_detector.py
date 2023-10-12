#!/usr/bin/python3
import rospy
import os
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO

class TrafficlightDetector:
    def __init__(self):
        self.node_name="traffic_light_detector"
        rospy.init_node(self.node_name)

        # self.model=YOLO("yolov8n.pt")
        pwd=os.getcwd()
        self.model=YOLO(os.path.join( "weights", "vidvip_yolov8n_2023-05-19.pt"))

        image_sub = rospy.Subscriber('/CompressedImage', CompressedImage, self.image_callback)
        exec_flag_sub = rospy.Subscriber('/request_detect_traffic_light', Bool, self.exec_flag_callback)
        self.pub_img = rospy.Publisher('/yolo_result', Image, queue_size=10)
        self.pub_flag = rospy.Publisher('/cross_traffic_light_flag', Bool, queue_size=1)

        self.exec_flag:bool = False
        self.count_blue:int = 0
        self.count_red:int = 0

    def check_param(self):
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.1)
        self.blue_count_threshold = rospy.get_param('~blue_count_threshold', 20)
        self.red_count_threshold = rospy.get_param('~red_count_threshold', 50)


    def exec_flag_callback(self, msg):
        self.exec_flag = msg.data

    def image_callback(self, msg):
        cross_traffic_light_flag = False
        contain_blue = False
        contain_red = False

        if(not self.exec_flag):
            self.count_blue=0
            self.count_red=0
        else:
            bridge = CvBridge()
            image = bridge.compressed_imgmsg_to_cv2(msg)
            infer_result = self.model(image, classes=[12, 15, 16], conf=self.conf_threshold)

            for box in infer_result[0].boxes:

                if(int(box.cls.item()) == 16):
                    print("\033[32m###########################\n########SIGNAL BLUE########\n###########################\033[0m")
                    contain_blue = True

                elif(int(box.cls.item()) == 15):
                    print("\033[31m#########################\n#######SIGNAL  RED#######\n#########################\033[0m")
                    contain_red = True

            if(not contain_blue):
                self.count_red += contain_red

            if(not contain_red and self.count_red > self.red_count_threshold):
                self.count_blue += contain_blue
                print("BRUE", self.count_blue)
                if(self.count_blue > self.blue_count_threshold):
                    cross_traffic_light_flag = True
                    self.count_blue=0
                    self.count_red=0

            self.pub_flag.publish(cross_traffic_light_flag)
            result_msg = bridge.cv2_to_imgmsg(infer_result[0].plot(), encoding="passthrough")
            self.pub_img.publish(result_msg)

if __name__=="__main__":
    traffic_right_detector=TrafficlightDetector()

    rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        traffic_right_detector.check_param()
        rospy.spin()
        rate.sleep()
