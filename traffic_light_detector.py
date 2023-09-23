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

        self.exec_flag = False
        self.count_red:list = [0]*10                          #　Number of frames including red in the last 10 frames
        self.red_count_threshold:int = 7


    def exec_flag_callback(self, msg):
        self.exec_flag = msg.data

    def image_callback(self, msg):

        cross_traffic_light_flag = False
        contain_blue = False
        contain_red = False
        if(not self.exec_flag):
            pass
        else:
            bridge = CvBridge()
            image = bridge.compressed_imgmsg_to_cv2(msg)

            infer_result = self.model(image, classes=[15, 16], conf=0.10)

            for box in infer_result[0].boxes:

                if(int(box.cls.item()) == 16):
                    print("\033[32m###########################\n########SIGNAL BLUE########\n###########################\033[0m")
                    contain_blue = True

                elif(int(box.cls.item()) == 15):
                    print("\033[31m#########################\n#######SIGNAL  RED#######\n#########################\033[0m")
                    contain_red = True

            self.count_red.append(contain_red)
            del self.count_red[0]

            print("COUNT RED", sum(self.count_red))

            if(sum(self.count_red) >= self.red_count_threshold and contain_blue):
                cross_traffic_light_flag = True
            else:
                cross_traffic_light_flag = False

            self.pub_flag.publish(cross_traffic_light_flag)
            result_msg = bridge.cv2_to_imgmsg(infer_result[0].plot(), encoding="passthrough")
            self.pub_img.publish(result_msg)

if __name__=="__main__":
    TrafficlightDetector()
    rospy.spin()
