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
        self.pub_img = rospy.Publisher('/yolo_result', Image, queue_size=10)
        self.pub_flag = rospy.Publisher('/signal_blue', Bool, queue_size=1)

        self.count_red = 0

    def image_callback(self, msg):
        bridge = CvBridge()
        image = bridge.compressed_imgmsg_to_cv2(msg)

        infer_result = self.model(image, classes=[15, 16], conf=0.10)

        for box in infer_result[0].boxes:
            if(int(box.cls.item()) == 15):
                print("\033[31m#########################\n#######SIGNAL  RED#######\n#########################\033[0m")
                self.count_red += 1

            elif(int(box.cls.item()) == 16):
                print("\033[32m###########################\n########SIGNAL BLUE########\n###########################\033[0m")
                if(self.count_red > 5):
                    self.pub_flag.publish(True)
                    print("GOGOGOGOGOGOGOGOGOGOOGOGOGOGOGOGOGO")
                    print("GOGOGOGOGOGOGOGOGOGOOGOGOGOGOGOGOGO")
                    print("GOGOGOGOGOGOGOGOGOGOOGOGOGOGOGOGOGO")
                    print("GOGOGOGOGOGOGOGOGOGOOGOGOGOGOGOGOGO")

            else:
                self.count_red = 0

        result_msg = bridge.cv2_to_imgmsg(infer_result[0].plot(), encoding="passthrough")
        self.pub_img.publish(result_msg)

if __name__=="__main__":
    TrafficlightDetector()
    rospy.spin()
