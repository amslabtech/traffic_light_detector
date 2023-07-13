#!/usr/bin/python3
import rospy
import os
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class TrafficlightDetector:
    def __init__(self):
        self.node_name="traffic_light_detector"
        rospy.init_node(self.node_name)

        # self.model=YOLO("yolov8n.pt")
        pwd=os.getcwd()
        self.model=YOLO(os.path.join(pwd[:-3], "weights", "vidvip_yolov8n_2023-05-19.pt"))

        # image_sub = rospy.Subscriber('/CompressedImage', CompressedImage, self.image_callback)
        image_sub = rospy.Subscriber('/grass_cam/image_raw/compressed', CompressedImage, self.image_callback)
        self.pub = rospy.Publisher('/yolo_result', Image, queue_size=10)




    def image_callback(self, msg):
        # 画像データをROSメッセージから復元
        bridge = CvBridge()
        image = bridge.compressed_imgmsg_to_cv2(msg)

        # YOLOで推論
        infer_result = self.model(image)



        # 推論結果をImage型に変換
        result_msg = bridge.cv2_to_imgmsg(infer_result[0].plot(), encoding="passthrough")


        # 推論結果をpublish
        self.pub.publish(result_msg)

if __name__=="__main__":
    TrafficlightDetector()
    rospy.spin()
