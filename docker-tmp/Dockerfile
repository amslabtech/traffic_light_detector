FROM traffic_light_detector:v3.3
RUN apt-get update && apt-get install -y ros-noetic-cv-bridge
RUN pip3 install rospkg netifaces
RUN rm -rf /home/amsl/catkin_ws/src/traffic_light_detector
WORKDIR /home/amsl/catkin_ws/src/traffic_light_detector/launch
