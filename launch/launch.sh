#!/bin/bash

source /opt/ros/noetic/setup.bash
source /home/amsl/catkin_ws/devel/setup.bash
export ROS_WORKSPACE=/home/amsl/catkin_ws
export ROS_PACKAGE_PATH=/home/amsl/catkin_ws/src:$ROS_PACKAGE_PATH

roslaunch traffic_light_detector.launch
