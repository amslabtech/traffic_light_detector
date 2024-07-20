# traffic_light_detector
ROS package for traffic light detection with YOLOv8

## Environment
- Ubuntu 20.04
- ROS Noetic
- NVIDIA GPU
- Docker

## Dependencies
- [YOLOv8](https://github.com/ultralytics/ultralytics)

## Install and Build
```bash
git clone https://github.com/amslabtech/traffic_light_detector.git
cd traffic_light_detector
docker compose build
```

- **dataset and weight can be downloaded here**
  - [VIDVIP](https://tetsuakibaba.jp/project/vidvip/)

## How to use
```bash
cd traffic_light_detector
docker compose up
```
or
```bash
roslaunch traffic_light_detector traffic_light_detector.launch
```
