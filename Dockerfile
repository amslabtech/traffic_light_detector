ARG ROS_DISTRO=noetic
FROM ros:${ROS_DISTRO}-ros-base
ARG WORKDIR=/root/ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-catkin-tools \
    ros-noetic-cv-bridge
RUN pip3 install ultralytics
RUN rm -rf /etc/apt/apt.conf.d/docker-clean

RUN mkdir -p ${WORKDIR}/src
WORKDIR ${WORKDIR}
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin build"
