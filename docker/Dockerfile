FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

## Timezone
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8

##### UTC #####
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
########## non-interactive ##########
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y vim \
	wget \
	unzip \
	git \
	build-essential \
    tmux

RUN apt-get update && \
	apt-get install -y python3-pip \
	python3-tk \
    libsm6 \
	libxrender1 \
	libxext-dev \
    libgl1-mesa-dev \
    libopencv-dev \
    sharutils \
    iproute2 \
    iputils-ping \
    libeigen3-dev

RUN apt install -y libmuparser-dev

RUN python3 -m pip install -U --force-reinstall pip

## Pytorch Setup
RUN pip3 install torch \
                torchvision \
                torchvideo \
                'git+https://github.com/facebookresearch/fvcore' \
                simplejson \
                einops \
                timm \
                psutil \
                scikit-learn \
                opencv-python \
                tensorboard \
                torchsummary \
                tqdm \
                matplotlib \
                pyyaml \
                ttach \
                grad-cam==1.2.9 \
                tensorboardX
RUN pip install ultralytics

######### ROS Noetic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && apt-get install -y lsb-release &&\
	sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
	apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
	apt-get update && apt-get install -y ros-noetic-desktop-full
########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc
## cmk
RUN echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n cd /home/ros_catkin_ws \n catkin_make \$@ \n cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc
########## dnn_attitude_estimation ##########
##### NO cache #####
ARG CACHEBUST=1
###IMU
RUN cd /home && \
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=10y-YuB2QB48mQqsSwTpSID-qK7Hrgty5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10y-YuB2QB48mQqsSwTpSID-qK7Hrgty5" -O /home/MT_Software_Suite_linux-x86_2021.4.tar.gz && rm -rf /tmp/cookies.txt

# RUN cd /home && \
#     tar -xvf MT_Software_Suite_linux-x86_2021.4.tar.gz && \
#     cd MT_Software_Suite_linux-x86_2021.4 && \
#     yes " " | ./mtsdk_linux-x86_2021.4.sh && \
#     cp -r /usr/local/xsens/xsens_ros_mti_driver/ /home/ros_catkin_ws/src/ && \
#     cd /home/ros_catkin_ws && \
#     /bin/bash -c "pushd src/xsens_ros_mti_driver/lib/xspublic; make; popd;"

RUN cd /home/ros_catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make"

### Realsemse
# RUN apt-get update && apt-get -y upgrade
# RUN apt-get install -y software-properties-common && \
#     apt-key adv --keyserver keyserver.ubuntu.com \
#             --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || \
#     apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 \
#             --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
#     add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# RUN apt-get install -y librealsense2-dkms \
#                    librealsense2-utils \
#                    librealsense2-dev \
#                    librealsense2-dbg && \
#     apt-get install -y ros-noetic-realsense2-camera


# RUN mkdir -p /home/log && \
#     mkdir -p /home/log/train_log && \
#     mkdir -p /home/log/inference_test_log && \
#     mkdir -p /home/log/other_test_log && \
#     mkdir -p /home/bagfiles

RUN cd /home/ros_catkin_ws/src && \
    git clone -b yolov8n https://github.com/YoshikiMaekawa2000/traffic_light_detector.git   && \
    cd /home/ros_catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make; source devel/setup.bash"

WORKDIR /home/ros_catkin_ws/
