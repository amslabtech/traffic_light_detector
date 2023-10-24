FROM ultralytics/ultralytics
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

USER root
##########  bias ##########
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
	rsync \
    sudo \
    wget \
    x11-xserver-utils \
    git \
    tmux \
    vim \
    neovim \
    python3 \
    python3-dev \
    python3-pip \
    cmake \
    build-essential \
    curl \
    wget \
    tzdata \
    tree \
    nodejs \
    npm
# ######### neovim install #########
# RUN cd /root/ && wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz
# RUN cd /root/ && tar xzvf nvim-linux64.tar.gz
# RUN rm /root/nvim-linux64.tar.gz
# RUN mv /root/nvim-linux64 /
# RUN ln -sf /nvim-linux64/bin/nvim /usr/bin/nvim
# ######### vim/neovim setup #########
# RUN pip3 install jedi-language-server
# RUN npm install -g n
# RUN n stable
# RUN mkdir -p /root/.vim/undo
# RUN mkdir -p /root/.vim/plugged
# RUN mkdir -p /root/.config/nvim
# ######### dotfiles #########
# RUN git clone https://github.com/ToshikiNakamura0412/dotfiles_for_docker.git /root/dotfiles
# ENV DOTFILES="/root/dotfiles"
# RUN ln -sf $DOTFILES/profile.sh /etc/profile.d/profile.sh
# RUN ln -sf $DOTFILES/vim/vimrc /root/.vimrc
# RUN ln -sf $DOTFILES/vim/init.vim /root/.config/nvim/init.vim
# RUN ln -sf $DOTFILES/vim/coc-settings.json /root/.config/nvim/coc-settings.json
# RUN ln -sf $DOTFILES/tmux.conf /root/.tmux.con
# ########## nvidia-docker1 hooks ##########
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# ENV PATH /usr/local/nvidia/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# ########## non-interactive ##########
# ENV DEBIAN_FRONTEND=noninteractive
#
# RUN apt-get update && apt-get upgrade -y
#
# RUN apt-get install -y vim \
# 	wget \
# 	unzip \
# 	git \
# 	build-essential \
#     tmux
#
#
# RUN apt-get update && \
# 	apt-get install -y python3-pip \
# 	python3-tk \
#     libsm6 \
# 	libxrender1 \
# 	libxext-dev \
#     libgl1-mesa-dev \
#     libopencv-dev \
#     sharutils \
#     iproute2 \
#     iputils-ping \
#     libeigen3-dev
#
# RUN python3 -m pip install -U --force-reinstall pip
#
# ## Pytorch Setup
# RUN pip install torch \
#                 torchvision \
#                 ultralytics
#
######### ROS Noetic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && apt-get install -y lsb-release &&\
	sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' &&\
	apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 &&\
	apt-get update && apt-get install -y ros-noetic-desktop-full
########## ROS setup ##########
RUN mkdir -p /home/catkin_ws/src && \
	cd /home/catkin_ws/src && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_init_workspace" && \
	cd /home/catkin_ws && \
	/bin/bash -c "source /opt/ros/noetic/setup.bash; catkin_make" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /home/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/catkin_ws" >> ~/.bashrc
ENV PATH=$PATH:/opt/ros/noetic/bin/roslaunch
## cmk
# RUN echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n cd /home/catkin_ws \n catkin_make \$@ \n cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc
# RUN cd /home/catkin_ws/src && \
RUN git clone https://github.com/YoshikiMaekawa2000/traffic_light_detector.git

# RUN ln -sf /home/catkin_ws/src/traffic_light_detector/launch/launch.sh /etc/profile.d/launch.sh

# WORKDIR /home/catkin_ws/src/traffic_light_detector/launch
