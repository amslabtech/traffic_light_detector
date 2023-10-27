FROM ultralytics/ultralytics
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

##########  bias ##########
RUN apt-get update && apt-get install -y --no-install-recommends \
    x11-xserver-utils \
    git \
    tmux \
    vim \
    neovim \
    python3 \
    python3-dev \
    python3-pip \
    curl \
    wget \
    tzdata \
    tree \
    nodejs \
    npm \
    lsb-release
######### neovim install #########
RUN cd /root/ && wget https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz
RUN cd /root/ && tar xzvf nvim-linux64.tar.gz
RUN rm /root/nvim-linux64.tar.gz
RUN mv /root/nvim-linux64 /
RUN ln -sf /nvim-linux64/bin/nvim /usr/bin/nvim
######### vim/neovim setup #########
RUN pip3 install jedi-language-server
RUN npm install -g n
RUN n stable
RUN mkdir -p /root/.vim/undo
RUN mkdir -p /root/.vim/plugged
RUN mkdir -p /root/.config/nvim
######### dotfiles #########
RUN git clone https://github.com/ToshikiNakamura0412/dotfiles_for_docker.git /root/dotfiles
ENV DOTFILES="/root/dotfiles"
RUN ln -sf $DOTFILES/profile.sh /etc/profile.d/profile.sh
RUN ln -sf $DOTFILES/vim/vimrc /root/.vimrc
RUN ln -sf $DOTFILES/vim/init.vim /root/.config/nvim/init.vim
RUN ln -sf $DOTFILES/vim/coc-settings.json /root/.config/nvim/coc-settings.json
RUN ln -sf $DOTFILES/tmux.conf /root/.tmux.con
######### ROS Noetic insatall ##########
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    # apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-noetic-ros-base
# ########## ROS setup ##########
RUN mkdir -p /home/amsl/catkin_ws/src && \
    cd /home/amsl/catkin_ws/src && \
    cd /home/amsl/catkin_ws && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /home/amsl/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    echo "export ROS_PACKAGE_PATH=/home/amsl/catkin_ws/src:$ROS_PACKAGE_PATH" >> ~/.bashrc && \
    echo "export ROS_WORKSPACE=/home/amsl/catkin_ws" >> ~/.bashrc && \
    export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
# ENV PATH=$PATH:/opt/ros/noetic/bin/roslaunch
# ## cmk
# # RUN echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n cd /home/catkin_ws \n catkin_make \$@ \n cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc
# # RUN cd /home/catkin_ws/src && \
# RUN git clone https://github.com/YoshikiMaekawa2000/traffic_light_detector.git
#
# # RUN ln -sf /home/catkin_ws/src/traffic_light_detector/launch/launch.sh /etc/profile.d/launch.sh
#
WORKDIR /home/amsl/catkin_ws/src/traffic_light_detector/launch
