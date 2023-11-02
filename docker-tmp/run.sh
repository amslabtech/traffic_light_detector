image_name="traffic_light_detector"
tag_name="tmp"

docker run \
    --ipc=host \
    --network=host \
    -it \
    --rm \
    --gpus all \
    -v ~/catkin_ws/src/traffic_light_detector:/home/amsl/catkin_ws/src/traffic_light_detector \
    --name "traffic_light_detector" \
    $image_name:$tag_name \
    ./launch.sh
    # bash \
    # --login
