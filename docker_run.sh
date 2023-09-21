image_name="traffic_light_detector"
tag_name="noetic"

docker run -it \
    --ipc=host \
    --network=host \
    --rm \
    --gpus all \
    --name "traffic_light_detector" \
    $image_name:$tag_name \
    bash \
    --login
