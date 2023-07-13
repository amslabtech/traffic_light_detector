image_name="traffic_light_detector"
tag_name="noetic"

docker run -it \
    --network=host \
    --gpus all \
    --name "traffic_light_detector" \
    $image_name:$tag_name \
    bash
