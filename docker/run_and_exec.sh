#!/bin/bash

image_name='traffic_light_detector'
image_tag='noetic'

# docker build -t $image_name:$image_tag .

# ./docker_build.sh

docker run -idt \
    --ipc=host \
    --network=host \
    --rm \
    --gpus all \
    --name "traffic_light_detector" \
    $image_name:$image_tag \
    bash \
    --login



