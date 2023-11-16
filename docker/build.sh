#!/bin/bash

image_name='traffic_light_detector'
image_tag='emergency'

# docker build -t $image_name:$image_tag --no-cache --network host .
docker build -t $image_name:$image_tag .
