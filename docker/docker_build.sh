#!/bin/bash

image_name='traffic_light__detector'
image_tag='noetic'

docker build -t $image_name:$image_tag .
