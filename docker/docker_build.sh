#!/bin/bash

image_name='signal_detector'
image_tag='noetic'

docker build -t $image_name:$image_tag .