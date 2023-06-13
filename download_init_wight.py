#!/bin/bash
import os
import urllib.request
import zipfile

weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)
    url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
    target_path = os.path.join(weights_dir, "vgg16_reducedfc.pth") 
    urllib.request.urlretrieve(url, target_path)