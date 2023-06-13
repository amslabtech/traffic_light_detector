#!/bin/bash
import os
import urllib.request
import zipfile


data_dir = os.path.join(os.getcwd(), "./dataset/")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    url = 'https://www.dropbox.com/sh/3moys2qgtgzm8ct/AACtyi8hkM3eennV6LnN1aD0a/vidvipo_full_2023_05_27.zip?dl=0'
    urllib.request.urlretrieve(url, data_dir)
    with zipfile.ZipFile(os.path.join(data_dir, 'vidvipo_full_2023_05_27.zip?dl=0'), 'r') as zip_ref:
        zip_ref.extractall(data_dir)


'''
script_dir=$(cd $(dirname $0); pwd)


cd ../.. 
mkdir dataset && cd dataset


wget https://www.dropbox.com/sh/3moys2qgtgzm8ct/AACtyi8hkM3eennV6LnN1aD0a/vidvipo_full_2023_05_27.zip?dl=0
unzip 'vidvipo_full_2023_05_27.zip?dl=0'
'''
