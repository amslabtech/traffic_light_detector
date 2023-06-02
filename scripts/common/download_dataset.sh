#!/bin/bash
script_dir=$(cd $(dirname $0); pwd)


cd ../.. 
mkdir dataset && cd dataset


wget https://www.dropbox.com/sh/3moys2qgtgzm8ct/AACtyi8hkM3eennV6LnN1aD0a/vidvipo_full_2023_05_27.zip?dl=0
unzip 'vidvipo_full_2023_05_27.zip?dl=0'