#!/bin/bash
import os
import urllib.request
import zipfile


data_dir = os.path.join(os.getcwd(), "dataset")
print(data_dir)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
    url = 'https://www.dropbox.com/sh/3moys2qgtgzm8ct/AACtyi8hkM3eennV6LnN1aD0a/vidvipo_full_2023_05_27.zip?dl=1'
    print("DOWNLOADIN ZIP")
    save_path = os.path.join(data_dir, 'dataset.zip')
    urllib.request.urlretrieve(url, save_path)
    print("UNZIP")
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(save_path)
    print("SUCSESSFURRY DOWNLOADED")

