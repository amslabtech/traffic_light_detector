mage_name="integrated_attitude_estimator"
tag_name="noetic"
script_dir=$(cd $(dirname $0); pwd)

docker run -it --net=host --gpus=all --env="DISPLAY" --name="traffic_light_detector" $image_name:$tag_name bash
