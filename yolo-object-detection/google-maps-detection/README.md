# How to download the dataset
To download the google maps datase go go https://app.roboflow.com/telephonepoledetection/telephone-poles/7 SELECT A VERSION AND PRESS DOWNLOAD DATASET

 OR YOU CAN USE ROBOFLOW USING PIP AND DOWNLOAD IT USING THE CLI COMMAND

    pip install roboflow
    roboflow download -f yolov8 -l ./ /telephonepoledetection/telephone-poles/7

# Training
Make sure to specify the datase path in data.yaml file, training and testing can be done using the pole_detection.ipynb notebook