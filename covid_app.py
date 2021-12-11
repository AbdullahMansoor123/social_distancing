from yolo_v5_use_44 import social_distance_detector
import torch
import numpy as np


##initalizing yoloV5 for only person class
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [0]

##Runs the Social Distancing Detector##
# test_video = 0 #web_camq
test_video = 'people_walking.mp4'
covid_app = social_distance_detector(model)
covid_app.run(test_video)

