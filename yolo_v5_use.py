import math
import torch
import numpy as np
import matplotlib.pylab as plt
import cv2
from itertools import combinations

#resize the results window
def resize(image):
    return cv2.resize(image,(640,340))

def is_close(p1, p2):
    dst = math.sqrt(p1**2 + p2**2)
    return dst


# weights_path = 'yolov5/yolov5s.pt'
#load yoloV5 model
# model = torch.hub.load("yolov5", "custom", path=weights_path, source="local", force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
##Incase of error run below command
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s',classes = 1,force_reload=True)
model.classes = [0] #filter out person

#capture video
v_cap = cv2.VideoCapture('people_walking.mp4')
# v_cap = cv2.VideoCapture(0)


while True:
    success, frame = v_cap.read()
    if not success:
        break

    #make detection
    results = model(frame)

    if len(results.xyxy[0]) > 0:
        centroid_dict = dict()
        objectID = 0

        #print bounding box
        for person in results.xyxy[0]:
            x, y, w, h, confidence, label = person
            cx = int((x + w) / 2)
            cy = int((y + h) / 2)

            centroid_dict[objectID] =(int(cx), int(cy), x, y, w, h)
            objectID +=1
    # Determine which person bbox is close to each other
            red_zone_list = []
            red_line_list = []
            for (id1,p1),(id2,p2) in combinations(centroid_dict.items(),2):
                dx, dy = p1[0]-p2[0], p1[1]-p2[1]
                distance = is_close(dx,dy)
                if distance < 75.0:
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)
                        red_line_list.append(p1[0:2])
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
                        red_line_list.append(p2[0:2])
            for idx,box in enumerate(centroid_dict):
                if idx in red_line_list:
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0,0,255),2)
                else:
                    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0,255,0),2)
# Display risk analytics and indicators
    text = f'People at covid risk: {str(len(red_zone_list))}'
    location = (10,25)
    cv2.putText(frame, text,location,5,2,(255,0,0),2)

#### Draw red line when at less then 1m

    for check in range(0, len(red_line_list) - 1):  # Draw line between nearby bboxes iterate through redlist items
        start_point = red_line_list[check]
        end_point = red_line_list[check + 1]
        check_line_x = abs(end_point[0] - start_point[0])  # Calculate the line coordinates for x
        check_line_y = abs(end_point[1] - start_point[1])  # Calculate the line coordinates for y
        if (check_line_x < 75) and (check_line_y < 25):  # If both are We check that the lines are below our threshold distance.
            cv2.line(frame, start_point, end_point, (0, 0, 255),2)


    #show the results
    cv2.imshow('Yolo_V5',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
v_cap.release()
cv2.destroyAllWindows()