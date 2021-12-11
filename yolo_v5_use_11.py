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
# model = torch.hub.load("yolov5", "custom", path=weights_path, source="local")
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

            centroid_dict[objectID] = (int(cx),int(cy),int(x),int(y),int(w),int(h))
            objectID += 1

            red_zone_list = [] #list contains object ids that do distance violation (threshold distance)
            red_line_list = []
            for (id1,p1),(id2,p2) in combinations(centroid_dict.items(),2):
                dx, dy = p1[0]-p2[0], p1[1]-p2[1]
                distance = is_close(dx,dy) # shortest distance (Eculadian distance)
                if distance < 60.0: #Making a threshold for social distancing
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)
                        red_line_list.append(p1[:2])
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
                        red_line_list.append(p2[:2])

                for idx,box in centroid_dict.items():
                    if idx in red_zone_list:
                        cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 1)
                        # cv2.circle(frame, (box[0], box[1]), 2, (0, 0, 255), 5, 1)
                    else:
                        cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 1)
                        # cv2.circle(frame, (box[0], box[1]), 2, (0, 255, 0), 5, 1)

        #Social distance volition counter
        text = f'People at risk of Covid19: {int(len(red_zone_list))}'
        location = (10,30)
        cv2.putText(frame, text, location,2,1,(255,0,0))

        #connect the distance voilaters
        for check in range(0,len(red_zone_list)-1):
            start_pt = red_line_list[check]
            end_pt = red_line_list[check+1]
            check_line_x = abs(end_pt[0]-start_pt[0])
            check_line_y = abs(end_pt[1]-start_pt[1])
            if (check_line_x<50) and (check_line_y<25):
                cv2.line(frame, start_pt,end_pt,(0,0,255))
            ### added feature if comes in perticular distance the person is same
            # elif (check_line_x == 50) and (check_line_y == 25):
            #     cv2.line(frame, start_pt, end_pt, (0, 255, 0))

    #show the results
    cv2.imshow('Yolo_V5',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
v_cap.release()
cv2.destroyAllWindows()