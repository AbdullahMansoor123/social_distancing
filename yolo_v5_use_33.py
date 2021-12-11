import math
import cv2
from itertools import combinations
############## Version 33 Turn script to Class ################

class social_distance_detector(object):
    """
    Social Distance Detector
    """
    #resize the results window
    def resize(self,image):
        return cv2.resize(image,(720,640))

    #Calculates the euclidean distance
    def is_close(self,p1, p2):
        dst = math.sqrt(p1**2 + p2**2)
        return dst

    def __init__(self,model):
        self.model = model


    def run(self,test_video):
        #capture video
        global red_zone_list, red_line_list
        v_cap = cv2.VideoCapture(test_video)

        while True:
            success, frame = v_cap.read()
            if not success:
                break

            #make detection
            results = self.model(frame)

            if len(results.xyxy[0]) > 0:
                centroid_dict = dict()
                objectID = 0

                #show bbox around all detected person class
                for person in results.xyxy[0]:
                    x, y, w, h, confidence, label = person
                    cx = int((x + w) / 2)
                    cy = int((y + h) / 2)

                    centroid_dict[objectID] = (cx, cy, x, y, w, h)
                    objectID += 1

                    red_zone_list = [] #list contains object ids that do distance violation (threshold distance)
                    red_line_list = [] #stores centroid of people not following social distancing

                    for (id1,p1),(id2,p2) in combinations(centroid_dict.items(),2):
                        dx, dy = p1[0]-p2[0], p1[1]-p2[1]
                        distance = self.is_close(dx,dy) # shortest distance (Euclidean distance)

                        if distance < 60.0: #Making a condition for social distancing
                            if id1 not in red_zone_list:
                                red_zone_list.append(id1)
                                red_line_list.append(p1[:2])
                            if id2 not in red_zone_list:
                                red_zone_list.append(id2)
                                red_line_list.append(p2[:2])

                        for idx in centroid_dict:
                            x,y = centroid_dict.get(idx)[2], centroid_dict.get(idx)[3]
                            w,h = centroid_dict.get(idx)[4], centroid_dict.get(idx)[5]
                            if idx in red_zone_list:
                                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 1)
                                # cv2.circle(frame, (box[0], box[1]), 2, (0, 0, 255), 5, 1)
                            else:
                                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 1)
                                # cv2.circle(frame, (box[0], box[1]), 2, (0, 255, 0), 5, 1)

                #Social distance volition counter
                text = f'People at risk of Covid19: {int(len(red_zone_list))}'
                location = (10,30)
                cv2.putText(frame, text, location,2,1,(255,0,0))

                #connect the distance voilaters
                for check in range(0,len(red_zone_list)-1):
                    start_pt = red_line_list[check]
                    end_pt = red_line_list[check-1]
                    check_line_x = abs(end_pt[0]-start_pt[0])
                    check_line_y = abs(end_pt[1]-start_pt[1])
                    if (check_line_x < 50) and (check_line_y < 25):
                        cv2.line(frame, start_pt,end_pt,(0,0,255))

                    ### (added feature)turn the line green if the person comes in safe distance
                    # elif (check_line_x == 50) and (check_line_y == 25):
                    #     cv2.line(frame, start_pt, end_pt, (0, 255, 0))

            #show the results
            cv2.imshow('Yolo_V5',self.resize(frame))

            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        v_cap.release()
        cv2.destroyAllWindows()