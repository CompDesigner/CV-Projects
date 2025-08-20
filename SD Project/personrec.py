import cv2 as cv
import numpy as npy
#import cvzone
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
names = model.names
model.classes = [0] #Detects people only

vid = cv.VideoCapture(0)

#Frame counter
# cnt = 0 

while True: #if True then loop will run
    detected, frame = vid.read()
    # cnt += 1 #Increment frame count
    # if cnt % 2 != 0: #Skip every other frame
    #     continue

    #Checks if video is detected if not breaks out of loop
    if not detected: 
        break

    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    gray = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)

    #Set result to run YOLO11 tracking method to track objects in frames, persisting tracking amoung frames
    # iou = intersection over union (for intersection of detection boxes) 
    result = model.track(frame, persist=True, iou = .3)

    #Person counter
    person_cnt = 0

    #if no detection boxes and no tracking ids for the detection boxes is found 
    if result[0].boxes is not None and result[0].boxes.id is not None:
        #Gets boxes, person ID dependent on class and confidence level
        boxed = result[0].boxes.xyxy.int().cpu().tolist()
        person_id = result[0].boxes.cls.int().cpu().tolist()
        confidence = result[0].boxes.conf.cpu().tolist()

    #Plotting tracks
    for box, person, conf in zip(boxed, person_id, confidence): 
         person_name = names[person] #set person names to the class name (person)
         if person_name == 'person': #if person name is the same as the class name 
            person_cnt += 1 #Increment person counter
            x,y,x1,y1 = box
            cv.rectangle(gray, (x,y), (x1,y1), (0, 0, 255), 2) #Bounding box (detection box)
            cv.putText(gray, f'Person,Conf: {conf:.2f}', (x,y - 10), 1, 1, (0, 0, 255), 1) #Label

    #Label for total count of people detected on frames
    cv.putText(gray, f'Total Count: {person_cnt}', (10, 50), 1, 1, (0, 0, 255), 1)
            
    #implement grayscaling 

    #Displays result
    cv.imshow('Person', gray)
    
    #Break when 'Esc' key pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

vid.release() #Releases video
cv.destroyAllWindows() #Closes windows