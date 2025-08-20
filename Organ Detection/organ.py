import cv2 as cv 
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

path = 'C:\\Users\\Death\\OneDrive\\Documents\\'
frame = cv.imread(path) #Reads input image

result = model(frame)

for results in result: 
    for box in results.boxes: 
        x1,y1,x2,y2 = box.xyxy[0].tolist() #Generates bounding boxes
        conf = box.conf[0].item() #Generates confidence leveling   

        if conf > 0.5:
            label = f'Organ: {conf:.2f}'
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),)
            cv.putText(frame, label, (int(x1) + 10, int(y1) - 5), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1) 

cv.imshow('Organ', frame)
cv.waitKey()