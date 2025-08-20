import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.classes = [0] #Detects people only

vid = cv.VideoCapture(0)


while True: #if True then loop will run
    detected, frame = vid.read()

    #Checks if video is detected if not breaks out of loop
    if not detected: 
        break

    result = model.track(frame, persist=True, iou = .3) #Model prediction on the frames from video includes intersection over union
    #result = model(frame, iou = .1) 

    track_ids = set() #Set that stores tracked ids

    for results in result: 
        for box in results.boxes: 
            x1,y1,x2,y2 = box.xyxy[0].tolist() #Generates bounding boxes
            conf = box.conf[0].item() #Generates confidence leveling
            person = int(box.cls[0]) #Boxing based off Class ID 

            track_id = box.id  #Sets track id to detected objects with bounding boxes
            if track_id is not None:  #Counts valid track id
                track_ids.add(track_id)

            #implement grayscaling 

            #Lower conf = increased blurriness
            if person == 0 and conf > 0.50:
                label = f'Person: {conf:.2f}'
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),)
                cv.putText(frame, label, (int(x1) + 10, int(y1) - 5), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
    
    #Sets total amount of people counted by the length of the track ids set(array)
    total_amt= len(track_ids)

    #Displays the total number of detected people
    cv.putText(frame, f'Total: {total_amt}', (10, 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2) 

    cv.imshow('Person', frame)
    
    #Break when 'Esc' key pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

vid.release()
cv.destroyAllWindows()