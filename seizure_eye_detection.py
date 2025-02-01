import cv2 as cv 
import mediapipe as mp
import numpy as npy 

#Setup mapping for iris
face_mapping = mp.solutions.face_mesh 
iris_mapping = face_mapping.FaceMesh(max_num_faces = 1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

#Define slight/fast thresholds
slight_threshold = .25
fast_threshold = 2.0

video = cv.VideoCapture(0)

#Capture prior left/right iris positions 
prior_left = None
prior_right =  None

while True:
    ret, frame = video.read()
    if not ret:
        break

    #Convert frame to rgb
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #Process the rgb frame to detect iris landmarks
    results = iris_mapping.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #Captures the coordinates of left/right iris
            left_iris = npy.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                                  for i in range(474, 478)])
            right_iris = npy.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y)
                                   for i in range(469, 473)])
            
            #Calculates the center for the left/right iris
            left_center = npy.mean(left_iris, axis=0) * [frame.shape[1], frame.shape[0]]
            right_center = npy.mean(right_iris, axis=0) * [frame.shape[1], frame.shape[0]]

            #Draws circles around left/right iris
            cv.circle(frame, tuple(left_center.astype(int)), 2, (0,0,255), -1)
            cv.circle(frame, tuple(right_center.astype(int)), 2, (0,0,255), -1)

            #Calculate movement of left/right iris
            if prior_left is not None and prior_right is not None:
                left_iris_movement = npy.linalg.norm(left_center - prior_left)
                right_iris_movement = npy.linalg.norm(right_center - prior_right)

                 #Classify no left/right iris movement
                left_movement = "none"
                right_movement = "none"

                #Classify left iris movement
                if left_iris_movement < slight_threshold:
                    left_movement = "slight"
                elif left_iris_movement > fast_threshold:
                    left_movement = "fast"
                
                #Classify right iris movement
                if right_iris_movement < slight_threshold:
                    right_movement = "slight"
                elif right_iris_movement > fast_threshold:
                    right_movement = "fast"

                #Determines if person will have a seizure
                if left_movement == "slight" and right_movement == "slight":
                    cv.putText(frame, f"Potential Seizure", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
                elif left_movement == "fast" and right_movement == "fast":
                    cv.putText(frame, f"Seizure", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

            #Update the prior left/right iris position
            prior_left = left_center
            prior_right = right_center


    #Display the resulting frame
    cv.imshow("Seizure Detection", frame)

    #Break the loop if "Esc" key is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the capture and close all OpenCV windows
video.release()
cv.destroyAllWindows()