import cv2 as cv 
import numpy as npy 
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

#MediaPipe Face Landmarker setup
base_options = python.BaseOptions(
    model_asset_path="face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

#Eye Landmark Indices
left_iris = [469,470,471,472]
right_iris = [474,475,476,477]

left_eye = [33, 160, 158, 133, 153, 144]
right_eye = [362, 385, 387, 263, 373, 380]

#Temporal Buffers
h_vel_buf = deque(maxlen = 90)
v_vel_buf = deque(maxlen = 90)

ied_buf = deque(maxlen = 90)
blink_buf = deque(maxlen=90)

left_spd_buf = deque(maxlen=90)
right_spd_buf = deque(maxlen=90)
left_theta_buf = deque(maxlen=90)
right_theta_buf = deque(maxlen=90)

left_ied_buf = deque(maxlen=90)
right_ied_buf = deque(maxlen=90)

#Tracking variables 
prior_left_center = None
prior_right_center =  None

smooth_left = None
smooth_right = None

alpha = 0.3 #Smoothing factor


#Detection variables
frame_cnt = 0
max_ear = 0.0
ied_cnt = 0
osc_cnt = 0

close_threshold = 0.25 #Inital eye's close threshold value
open_threshold = 0.32 #Inital eye's open threshold value

is_eyes_closed = False

current_status = "Normal"
last_risk_time = 0.0  #Tracks timestamp of the last anomaly
cooldown_dur = 2.0     #Time in seconds required to return to Normal

#Eye Aspect Ratio (EAR) Function
def eye_aspect_ratio(eye_point):
    vert_1 = npy.linalg.norm(eye_point[1]- eye_point[5])
    vert_2 = npy.linalg.norm(eye_point[2]- eye_point[4])
    horiz = npy.linalg.norm(eye_point[0]- eye_point[3])

    ear=(vert_1 + vert_2) / (2 * horiz)

    return ear

def get_iris_center (landmarks, iris_indices, frame_w, frame_h):
     
     center_points = npy.array(
          [(landmarks[i].x * frame_w, landmarks[i].y * frame_h)] 
          for i in iris_indices
          )
     
     return npy.mean(center_points, axis=0) 

def calculate_ied (iris_center, outer_corner, inner_corner):
     eye_axis = outer_corner - inner_corner

     denom = npy.linalg.norm(eye_axis)*2

     if denom == 0:
          return 0.5

     ied = npy.dot(
          iris_center - inner_corner,
          eye_axis
     )/denom 

     return npy.clip(ied, 0.0, 1.0)


video = cv.VideoCapture(0)

#Start of live feed loop
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    #Flip video frame orientation
    frame = cv.flip(frame, 1)

    #Frame Coordinates
    frame_h, frame_w = frame.shape[:2]

    #Reset seizure risk at start
    seizure_risk = 0

    #Convert frame to rgb
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #MediaPipe Detection
    #Process the rgb frame to detect iris landmarks
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
            landmarks = results.face_landmarks[0] #Detect the first face in the frame

            #Iris Centers
            left_center = get_iris_center(
                 landmarks, left_iris, frame_w, frame_h
            )

            right_center = get_iris_center(
                 landmarks, right_iris, frame_w, frame_h
            ) 

            #Smoothing of both left/right eye
            if smooth_left is None:
                 smooth_left = left_center.copy()
            else:
                 smooth_left = (alpha * left_center + (1-alpha) * smooth_left)
            
            if smooth_right is None:
                 smooth_right = right_center.copy()
            else:
                 smooth_right = (alpha * right_center + (1-alpha) * smooth_right)

            #Bilateral Center Calcualtion 
            bilat_center = (smooth_left + smooth_right)/2.0 

            #Draws circles around left/right iris
            cv.circle(frame, tuple(smooth_left.astype(int)), 3, (0,0,255), -1)
            cv.circle(frame, tuple(smooth_right.astype(int)), 3, (0,0,255), -1)
            cv.circle(frame, tuple(bilat_center.astype(int)), 3, (0,0,255), -1)

            #Eye Corners
            left_out = npy.array([landmarks[33].x * frame_w, landmarks[33].y * frame_h])
            left_in = npy.array([landmarks[133].x * frame_w, landmarks[133].y * frame_h])

            right_out = npy.array([landmarks[263].x * frame_w, landmarks[263].y * frame_h])
            right_in = npy.array([landmarks[362].x * frame_w, landmarks[362].y * frame_h])

            #IED Calculation
            left_ied = calculate_ied(smooth_left, left_in, left_out)
            
            right_ied = calculate_ied(smooth_right, right_in, right_out)

            #Bilateral IED Calculation
            bilat_ied = (left_ied + right_ied)/2.0

            ied_buf.append(bilat_ied)

            left_ied_buf.append(left_ied)
            right_ied_buf.append(right_ied)

            #Velocity Calculation
            noise = 0.3 #Added noise factor

            if (prior_left_center is not None and prior_right_center is not None):

                 #Calculate each eye's velocity
                 left_h_vel = smooth_left[0] - prior_left_center[0]
                 left_v_vel = smooth_left[1] - prior_left_center[1]

                 left_spd = npy.hypot(left_h_vel, left_v_vel)
                 left_spd_buf.append(left_spd)

                 left_theta = npy.atan2(left_v_vel, left_h_vel)
                 left_theta_buf.append(left_theta)

                 right_h_vel = smooth_right[0] - prior_right_center[0]
                 right_v_vel = smooth_right[1] - prior_right_center[1]

                 right_spd = npy.hypot(right_h_vel, right_v_vel)
                 right_spd_buf.append(right_spd)

                 right_theta = npy.atan2(right_v_vel, right_h_vel)
                 right_theta_buf.append(right_theta)

                 #Bilateral Velocities Calculation
                 bilat_h_vel = (left_h_vel + right_h_vel)/2.0
                 bilat_v_vel = (left_v_vel + right_v_vel)/2.0

                 #Remove small noise
                 if abs(bilat_h_vel) <= noise:
                      bilat_h_vel = 0.0

                 if abs(bilat_v_vel) <= noise:
                      bilat_v_vel = 0.0

                 #Store bilateral velocities
                 h_vel_buf.append(bilat_h_vel)
                 v_vel_buf.append(bilat_v_vel)

            #Updates current position with prior
            prior_left_center = smooth_left.copy() 
            prior_right_center = smooth_right.copy()

            #EAR Calculation
            #Captures the coordinates of left/right eye
            LEFT_EYE = npy.array([(landmarks[i].x * frame_w, 
                                   landmarks[i].y * frame_h)
                                for i in left_eye])
            
            RIGHT_EYE = npy.array([(landmarks[i].x * frame_w, 
                                   landmarks[i].y * frame_h)
                                for i in right_eye])
            
            #Utilize EAR function to determine left/right eye aspect ratios
            left_ear = eye_aspect_ratio(LEFT_EYE)
            right_ear = eye_aspect_ratio(RIGHT_EYE)

            #Calculate the bilateral EAR from left/right EAR
            bilat_ear = (left_ear + right_ear) / 2.0

            #Blink Detection
            #No blink detected at start
            blink_detected = 0

            #Calibrate (Frames 0 to 59)
            if frame_cnt < 60:
                 max_ear = max(max_ear, bilat_ear)
                 frame_cnt += 1
                 
                 #Set the thresholds on the final frame of calibration
                 if frame_cnt == 60:
                      close_threshold = max_ear * .70
                      open_threshold = max_ear * .85
                      print(f"Calibration Complete! Close:{close_threshold:.2f}, Open: {open_threshold:.2f}")

            #Running Phase (Frame 60 and beyond)
            else:
                 if not is_eyes_closed:
                      #Eye is open, watching bilateral EAR to drop below close threshold 
                      if bilat_ear < close_threshold:
                        is_eyes_closed = True
                        blink_detected = 1
                 else:
                      #Eye is closed, watching bilateral EAR to go back above open threshold
                      if bilat_ear > open_threshold:
                           is_eyes_closed = False

            #Stores blink detected value in set time frame
            blink_buf.append(blink_detected)

            #Temporal Analysis
            bufs_ready = (len(h_vel_buf) >= 90 and
                          len(v_vel_buf) >= 90 and
                          len(ied_buf) >= 90 and 
                          len(blink_buf) >= 90) 

            #No frequency at start
            bilat_h_fq = bilat_v_fq = blink_fq = 0.0

            #Buffers are ready
            if bufs_ready:
                
                h_change = 0 #No frequency changes at start
                v_change = 0

                for i in range(1, len(h_vel_buf)):

                    prev = h_vel_buf[i-1]
                    cur= h_vel_buf[i]

                    if(prev != 0.0 and
                       cur != 0.0 and
                       prev * cur < 0):

                         h_change += 1

                for i in range(1, len(v_vel_buf)):

                     prev = v_vel_buf[i-1]
                     cur = v_vel_buf[i]

                     if(prev != 0.0 and
                        cur != 0.0 and
                        prev * cur < 0):

                          v_change += 1

                #Frequency Calculation
                win_secs = 90 / 30.0 #Converts each window into seconds

                bilat_h_fq = (h_change / (2.0 * win_secs))
                bilat_v_fq = (v_change / (2.0 * win_secs))
                

                #Current Oscillation
                osc_now = (
                     (2.0 <= bilat_h_fq <= 5.0) or 
                     (2.0 <= bilat_v_fq <= 5.0))

                if osc_now:
                     osc_cnt += 1
                else:
                     osc_cnt = 0 

                #Blink count/frequence      
                blink_cnt = sum(blink_buf)
                blink_fq = blink_cnt / win_secs

                #Bilateral IED result
                bilat_ied_mean = npy.mean(ied_buf)
                
                bilat_ied_var = npy.var(ied_buf)

                #Deviation
                bilat_dev = (bilat_ied_mean > 0.65 or bilat_ied_mean < 0.35)

                stable_ied = (bilat_ied_var < 0.002)

                if bilat_dev and stable_ied:
                    ied_cnt += 1
                else:
                    ied_cnt = 0
                     
                #Risk Log
                oscillation_detected = (
                osc_cnt >= 60
               )

                ied_detected = (
                ied_cnt >= 30
               )

                blink_detected = (
                blink_fq > 2.5
               )
                
                #Seizure Risk Scoring          
                if oscillation_detected:
                    seizure_risk += 2
                    print("Bilateral Oscillation")

                if blink_detected:
                    seizure_risk += 2
                    print("Blink")

                if ied_detected:
                    seizure_risk += 2
                    print("Bilateral IED")

          #Classification
            cur_time = time.time()

          #Immediate Status Escalation based on accumulated risk thresholds
            if seizure_risk >= 4:
                    cur_status = "High Risk"
                    last_risk_time = cur_time  #Reset the cooldown clock
            elif seizure_risk >= 2:
                    cur_status = "Suspicious"
                    last_risk_time = cur_time  #Reset the cooldown clock
            else:
               #Hysteresis Downgrade: Only lower the status if the safety window has cleared
               if (cur_time - last_risk_time > cooldown_dur):
                    cur_status = "Normal"
     
          #Color mapping based on status state
            if cur_status == "High Risk":
               status_color = (0, 0, 255)     #Red
            elif cur_status == "Suspicious":
               status_color = (0, 255, 255)   #Yellow
            else:
               status_color = (0, 255, 0)     #Green

          #Draw the main tracking classifications
            cv.putText(frame, f"STATUS: {cur_status}", (20, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
        
            cv.putText(frame, f"Risk Score: {seizure_risk}", (20, 90), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

          #Output the mathematical frequencies
            if bufs_ready:

               bilat_osc = max(bilat_h_fq, bilat_v_fq)

               cv.putText(frame, f"Bilateral Oscillation: {bilat_osc:.1f} Hz", 
                       (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Blink Frequency: {blink_fq:.1f} Hz", 
                       (20, 160), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Bilateral IED: {bilat_ied_mean:.2f}",
                       (20, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"IED Var: {bilat_ied_var:.4f}",
                       (20, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               '''cv.putText(frame, f"Speed: {left_spd_buf[-1]:.1f}",
                       (20, 280), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)'''
               cv.putText(frame, f"IED Counter: {ied_cnt}", 
                       (20,310), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
               cv.putText(frame, f"OSC Counter: {osc_cnt}",
                       (20,340), cv.FONT_HERSHEY_SIMPLEX, 0.6,(200,200,200), 2)

    #Display the resulting frame
    cv.imshow("Seizure Detection", frame)

    #Break the loop if "Esc" key is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the capture and close all OpenCV windows
video.release()
cv.destroyAllWindows()