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
left_vel_h_buf = deque(maxlen=90)
left_vel_v_buf = deque(maxlen=90)
right_vel_h_buf = deque(maxlen=90)
right_vel_v_buf = deque(maxlen=90)
left_spd_buf = deque(maxlen=90)
right_spd_buf = deque(maxlen=90)
left_theta_buf = deque(maxlen=90)
right_theta_buf = deque(maxlen=90)
left_ied_buf = deque(maxlen=90)
right_ied_buf = deque(maxlen=90)
blink_buf = deque(maxlen=90)

#Tracking variables 
prior_left_center = None
prior_right_center =  None

smooth_left = None
smooth_right = None

alpha = 0.3 #Smoothing factor

frame_cnt = 0
max_ear = 0.0
ied_cnt = 0
osc_cnt = 0

close_threshold = 0.25 
open_threshold = 0.32

is_eyes_closed = False
was_eyes_closed = False

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

video = cv.VideoCapture(0)

#Start of live feed loop
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    #Flip video frame orientation
    frame = cv.flip(frame, 1)

    #Coordinates of the frame
    frame_h, frame_w = frame.shape[:2]

    #No seizure risk at start
    seizure_risk = 0

    #Convert frame to rgb
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #Process the rgb frame to detect iris landmarks
    results = face_landmarker.detect(mp_image)

    if results.face_landmarks:
            landmarks = results.face_landmarks[0] #Detect the first face in the frame

            #Captures the coordinates of left/right iris
            LEFT_IRIS = npy.array([(landmarks[i].x * frame_w,
                                     landmarks[i].y * frame_h)
                                  for i in left_iris])
            
            RIGHT_IRIS = npy.array([(landmarks[i].x * frame_w, 
                                     landmarks[i].y * frame_h)
                                   for i in right_iris])
            
            #Calculates the center for the left/right iris
            left_center = npy.mean(LEFT_IRIS, axis=0)
            right_center = npy.mean(RIGHT_IRIS, axis=0)

            #Smoothing of both left/right eye
            if smooth_left is None:
                 smooth_left = left_center
            else:
                 smooth_left = alpha * left_center + (1-alpha) * smooth_left
            
            if smooth_right is None:
                 smooth_right = right_center
            else:
                 smooth_right = alpha * right_center + (1-alpha) * smooth_right

            #Draws circles around left/right iris
            cv.circle(frame, tuple(smooth_left.astype(int)), 3, (0,0,255), -1)
            cv.circle(frame, tuple(smooth_right.astype(int)), 3, (0,0,255), -1)

            #IED Calculation
            left_out = npy.array([landmarks[33].x * frame_w, landmarks[33].y * frame_h])
            left_in = npy.array([landmarks[133].x * frame_w, landmarks[133].y * frame_h])

            right_out = npy.array([landmarks[263].x * frame_w, landmarks[263].y * frame_h])
            right_in = npy.array([landmarks[362].x * frame_w, landmarks[362].y * frame_h])

            cv.circle(frame, tuple(left_center.astype(int)), 5, (0,255,0), -1)
            cv.circle(frame, tuple(left_in.astype(int)), 5, (255,0,0), -1)
            cv.circle(frame, tuple(left_out.astype(int)), 5, (0,0,255), -1)

            cv.circle(frame, tuple(right_center.astype(int)), 5, (0,255,0), -1)
            cv.circle(frame, tuple(right_in.astype(int)), 5, (255,0,0), -1)
            cv.circle(frame, tuple(right_out.astype(int)), 5, (0,0,255), -1)

            left_ied = npy.dot(
                 smooth_left-left_in,
                 left_out-left_in
               ) / npy.linalg.norm(left_out-left_in)**2
            
            right_ied = npy.dot(
                 smooth_right-right_in,
                 right_out-right_in
               ) / npy.linalg.norm(right_out-right_in)**2

            left_ied = npy.clip(left_ied, 0.0, 1.0)
            right_ied = npy.clip(right_ied, 0.0, 1.0)

            #Velocity Calculation
            noise_dz = 0.3 #Added noise factor

            if prior_left_center is not None:
                 left_vel_h = smooth_left[0] - prior_left_center[0]
                 left_vel_v = smooth_left[1] - prior_left_center[1]

                 left_spd = npy.hypot(left_vel_h, left_vel_v)
                 left_spd_buf.append(left_spd)

                 left_theta = npy.atan2(left_vel_v, left_vel_h)
                 left_theta_buf.append(left_theta)
                 left_ied_buf.append(left_ied)

                 left_vel_h_buf.append(left_vel_h if abs(left_vel_h) > noise_dz else 0.0)
                 left_vel_v_buf.append(left_vel_v if abs(left_vel_v) > noise_dz else 0.0)

            if prior_right_center is not None:
                 right_vel_h = smooth_right[0] - prior_right_center[0]
                 right_vel_v = smooth_right[1] - prior_right_center[1]

                 right_spd = npy.hypot(right_vel_h, right_vel_v)
                 right_spd_buf.append(right_spd)

                 right_theta = npy.atan2(right_vel_v, right_vel_h)
                 right_theta_buf.append(right_theta)
                 right_ied_buf.append(right_ied)

                 right_vel_h_buf.append(right_vel_h if abs(right_vel_h) > noise_dz else 0.0)
                 right_vel_v_buf.append(right_vel_v if abs(right_vel_v) > noise_dz else 0.0)

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

            #Calculate the average eye aspect ratio from left/right EAR
            avg_ear = (left_ear + right_ear) / 2.0

            #Blink Detection
            #No blink detected at start
            blink_detected = 0

            #Calibration Phase (Frames 0 to 59)
            if frame_cnt < 60:
                 max_ear = max(max_ear, avg_ear)
                 frame_cnt += 1
                 
                 #Set the thresholds on the final frame of calibration
                 if frame_cnt == 60:
                      close_threshold = max_ear * .70
                      open_threshold = max_ear * .85
                      print(f"Calibration Complete! Close:{close_threshold:.2f}, Open: {open_threshold:.2f}")

            #Running Phase (Frame 60 and beyond)
            else:
                 if not is_eyes_closed:
                      #Eye is open, watching average EAR to drop below close threshold 
                      if avg_ear < close_threshold:
                        is_eyes_closed = True
                        blink_detected = 1
                 else:
                      #Eye is closed, watching average EAR to go back above open threshold
                      if avg_ear > open_threshold:
                           is_eyes_closed = False

            #Set the prior value to be current value
            was_eyes_closed = is_eyes_closed

            #Get blink detected value in set time frame
            blink_buf.append(blink_detected)

            #Temporal Analysis
            bufs_ready = (len(left_vel_h_buf) >= 90 and
                          len(left_vel_v_buf) >= 90 and
                          len(right_vel_h_buf) >= 90 and
                          len(right_vel_v_buf) >= 90 and
                          len(left_ied_buf) >= 90 and 
                          len(right_ied_buf) >= 90 and
                          len(blink_buf) >= 90) 

            #No frequency at start
            left_h_fq = left_v_fq = right_h_fq = right_v_fq = blink_fq = 0.0

            #Buffers are ready
            if bufs_ready:
                left_h_change = 0 #No frequency changes at start
                left_v_change = 0
                right_h_change = 0
                right_v_change = 0

                for i in range(1, len(left_vel_h_buf)):
                    if left_vel_h_buf[i] != 0.0 and left_vel_h_buf[i-1] != 0.0:
                         if (left_vel_h_buf[i] * left_vel_h_buf[i-1]) < 0: left_h_change += 1

                    if left_vel_v_buf[i] != 0.0 and left_vel_v_buf[i-1] != 0.0:
                         if (left_vel_v_buf[i] * left_vel_v_buf[i-1]) < 0: left_v_change += 1

                    if right_vel_h_buf[i] != 0.0 and right_vel_h_buf[i-1] != 0.0:
                         if (right_vel_h_buf[i] * right_vel_h_buf[i-1]) < 0: right_h_change += 1

                    if right_vel_v_buf[i] != 0.0 and right_vel_v_buf[i-1] != 0.0:
                         if (right_vel_v_buf[i] * right_vel_v_buf[i-1]) < 0: right_v_change += 1
                    
                win_secs = 90 / 30.0 #Converts each window into seconds 

                #Frequency calculation
                left_h_fq = left_h_change / (2.0 * win_secs)
                left_v_fq = left_v_change / (2.0 * win_secs)
                right_h_fq = right_h_change / (2.0 * win_secs)
                right_v_fq = right_v_change / (2.0 * win_secs)

                #Current oscillation
                osc_now = (
                     (2.0 <= left_h_fq <= 5.0) or 
                     (2.0 <= left_v_fq <= 5.0) or 
                     (2.0 <= right_h_fq <= 5.0) or 
                     (2.0 <= right_v_fq <= 5.0)
                )

                if osc_now:
                     osc_cnt += 1
                else:
                     osc_cnt = 0 

                #Blink count/frequence      
                blink_cnt = sum(blink_buf)
                blink_fq = blink_cnt / win_secs

                #IED calculation
                left_ied_mean = npy.mean(left_ied_buf)
                right_ied_mean = npy.mean(right_ied_buf)

                #IED variance
                left_ied_var = npy.var(left_ied_buf)
                right_ied_var = npy.var(right_ied_buf)

                #Deviation
                left_dev = (left_ied_mean > 0.65 or left_ied_mean < 0.35)
                right_dev = (right_ied_mean > 0.65 or right_ied_mean < 0.35)

                stable_ied = (
                    left_ied_var < 0.002 and
                    right_ied_var < 0.002
                )

                if left_dev and right_dev and stable_ied:
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
                          
                if oscillation_detected:
                    seizure_risk += 2
                    print("oscillation")

                if blink_detected:
                    seizure_risk += 2
                    print("blink")

                if ied_detected:
                    seizure_risk += 2
                    print("IED")

          #Classification
            current_time = time.time()

          #Immediate Status Escalation based on accumulated risk thresholds
            if seizure_risk >= 4:
                    current_status = "High Risk"
                    last_risk_time = current_time  #Reset the cooldown clock
            elif seizure_risk >= 2:
                    current_status = "Suspicious"
                    last_risk_time = current_time  #Reset the cooldown clock
            else:
               #Hysteresis Downgrade: Only lower the status if the safety window has cleared
               if current_time - last_risk_time > cooldown_dur:
                    current_status = "Normal"
     
          #Color mapping based on status state
            if current_status == "High Risk":
               status_color = (0, 0, 255)     #Red
            elif current_status == "Suspicious":
               status_color = (0, 255, 255)   #Yellow
            else:
               status_color = (0, 255, 0)     #Green

          #Draw the main tracking classifications
            cv.putText(frame, f"STATUS: {current_status}", (20, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
        
            cv.putText(frame, f"Risk Score: {seizure_risk}", (20, 90), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

          #Output the mathematical frequencies
            if bufs_ready:
               cv.putText(frame, f"Ocular Oscillation: {max(left_h_fq, left_v_fq, right_h_fq, right_v_fq):.1f} Hz", 
                       (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Blink Frequency: {blink_fq:.1f} Hz", 
                       (20, 160), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Left IED: {left_ied_mean:.2f}",
                       (20, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Right IED: {right_ied_mean:.2f}",
                       (20, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"IED Var: {left_ied_var:.4f}",
                       (20, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
               cv.putText(frame, f"Speed: {left_spd_buf[-1]:.1f}",
                       (20, 280), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
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