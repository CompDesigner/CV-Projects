import cv2 as cv
import numpy as npy
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Indices for face landmarks for pose estimation
left_iris = [471, 472, 469, 470]
right_iris = [476, 477, 474, 475]
left_h_left = [33]
left_h_right = [133]
right_h_left = [362]
right_h_right = [263]
left_v_up = [159]
left_v_down = [145]
right_v_up = [385]
right_v_down = [374]

# Smoothing factor and initial values
alpha = 0.5
prev_yaw = 0.0
prev_pitch = 0.0

def eculidean (pt1, pt2):
    return npy.linalg.norm(pt1-pt2)

def eye_position(iris_center, left, right, up, down):
    h_ratio = eculidean(iris_center, right) / eculidean(left, right)
    v_ratio = eculidean(iris_center, down) / eculidean(up, down)
    return h_ratio, v_ratio

cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            mesh = result.multi_face_landmarks[0]
            landmarks = npy.array([npy.multiply([p.x, p.y], [w, h]).astype(int) for p in mesh.landmark])

        try:
            # Left eye
            left_iris_center = landmarks[left_iris]
            left_center = npy.mean(left_iris_center, axis=0).astype(int)
            left_hor_left = landmarks[left_h_left]
            left_hor_right = landmarks[left_h_right]
            left_ver_up = landmarks[left_v_up]
            left_ver_down = landmarks[left_v_down]

            # Right eye
            right_iris_center = landmarks[right_iris]
            right_center = npy.mean(right_iris_center, axis=0).astype(int)
            right_hor_left = landmarks[right_h_left]
            right_hor_right = landmarks[right_h_right]
            right_ver_up = landmarks[right_v_up]
            right_ver_down = landmarks[right_v_down]

            # Calculate eye positions
            left_hor_ratio, left_ver_ratio = eye_position(left_center, left_hor_left, left_hor_right, left_ver_up, left_ver_down)
            right_hor_ratio, right_ver_ratio = eye_position(right_center, right_hor_left, right_hor_right, right_ver_up, right_ver_down)

            hor_ratio = (left_hor_ratio + right_hor_ratio) /2
            ver_ratio = (left_ver_ratio + right_ver_ratio) /2

            # Calculate yaw & pitch
            yaw = (0.5 - hor_ratio) * 60
            pitch = (0.5 - ver_ratio) * 60

            smoothed_yaw = alpha * yaw + (1 - alpha) * prev_yaw
            smoothed_pitch = alpha * pitch + (1 - alpha) * prev_pitch
            prev_yaw, prev_pitch = smoothed_yaw, smoothed_pitch

            print(f"Yaw: {smoothed_yaw:.2f}, Pitch: {smoothed_pitch:.2f}")

            # Implement hover control (blink/close eyes)

            # Visualization
            cv.circle(frame, tuple(left_center), 3, (0, 255, 255), -1)
            cv.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

            cv.putText(frame, f"Yaw: {smoothed_yaw:.2f}, Pitch: {smoothed_pitch:.2f}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except IndexError:
            print("Missing eye landmarks")


        cv.imshow("Frame", frame)
        if cv.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()
