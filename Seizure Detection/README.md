# Seizure Detection  
A Python-based eye tracking program that uses OpenCV and MediaPipe Face Mesh. It detects also differentiates between a person have a potential seizure rather than a seizure 
by the iris movement live feed. The program bases the movement off of two speed thresholds which determine slight/fast movement.

## Features
* Real-time iris tracking
* Iris landmark detection using MediaPipe
* Calculates iris movement off of center iris point and prior iris position
* Live visual
* Easy to modify

## Languages and Libraries
* Python
* OpenCV
* MediaPipe
* Numpy

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/CompDesigner/CV-Projects.git
cd CV-Projects
```
### 2) Install dependencies
```bash
pip install opencv-python mediapipe numpy
```
### 3) Run script
```bash
python "Seizure Detection/seizure_eye_detection.py"
```
Press `ESC` to exit the application

## How it works  
The program:  
1. Creates slight/fast thresholds
2. Captures frames using OpenCV
3. Uses MediaPipe Face Mesh to detect facial landmarks
4. Tracks iris motion
5. Calculates:
   - Iris center point
   - Movement of the iris based on prior positioning verse center point
6. Classifies motion:
   - if no movement (none)
   - if movement is less than slight threshold (slight)
   - if movment is higher than fast threshold (fast)
7. Classifies Seizure and Potential Seizure labeling based on motion

## Output
Is displayed in the left hand corner of the frame showing Seizure or Potential Seizure

## Structure
```text
Seizure Detection/
|
|- seizure_eye_detection.py
|- README.md
|- LICENSE
```

## Requirements
* Python 3.9+
* Webcam/Camera Sensor
* Decent lighting conditions

## Additional Info
MediaPipe Face Mesh landmark indices are used for:
* Iris center
* Eye corners
* Upper/lower eyelids

Movement is currently based on thresholds  
Updates prior iris positioning  

## License
MIT License

## Acknowledgements
Special contributions to the project:
* Google MediaPipe
* OpenCV community
