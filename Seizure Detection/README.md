# Seizure Detection  
A Python-based eye tracking program that uses OpenCV and MediaPipe Face Mesh. It determines the if a person is having a seizure by seizure risk scoring and the risk score is determined by calculated factors. The calculated factors are IED, Eye Aspect Ratio (EAR), excessive blinking and oscillation that determine the risk of having a seizure.

## Features
* Real-time iris tracking
* Iris landmark detection using MediaPipe
* Calculates iris movement off of center iris points and prior iris position
* Live visual
* More realistic seizure detection

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
1. Defines temporal buffers, tracking variables and detection variables
2. Creates three functions: eye_aspect_ratio, get_iris_center and calculate_ied  
3. Uses MediaPipe Face Mesh to detect facial landmarks
4. Utilizes get_iris_center function to determine left/right iris centers
5. Calculates:
   - Bilateral center
   - IED for left, right and bilateral
   - Velocity for left, right and bilateral
   - EAR 
   - Blinking
   - Temporal Analysis
   - Frequency/Oscillation
   - Deviation
6. Classifies based on calculated values:
   - if no/low risk score (Normal)
   - if risk score is greater than or equal to 2 (suspicious/potential seizure)
   - if risk score is greater than or equal to 4 (high risk/seizure)
7. Classifies Normal, High Risk/Seizure and Suspicious/Potential Seizure labeling based on scoring 

## Output
Is displayed in the left hand corner of the frame showing values and risk scoring

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
 
Updates prior iris positioning  

## License
MIT License

## Acknowledgements
Special contributions to the project:
* **Google MediaPipe**
* **OpenCV community**
