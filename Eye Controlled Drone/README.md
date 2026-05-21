# Eye Controller

A simply Python-based eye tracking controller using OpenCV and MediaPipe Face Mesh. This project detects the iris movement from a webcam feed then estimates the yaw and pitch values based on eye position.  

Potential Usage:  
* Eye-controller interfaces  
* Cursor control interface
* Accesibility tools
* Gaze Tracking

# Features 
* Real-time webcam eye tracking
* Iris landmark detection using MediaPipe
* Horizontial and vertical eye movement estimation
* Smoothed yaw/pitch calculations
* Live visualization overlay
* Easy to modify

# Languages and Libraries  
* Python
* OpenCV
* MediaPipe
* Numpy

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CompDesigner/eye-controller.git
cd eye-controller
```
### 2. Install dependencies  
```bash
pip install opencv-python mediapipe numpy
```
### 3. Run script  
```bash
python eye_controller.py
```
Press `ESC` to exit the application. 
## How It Works 
The program:   
1.  Captures webcam frames using OpenCV
2.  Uses MediaPipe Face Mesh to detecct facial landmarks
3.  Tracks iris positions
4.  Calculates:
    - Horizontal eye ratio
    - Vertical eye ratio
5.  Converts those ratios into:
    - Yaw (left/right movement)
    - Pitch (up/down movement)
6.  Applies smoothing for stable motion values
## Output  
```text
Yaw: -12.48, Pitch: 4.62
Yaw: -10.82, Pitch: 3.76
```
## Structure  
```text
eye_controller/
|
|- eye_controller.py
|- README.md
```
## Requirements  
* Python 3.9+
* Webcam/Camera Sensor
* Good lighting conditions
## Additional Info  
MediaPipe Face Mesh landmark indices are used for:  
* Iris centers
* Eye corners
* Upper/lower eyelids
Tracking accuracy may vary depending on:
* Camera quality
* Lighting
* Distance from camera
## License  
MIT Lincense  
## Acknowledgments

This project would not have been possible without the following open-source contributions:

- **Google MediaPipe** – for providing a robust framework for real-time perception and tracking solutions.
- **OpenCV community** – for the extensive computer vision library and ecosystem used throughout this project.
