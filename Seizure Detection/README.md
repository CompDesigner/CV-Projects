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
3. 
