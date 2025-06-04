# Smart Fan Automation Using Image Processing

## Overview
This project simulates a smart fan automation system that adjusts fan speed based on the number of people detected in a room using computer vision. It uses the MobileNet-SSD model for real-time person detection and controls a virtual fan with speeds: Off (0 people), Low (1 person), Medium (2 people), and High (3+ people). The system displays live video with bounding boxes around detected individuals, shows the fan speed on-screen, and logs decisions to the console.

## Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Installation
1. Install the required libraries:
   ```bash
   pip install opencv-python numpy
   ```
2. Download the MobileNet-SSD model files:
   - [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt)
   - [MobileNetSSD_deploy.caffemodel](https://drive.google.com/file/d/0B3gersZ2L04xRm5PM2JNUzFfa3c/view)
   Place them in the `models/` directory.

## Usage
Run the simulation with:
```bash
python src/main.py --input 0
```
- `--input`: Specify a video file path or webcam index (default: 0 for webcam).
- `--prototxt`: Path to the prototxt file (default: models/MobileNetSSD_deploy.prototxt).
- `--model`: Path to the caffemodel file (default: models/MobileNetSSD_deploy.caffemodel).

## Design Rationale
- **Detection Model**: MobileNet-SSD was chosen for its balance of speed and accuracy, achieving over 40 FPS on modern CPUs, suitable for real-time applications ([LearnOpenCV](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/)).
- **Modular Structure**: The code is organized into `detector.py`, `fan_controller.py`, and `main.py` for maintainability and scalability, allowing easy integration of future features like temperature sensors.
- **Efficiency**: Non-maximum suppression reduces false positives, and the system is optimized for minimal frame lag.
- **Lighting Adaptability**: MobileNet-SSD's deep learning approach ensures robustness to varying lighting conditions compared to traditional methods like HOG.

## Demo Visuals
Screenshots of the simulation showing bounding boxes and fan speed display are available in the `docs/` directory (to be added after testing).

## Future Extensions
- **Temperature Sensors**: Integrate sensor data to adjust fan speed based on room temperature.
- **Facial Recognition**: Add identity recognition for personalized settings.
- **IoT Integration**: Connect to cloud services for remote monitoring and control.

## Troubleshooting
- Ensure model files are correctly placed in the `models/` directory.
- Verify webcam access or video file path if capture fails.
- Adjust confidence threshold in `detector.py` if detection is too sensitive or misses people.
