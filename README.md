# Altitude-based Object Detection

This project implements an object detection system that analyzes how object detection changes with varying altitudes using OpenCV and HSV color space filtering.

## Features
- Real-time object detection using OpenCV
- HSV color space filtering for object isolation
- Altitude-based detection analysis
- Modular code structure for easy maintenance and extension

## Project Structure
```
.
├── data/               # Sample videos and test data
├── src/               # Source code
│   ├── utils/         # Helper functions
│   └── main.py        # Main detection script
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your video files in the `data` directory
2. Run the main script:
```bash
python src/main.py
```

## Dependencies
- OpenCV 4.8.1
- NumPy 1.24.3
- Matplotlib 3.7.2 