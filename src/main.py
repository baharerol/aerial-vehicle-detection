import cv2
import numpy as np
import os
from datetime import datetime
from utils.detector import ObjectDetector
from utils.altitude_analyzer import AltitudeAnalyzer

def main():
    # Initialize video capture
    video_path = os.path.join('data', 'sample_video.mp4')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize detector and analyzer
    detector = ObjectDetector()
    analyzer = AltitudeAnalyzer()

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, detections = detector.process_frame(frame)
        
        # Analyze detections based on altitude
        altitude_data = analyzer.analyze_detections(detections)

        # Display results
        cv2.imshow('Object Detection', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 