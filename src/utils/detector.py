import cv2
import numpy as np

class ObjectDetector:
    def __init__(self):
        # HSV color range for cars
        # Genel araba renkleri için daha geniş aralık
        self.lower_hsv = np.array([0, 30, 30])
        self.upper_hsv = np.array([180, 255, 255])
        
        # Minimum contour area to filter noise
        self.min_contour_area = 400  # Hareketli arabalar için uygun değer
        
        # Store previous detections for tracking
        self.previous_detections = []
        
        # Hareket tespiti için önceki kare
        self.prev_frame = None

    def process_frame(self, frame):
        """
        Process a single frame to detect objects using HSV filtering.
        
        Args:
            frame: Input frame from video capture
            
        Returns:
            processed_frame: Frame with detection visualization
            detections: List of detected object coordinates and sizes
        """
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Hareket tespiti
        if self.prev_frame is not None:
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Hareket maskesi oluştur
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = cv2.threshold(magnitude, 1.0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
            
            # Hareket maskesi ile renk maskesini birleştir
            mask = cv2.bitwise_and(mask, motion_mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process detections
        detections = []
        processed_frame = frame.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Araba şekli kontrolü (genişlik/yükseklik oranı)
                aspect_ratio = float(w)/h
                
                # Hareketli araba şekli kontrolü
                if 0.8 < aspect_ratio < 3.0:  # Hareketli arabalar için tipik oran
                    # Store detection
                    detections.append({
                        'position': (x + w//2, y + h//2),
                        'size': (w, h),
                        'area': area
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(processed_frame, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
                    
                    # Add text showing the area and aspect ratio
                    cv2.putText(processed_frame, f'Area: {int(area)} Ratio: {aspect_ratio:.2f}', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update previous frame
        self.prev_frame = frame.copy()
        
        # Update previous detections
        self.previous_detections = detections
        
        # Maske görüntüsünü de göster
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Her iki görüntüyü yan yana göster
        combined_display = np.hstack((processed_frame, mask_display))
        
        # Görüntü boyutlarını ayarla
        height, width = combined_display.shape[:2]
        max_width = 1600  # Maksimum genişlik
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            combined_display = cv2.resize(combined_display, (new_width, new_height))
        
        return combined_display, detections

    def set_hsv_range(self, lower_hsv, upper_hsv):
        """
        Update HSV range for object detection.
        
        Args:
            lower_hsv: Lower HSV bounds
            upper_hsv: Upper HSV bounds
        """
        self.lower_hsv = lower_hsv
        self.upper_hsv = upper_hsv 