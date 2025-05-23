import numpy as np
from datetime import datetime

class AltitudeAnalyzer:
    def __init__(self):
        # Store detection history
        self.detection_history = []
        
        # Parameters for altitude analysis
        self.min_detection_size = 100  # Minimum size to consider a detection valid
        self.max_detection_size = 1000  # Maximum size to consider a detection valid
        
        # Altitude estimation parameters
        self.reference_size = 500  # Reference size at known altitude
        self.reference_altitude = 100  # Known altitude in meters

    def analyze_detections(self, detections):
        """
        Analyze detections and estimate altitude based on object sizes.
        
        Args:
            detections: List of detected objects with their properties
            
        Returns:
            altitude_data: Dictionary containing altitude analysis results
        """
        current_time = datetime.now()
        altitude_data = {
            'timestamp': current_time,
            'num_objects': len(detections),
            'estimated_altitude': None,
            'object_sizes': []
        }
        
        if not detections:
            return altitude_data
        
        # Calculate average object size
        valid_sizes = []
        for detection in detections:
            size = detection['size']
            area = size[0] * size[1]
            if self.min_detection_size <= area <= self.max_detection_size:
                valid_sizes.append(area)
        
        if valid_sizes:
            avg_size = np.mean(valid_sizes)
            altitude_data['object_sizes'] = valid_sizes
            
            # Estimate altitude based on object size
            # Using inverse square relationship between size and altitude
            altitude_data['estimated_altitude'] = self._estimate_altitude(avg_size)
        
        # Store in history
        self.detection_history.append(altitude_data)
        
        return altitude_data

    def _estimate_altitude(self, object_size):
        """
        Estimate altitude based on object size using inverse square relationship.
        
        Args:
            object_size: Average size of detected objects
            
        Returns:
            estimated_altitude: Estimated altitude in meters
        """
        # Using inverse square relationship: size ∝ 1/altitude²
        ratio = self.reference_size / object_size
        estimated_altitude = self.reference_altitude * np.sqrt(ratio)
        return estimated_altitude

    def get_altitude_trend(self, window_size=10):
        """
        Get altitude trend over recent detections.
        
        Args:
            window_size: Number of recent detections to analyze
            
        Returns:
            trend_data: Dictionary containing trend analysis
        """
        if not self.detection_history:
            return None
        
        recent_data = self.detection_history[-window_size:]
        altitudes = [data['estimated_altitude'] for data in recent_data 
                    if data['estimated_altitude'] is not None]
        
        if not altitudes:
            return None
        
        trend_data = {
            'mean_altitude': np.mean(altitudes),
            'std_altitude': np.std(altitudes),
            'trend': 'increasing' if altitudes[-1] > altitudes[0] else 'decreasing'
        }
        
        return trend_data 