import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

class ObjectTracker:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.track_history = deque(maxlen=30)  # Keep track of last 30 frames
        
        # Define sports-related classes we're interested in
        self.sports_classes = {
            0: 'person',  # Players
            32: 'sports ball',  # Ball
            33: 'kite',  # Could be used for other sports equipment
        }
        
    def detect_objects(self, frame, debug=False):
        """
        Detect objects in the frame and return relevant information
        Args:
            frame: Input frame
            debug: Whether to show debug visualization
        Returns:
            dict containing detection results
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=self.confidence_threshold, classes=list(self.sports_classes.keys()))
        
        # Process results
        detections = {
            'players': [],
            'ball': None,
            'has_action': False
        }
        
        # Get the first result (since we're processing one frame at a time)
        if len(results) > 0:
            result = results[0]
            
            # Process each detection
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                
                # Convert to (x1, y1, x2, y2) format
                x1, y1, x2, y2 = map(int, bbox)
                
                # Store detection based on class
                if class_id == 0:  # Person/player
                    detections['players'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
                elif class_id == 32:  # Sports ball
                    detections['ball'] = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    }
            
            # Determine if there's significant action
            # Action is considered significant if:
            # 1. There are multiple players detected
            # 2. A ball is detected
            # 3. Players and ball are in close proximity
            if len(detections['players']) >= 2 and detections['ball'] is not None:
                detections['has_action'] = True
                
                # Calculate ball-player proximity
                ball_center = (
                    (detections['ball']['bbox'][0] + detections['ball']['bbox'][2]) // 2,
                    (detections['ball']['bbox'][1] + detections['ball']['bbox'][3]) // 2
                )
                
                # Check if any player is close to the ball
                for player in detections['players']:
                    player_center = (
                        (player['bbox'][0] + player['bbox'][2]) // 2,
                        (player['bbox'][1] + player['bbox'][3]) // 2
                    )
                    distance = np.sqrt((ball_center[0] - player_center[0])**2 + 
                                     (ball_center[1] - player_center[1])**2)
                    
                    # If a player is within 200 pixels of the ball, consider it significant
                    if distance < 200:
                        detections['has_action'] = True
                        break
        
        # Debug visualization
        if debug:
            debug_frame = frame.copy()
            
            # Draw player bounding boxes
            for player in detections['players']:
                x1, y1, x2, y2 = player['bbox']
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"Player: {player['confidence']:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ball bounding box
            if detections['ball']:
                x1, y1, x2, y2 = detections['ball']['bbox']
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_frame, f"Ball: {detections['ball']['confidence']:.2f}", 
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Show action status
            cv2.putText(debug_frame, f"Action: {'Yes' if detections['has_action'] else 'No'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Object Detection Debug", debug_frame)
            cv2.waitKey(1)
        
        return detections 