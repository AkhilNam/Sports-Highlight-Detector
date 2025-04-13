import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

class SportDetector:
    def __init__(self, sport_type='basketball', confidence_threshold=0.5):
        """
        Initialize sport-specific detector
        Args:
            sport_type: 'basketball' or 'soccer'
            confidence_threshold: Minimum confidence for detections
        """
        self.sport_type = sport_type.lower()
        self.confidence_threshold = confidence_threshold
        self.model = YOLO('yolov8n.pt')
        
        # Sport-specific parameters
        if self.sport_type == 'basketball':
            self.classes = [0]  # Only track people
            self.min_players = 4  # Minimum players for significant action
            self.action_threshold = 0.7  # Higher threshold for basketball
        else:  # soccer
            self.classes = [0, 32]  # Track people and sports ball
            self.min_players = 6  # Minimum players for significant action
            self.action_threshold = 0.5  # Lower threshold for soccer
            
        # Tracking history
        self.track_history = deque(maxlen=30)
        self.last_positions = {}
        
    def detect_basketball_action(self, frame, debug=False):
        """
        Basketball-specific action detection
        Focuses on player movements and court regions
        """
        results = self.model(frame, conf=self.confidence_threshold, classes=self.classes)
        
        detections = {
            'players': [],
            'has_action': False,
            'action_type': None
        }
        
        if len(results) > 0:
            result = results[0]
            
            # Process player detections
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                if class_id == 0:  # Person
                    detections['players'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
            
            # Analyze player positions and movements
            if len(detections['players']) >= self.min_players:
                # Check for fast break (players moving quickly in same direction)
                if self._detect_fast_break(detections['players']):
                    detections['has_action'] = True
                    detections['action_type'] = 'fast_break'
                
                # Check for scoring opportunity (players near basket)
                elif self._detect_scoring_opportunity(detections['players']):
                    detections['has_action'] = True
                    detections['action_type'] = 'scoring_opportunity'
        
        if debug:
            self._draw_basketball_debug(frame, detections)
            
        return detections
    
    def detect_soccer_action(self, frame, debug=False):
        """
        Soccer-specific action detection
        Focuses on ball tracking and player-ball interactions
        """
        results = self.model(frame, conf=self.confidence_threshold, classes=self.classes)
        
        detections = {
            'players': [],
            'ball': None,
            'has_action': False,
            'action_type': None
        }
        
        if len(results) > 0:
            result = results[0]
            
            # Process detections
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                if class_id == 0:  # Person
                    detections['players'].append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    })
                elif class_id == 32:  # Sports ball
                    detections['ball'] = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                    }
            
            # Analyze ball and player interactions
            if detections['ball'] and len(detections['players']) >= self.min_players:
                # Check for shot on goal
                if self._detect_shot_on_goal(detections):
                    detections['has_action'] = True
                    detections['action_type'] = 'shot_on_goal'
                
                # Check for scoring opportunity
                elif self._detect_scoring_opportunity(detections):
                    detections['has_action'] = True
                    detections['action_type'] = 'scoring_opportunity'
                
                # Check for fast break/counter attack
                elif self._detect_counter_attack(detections):
                    detections['has_action'] = True
                    detections['action_type'] = 'counter_attack'
        
        if debug:
            self._draw_soccer_debug(frame, detections)
            
        return detections
    
    def detect_action(self, frame, debug=False):
        """
        Main detection function that routes to sport-specific detectors
        """
        if self.sport_type == 'basketball':
            return self.detect_basketball_action(frame, debug)
        else:
            return self.detect_soccer_action(frame, debug)
    
    def _detect_fast_break(self, players):
        """Detect fast break in basketball"""
        if len(players) < 2:
            return False
            
        # Check if multiple players are moving in same direction
        directions = []
        for player in players:
            center = player['center']
            if center in self.last_positions:
                prev_center = self.last_positions[center]
                direction = (center[0] - prev_center[0], center[1] - prev_center[1])
                directions.append(direction)
            
            self.last_positions[center] = center
        
        if len(directions) >= 2:
            # Check if directions are similar
            avg_direction = np.mean(directions, axis=0)
            similarities = [np.dot(d, avg_direction) for d in directions]
            return np.mean(similarities) > self.action_threshold
        
        return False
    
    def _detect_scoring_opportunity(self, detections):
        """Detect scoring opportunity based on sport"""
        if self.sport_type == 'basketball':
            # Check if players are in scoring position (near basket)
            for player in detections:
                if self._is_near_basket(player['center']):
                    return True
        else:  # soccer
            if not detections['ball']:
                return False
            # Check if ball is in scoring position
            return self._is_near_goal(detections['ball']['center'])
        
        return False
    
    def _detect_shot_on_goal(self, detections):
        """Detect shot on goal in soccer"""
        if not detections['ball']:
            return False
            
        ball_center = detections['ball']['center']
        # Check if ball is moving toward goal
        if ball_center in self.last_positions:
            prev_center = self.last_positions[ball_center]
            direction = (ball_center[0] - prev_center[0], ball_center[1] - prev_center[1])
            return self._is_moving_toward_goal(direction)
        
        return False
    
    def _detect_counter_attack(self, detections):
        """Detect counter attack in soccer"""
        if not detections['ball'] or len(detections['players']) < 3:
            return False
            
        ball_center = detections['ball']['center']
        # Check if ball and players are moving quickly in same direction
        moving_players = 0
        for player in detections['players']:
            if self._is_moving_with_ball(player['center'], ball_center):
                moving_players += 1
        
        return moving_players >= 3
    
    def _is_near_basket(self, point):
        """Check if point is near basketball hoop"""
        # This would need to be calibrated based on court dimensions
        basket_region = (400, 200, 800, 400)  # Example coordinates
        return (basket_region[0] <= point[0] <= basket_region[2] and 
                basket_region[1] <= point[1] <= basket_region[3])
    
    def _is_near_goal(self, point):
        """Check if point is near soccer goal"""
        # This would need to be calibrated based on field dimensions
        goal_region = (300, 100, 500, 300)  # Example coordinates
        return (goal_region[0] <= point[0] <= goal_region[2] and 
                goal_region[1] <= point[1] <= goal_region[3])
    
    def _is_moving_toward_goal(self, direction):
        """Check if movement is toward goal"""
        # This would need to be calibrated based on field orientation
        goal_direction = (0, -1)  # Example: goal is upward
        similarity = np.dot(direction, goal_direction)
        return similarity > self.action_threshold
    
    def _is_moving_with_ball(self, player_center, ball_center):
        """Check if player is moving with the ball"""
        if player_center in self.last_positions:
            prev_center = self.last_positions[player_center]
            player_direction = (player_center[0] - prev_center[0], 
                              player_center[1] - prev_center[1])
            ball_direction = (ball_center[0] - player_center[0], 
                            ball_center[1] - player_center[1])
            similarity = np.dot(player_direction, ball_direction)
            return similarity > self.action_threshold
        return False
    
    def _draw_basketball_debug(self, frame, detections):
        """Draw basketball-specific debug visualization"""
        debug_frame = frame.copy()
        
        # Draw player bounding boxes
        for player in detections['players']:
            x1, y1, x2, y2 = player['bbox']
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Player: {player['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw action status
        cv2.putText(debug_frame, f"Action: {detections['action_type'] if detections['has_action'] else 'None'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Basketball Detection Debug", debug_frame)
        cv2.waitKey(1)
    
    def _draw_soccer_debug(self, frame, detections):
        """Draw soccer-specific debug visualization"""
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
        
        # Draw action status
        cv2.putText(debug_frame, f"Action: {detections['action_type'] if detections['has_action'] else 'None'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Soccer Detection Debug", debug_frame)
        cv2.waitKey(1) 