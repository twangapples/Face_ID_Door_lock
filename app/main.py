"""Main application: live face recognition loop with liveness detection."""

import cv2
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    CAMERA_INDEX, MAX_FRAME_RATE,
    RECOGNITION_THRESHOLD, LOGS_DIR
)
from app.db import load_encodings, find_best_match
from app.recognizer import InsightFaceRecognizer
from app.liveness_onnx import LivenessDetector
from app.utils import crop_face_from_bbox, FrameRateLimiter, save_denied_attempt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'diagnostics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AuthenticationLogger:
    """Logs authentication attempts to JSONL file."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _convert_to_native(self, obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_to_native(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_native(value) for key, value in obj.items()}
        return obj
    
    def log_attempt(self, frame_id: int, bbox: Optional[tuple], username: Optional[str],
                   recognition_distance: Optional[float], liveness_score: Optional[float],
                   blink_detected: bool, decision: str, face_crop: Optional[np.ndarray] = None):
        """Log an authentication attempt."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_id': self._convert_to_native(frame_id),
            'bbox': self._convert_to_native(bbox) if bbox else None,
            'username': username,
            'recognition_distance': self._convert_to_native(recognition_distance) if recognition_distance is not None else None,
            'liveness_score': self._convert_to_native(liveness_score) if liveness_score is not None else None,
            'blink_detected': blink_detected,
            'decision': decision
        }
        
        # Save to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Save denied face crop if available
        if decision == 'DENIED' and face_crop is not None:
            denied_dir = LOGS_DIR / 'denied_attempts'
            denied_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            crop_path = denied_dir / f"{timestamp_str}.jpg"
            cv2.imwrite(str(crop_path), face_crop)


def main():
    """Run the main recognition loop with liveness detection."""
    # Initialize logger
    auth_logger = AuthenticationLogger(LOGS_DIR / 'attempts.jsonl')
    
    # Load encodings database
    logger.info("Loading face encodings database...")
    known_encodings = load_encodings()
    
    if len(known_encodings) == 0:
        print("Warning: No users enrolled. Please run 'python scripts/enroll.py' first.")
        return
    
    print(f"Loaded {len(known_encodings)} user(s)")
    for username, user_data in known_encodings.items():
        if isinstance(user_data, dict) and 'embeddings' in user_data:
            count = len(user_data['embeddings'])
            print(f"  - {username}: {count} encoding(s)")
    
    # Initialize InsightFace recognizer
    logger.info("Initializing InsightFace recognizer...")
    try:
        recognizer = InsightFaceRecognizer(ctx_id=-1)  # CPU mode
        recognizer.prepare()
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        print("Error: Failed to initialize face recognition model.")
        return
    
    # Initialize liveness detector
    logger.info("Initializing ONNX liveness detector...")
    try:
        liveness_detector = LivenessDetector()
        if liveness_detector._initialized:
            print("Liveness detection: ENABLED")
            logger.info(f"Liveness model: {liveness_detector.model_path.name}")
        else:
            print("Liveness detection: DISABLED (model not found)")
            logger.warning("Liveness detection disabled - system will DENY all attempts")
            liveness_detector = None
    except Exception as e:
        logger.error(f"Failed to initialize liveness detector: {e}", exc_info=True)
        print("Error: Failed to initialize liveness detector.")
        liveness_detector = None
    
    # Initialize camera
    print(f"\nInitializing camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera at index {CAMERA_INDEX}")
        sys.exit(1)
    
    print("Camera ready. Starting recognition loop...")
    print("Press 'q' to quit\n")
    
    # Frame rate limiter
    rate_limiter = FrameRateLimiter(MAX_FRAME_RATE)
    
    # Per-identity cooldown (avoid repeated prints)
    last_authorized = {}  # username -> timestamp
    cooldown_seconds = 5
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Rate limiting
            if not rate_limiter.should_process():
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Detect faces using InsightFace
            faces = recognizer.detect_and_encode(frame)
            
            if len(faces) > 0:
                # Process first face (can be extended to handle multiple)
                face = faces[0]
                bbox = face['bbox']
                embedding = face['embedding']
                
                # Find best match
                username, distance = find_best_match(embedding, known_encodings)
                
                # Crop face for liveness detection
                try:
                    face_crop = crop_face_from_bbox(frame, bbox, pad=0.2)
                except Exception as e:
                    logger.error(f"Failed to crop face: {e}")
                    face_crop = None
                
                if username and distance <= RECOGNITION_THRESHOLD:
                    # Recognition match found - check liveness
                    # Decision logic: if recognition_match AND liveness_is_live: AUTHORIZE else: DENY
                    live_ok = False
                    liveness_score = None
                    
                    # Liveness detection (required for authorization)
                    if liveness_detector and liveness_detector._initialized and face_crop is not None:
                        try:
                            live_ok = liveness_detector.is_live(face_crop)
                            liveness_score = liveness_detector.predict_score(face_crop)
                        except Exception as e:
                            logger.error(f"Liveness detection error: {e}", exc_info=True)
                            live_ok = False  # Fail-closed
                    else:
                        # Liveness detector not available - deny access (fail-closed)
                        logger.warning("Liveness detector not available, denying access (fail-closed)")
                        live_ok = False
                    
                    # Decision: authorize if recognition match AND liveness passes
                    if live_ok:
                        # Check cooldown
                        now = datetime.now().timestamp()
                        last_time = last_authorized.get(username, 0)
                        
                        if now - last_time >= cooldown_seconds:
                            # Calculate recognition confidence from distance
                            confidence = max(0.0, 1.0 - distance)
                            liveness_str = f"{liveness_score:.3f}" if liveness_score is not None else "N/A"
                            print(f"AUTHORIZED: {username} (confidence: {confidence:.3f}, liveness: {liveness_str})")
                            last_authorized[username] = now
                            
                            # Log authorized attempt
                            auth_logger.log_attempt(
                                frame_count, bbox, username, distance,
                                liveness_score, False, 'AUTHORIZED'  # blink_ok removed
                            )
                            
                            # Draw green box and text
                            top, left, bottom, right = bbox
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            liveness_str = f"{liveness_score:.3f}" if liveness_score is not None else "N/A"
                            cv2.putText(frame, f"{username} | {distance:.3f} | LIVE {liveness_str}", 
                                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
                        else:
                            # Still in cooldown, don't print but show on screen
                            top, left, bottom, right = bbox
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            liveness_str = f"{liveness_score:.3f}" if liveness_score is not None else "N/A"
                            cv2.putText(frame, f"{username} | {distance:.3f} | LIVE {liveness_str} (cooldown)", 
                                       (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (0, 255, 0), 1)
                    else:
                        # DENIED: recognition match but liveness failed
                        # liveness_str = f"{liveness_score:.3f}" if liveness_score is not None else "N/A"
                        reason = "liveness failed" if liveness_score is not None else "liveness unavailable"
                        # print(f"DENIED: {username} (distance: {distance:.3f}, liveness: {liveness_str})")
                        
                        # Log denied attempt
                        # auth_logger.log_attempt(
                        #     frame_count, bbox, username, distance,
                        #     liveness_score, False, 'DENIED', face_crop
                        # )
                        
                        # Save denied attempt for audit
                        # try:
                        #     model_name = liveness_detector.model_path.name if liveness_detector and liveness_detector._initialized else None
                        #     model_shape = liveness_detector.input_shape if liveness_detector and liveness_detector._initialized else None
                        #     save_denied_attempt(
                        #         frame, bbox, username, liveness_score or 0.0,
                        #         recognition_distance=distance,
                        #         model_name=model_name,
                        #         model_input_shape=model_shape
                        #     )
                        # except Exception as e:
                        #     logger.warning(f"Failed to save denied attempt: {e}")
                        
                        # Draw red box and text
                        top, left, bottom, right = bbox
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        liveness_str = f"{liveness_score:.3f}" if liveness_score is not None else "N/A"
                        cv2.putText(frame, f"{distance:.3f} | SPOOF {liveness_str}", 
                                   (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 0, 255), 2)
                else:
                    # DENIED: no recognition match
                    # distance_str = f"{distance:.3f}" if distance is not None else "N/A"
                    # print(f"DENIED: Unknown person (distance: {distance_str})")
                    # if face_crop is not None:
                    #     auth_logger.log_attempt(
                    #         frame_count, bbox, None, distance if distance else None,
                    #         None, False, 'DENIED', face_crop
                    #     )
                    #     
                    #     # Save denied attempt for audit
                    #     try:
                    #         save_denied_attempt(
                    #             frame, bbox, None, 0.0,
                    #             recognition_distance=distance
                    #         )
                    #     except Exception as e:
                    #         logger.warning(f"Failed to save denied attempt: {e}")
                    
                    # Draw yellow box (unknown person)
                    top, left, bottom, right = bbox
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                    distance_str = f"{distance:.3f}" if distance is not None else "N/A"
                    cv2.putText(frame, f"{distance_str}", 
                               (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 165, 255), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (128, 128, 128), 2)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        print(f"Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Exiting.")
        print("Camera released. Exiting.")


if __name__ == "__main__":
    main()
