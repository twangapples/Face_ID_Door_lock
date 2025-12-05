"""Main application: live face recognition loop."""

import cv2
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import CAMERA_INDEX
from app.db import load_encodings
from app.recognizer import detect_and_encode_frame, find_match

def main():
    """Run the main recognition loop."""
    # Load encodings database
    print("Loading face encodings database...")
    known_encodings = load_encodings()
    
    if len(known_encodings) == 0:
        print("Warning: No users enrolled. Please run 'python scripts/enroll.py' first.")
        return
    
    print(f"Loaded {len(known_encodings)} user(s)")
    for username in known_encodings.keys():
        print(f"  - {username}: {len(known_encodings[username])} encoding(s)")
    
    # Initialize camera
    print(f"\nInitializing camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Failed to open camera at index {CAMERA_INDEX}")
        sys.exit(1)
    
    print("Camera ready. Starting recognition loop...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for performance (adjust as needed)
            if frame_count % 3 == 0:
                # Detect and encode face
                encoding = detect_and_encode_frame(frame)
                
                if encoding is not None:
                    # Find match
                    match = find_match(encoding, known_encodings)
                    
                    if match:
                        print(f"AUTHORIZED: {match}")
                        # Draw green box and text on frame
                        cv2.putText(frame, f"AUTHORIZED: {match}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 2)
                    else:
                        print("DENIED")
                        # Draw red text on frame
                        cv2.putText(frame, "DENIED", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 0, 255), 2)
                else:
                    # No face detected
                    cv2.putText(frame, "No face detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (128, 128, 128), 2)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Exiting.")


if __name__ == "__main__":
    main()

