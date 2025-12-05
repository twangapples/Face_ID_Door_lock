"""Module for capturing user photos from webcam."""

import cv2
import os
from pathlib import Path
from app.config import USERS_DIR, CAMERA_INDEX, PHOTOS_PER_USER


def capture_user_photos(username, num_photos=None):
    """
    Capture photos of a user from webcam and save them.
    
    Args:
        username: Name of the user to enroll
        num_photos: Number of photos to capture (defaults to config value)
    
    Returns:
        List of paths to saved image files
    """
    if num_photos is None:
        num_photos = PHOTOS_PER_USER
    
    # Create user directory
    user_dir = USERS_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera at index {CAMERA_INDEX}")
    
    saved_images = []
    photo_count = 0
    
    print(f"\nPreparing to capture {num_photos} photos for {username}")
    print("Press SPACE to capture a photo, 'q' to quit early")
    print("Make sure your face is clearly visible in the frame\n")
    
    try:
        while photo_count < num_photos:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Display countdown and instructions
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Photo {photo_count + 1}/{num_photos}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to capture", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check for face to ensure visibility
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                cv2.putText(display_frame, "Face detected - Ready!", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Capture Photos', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to capture
                if len(faces) > 0:
                    # Save image
                    img_path = user_dir / f"img{photo_count + 1}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    saved_images.append(str(img_path))
                    photo_count += 1
                    print(f"Captured photo {photo_count}/{num_photos}: {img_path}")
                else:
                    print("No face detected! Please position yourself in front of the camera.")
            
            elif key == ord('q'):
                print("Capture cancelled by user")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    if len(saved_images) < num_photos:
        print(f"Warning: Only captured {len(saved_images)}/{num_photos} photos")
    
    return saved_images

