"""CLI tool to enroll a new user."""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.capture import capture_user_photos
from app.recognizer import encode_face
from app.db import add_user_encodings


def main():
    """Enroll a new user."""
    print("=== User Enrollment ===")
    
    # Get username
    username = input("Enter username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return
    
    # Capture photos
    print(f"\nCapturing photos for {username}...")
    try:
        image_paths = capture_user_photos(username)
    except Exception as e:
        print(f"Error capturing photos: {e}")
        return
    
    if len(image_paths) == 0:
        print("Error: No photos captured")
        return
    
    # Generate encodings from captured images
    print(f"\nGenerating face encodings from {len(image_paths)} image(s)...")
    encodings = []
    
    for img_path in image_paths:
        encoding = encode_face(img_path)
        if encoding is not None:
            encodings.append(encoding)
            print(f"  ✓ Encoded: {img_path}")
        else:
            print(f"  ✗ No face found in: {img_path}")
    
    if len(encodings) == 0:
        print("Error: No valid face encodings generated. Please ensure faces are visible in the images.")
        return
    
    # Save encodings to database
    print(f"\nSaving {len(encodings)} encoding(s) to database...")
    try:
        add_user_encodings(username, encodings)
        print(f"✓ User '{username}' enrolled successfully!")
    except Exception as e:
        print(f"Error saving encodings: {e}")
        return


if __name__ == "__main__":
    main()

