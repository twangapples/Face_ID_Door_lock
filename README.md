# Basic Face Recognition System

A simple face recognition system that can enroll users and perform live recognition. Software-only implementation that prints "AUTHORIZED" or "DENIED" based on face matches.

## Features

- **User Enrollment**: Capture photos and generate face encodings
- **Live Recognition**: Real-time face detection and matching
- **Simple Storage**: Pickle-based database for face encodings
- **No Hardware Required**: Works with any USB webcam

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a webcam connected and accessible.

## Usage

### Enroll a New User

To add a new user to the system:

```bash
python scripts/enroll.py
```

This will:
1. Prompt for a username
2. Open the camera and capture photos (press SPACE to capture each photo)
3. Generate face encodings from the captured images
4. Save the encodings to the database

### Run Live Recognition

To start the recognition system:

```bash
python app/main.py
```

The system will:
- Load all enrolled users
- Open the camera feed
- Detect faces in real-time
- Print "AUTHORIZED: <username>" if a match is found
- Print "DENIED" if no match is found
- Press 'q' to quit

### Test Camera

To verify your camera is working:

```bash
python scripts/test_camera.py
```

This displays a live camera feed with frame information. Press 'q' to quit.

## Configuration

Edit `app/config.py` to adjust:

- `RECOGNITION_THRESHOLD`: Face distance threshold for matching (default: 0.5)
- `PHOTOS_PER_USER`: Number of photos to capture during enrollment (default: 5)
- `CAMERA_INDEX`: Webcam index (default: 0)

## Project Structure

```
face_lock_basic/
├── app/
│   ├── main.py          # Main recognition loop
│   ├── capture.py       # Photo capture functionality
│   ├── recognizer.py    # Face encoding and matching
│   ├── db.py            # Encoding storage/loading
│   └── config.py        # Configuration settings
├── data/
│   ├── users/           # User enrollment images
│   └── encodings.pkl    # Face encodings database
├── scripts/
│   ├── enroll.py        # User enrollment tool
│   └── test_camera.py   # Camera test utility
├── requirements.txt
└── README.md
```

## Troubleshooting

### Camera not opening
- Check that your webcam is connected and not being used by another application
- Try changing `CAMERA_INDEX` in `config.py` (try 1, 2, etc.)
- Run `python scripts/test_camera.py` to verify camera connectivity

### No face detected
- Ensure good lighting
- Face the camera directly
- Remove obstructions (glasses, masks, etc.)

### Recognition not working
- Make sure you've enrolled at least one user first
- Try capturing more photos during enrollment (increase `PHOTOS_PER_USER`)
- Adjust `RECOGNITION_THRESHOLD` in `config.py` (lower = stricter, higher = more lenient)

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure you're running scripts from the project root directory

## Notes

- This is a basic implementation and is **not secure** against printed photos or screens
- For production use, consider adding liveness detection
- The system uses the HOG model from `face_recognition` library (fast but basic)
- Face encodings are stored in a pickle file (`data/encodings.pkl`)

