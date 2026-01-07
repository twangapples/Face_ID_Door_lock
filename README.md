:# Secure Face Recognition System

A secure face recognition system with anti-spoofing protection. Features InsightFace 512-D embeddings and ML-based liveness detection.

## Features

- **InsightFace Recognition**: Robust 512-D face embeddings for accurate identification
- **ML Liveness Detection**: ONNX-based anti-spoofing to block photos, screens, and videos
- **Secure by Default**: Fail-closed security model (denies on errors)
- **Comprehensive Logging**: Structured logging of all authentication attempts

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Models

#### InsightFace Models
InsightFace models are automatically downloaded on first run. The system uses the `buffalo_l` model by default, which provides a good balance of speed and accuracy.

#### Liveness Detection Model (ONNX)
You need to download an ONNX liveness detection model from [hairymax/Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing):

1. Clone or download the repository
2. Navigate to `saved_models/` directory
3. Copy the ONNX model file(s) to `models/liveness/` in this project
4. Update `LIVENESS_MODEL_PATH` in `app/config.py` to point to your model file (default: `face_antispoof.onnx`)

**Note**: The liveness detector auto-detects model input requirements (shape, layout, etc.), so it should work with different ONNX models from the hairymax repository.

**Important**: Without a liveness model, the system will **DENY all access attempts** (fail-closed security model). Liveness detection is required for authorization.

### 3. Verify Installation

Test your camera:
```bash
python scripts/test_camera.py
```

## Usage

### Enroll a New User

To add a new user to the system:

```bash
python scripts/enroll.py
```

This will:
1. Prompt for a username
2. Open the camera and capture photos (press SPACE to capture each photo)
3. Generate 512-D face encodings using InsightFace
4. Save the encodings to the database

### Run Live Recognition

To start the secure recognition system:

```bash
python app/main.py
# or
python -m app.main
```

The system will:
- Load all enrolled users
- Open the camera feed
- Detect faces using InsightFace
- **Decision logic**: `if recognition_match AND liveness_is_live: AUTHORIZE else: DENY`
- Verify liveness using ONNX model (anti-spoofing)
- Print "AUTHORIZED: <username>" if both recognition and liveness pass
- Print "DENIED" if recognition fails or liveness check fails
- Log all attempts to `data/logs/attempts.jsonl`
- Save denied attempts to `data/logs/denied/` for audit
- Press 'q' to quit

### Benchmarking

#### Tune Recognition Threshold

```bash
python scripts/benchmark_recognition.py
```

This computes intra-class (same person) and inter-class (different people) distances to help you tune the recognition threshold.

#### Tune Liveness Threshold

```bash
python scripts/tune_liveness.py --real-dir tune/real --spoof-dir tune/spoof
```

This script:
- Loads labeled dataset from `tune/real/` and `tune/spoof/` directories
- Computes ROC curve and AUC
- Suggests optimal threshold targeting desired False Accept Rate (FAR)
- Optionally saves ROC curve plot with `--save-plot`

Example:
```bash
python scripts/tune_liveness.py --real-dir tune/real --spoof-dir tune/spoof --target-far 0.005 --save-plot
```

This will recommend a threshold for 0.5% False Accept Rate and save a plot to `roc_curve.png`.

#### Run Liveness Tests

```bash
pytest tests/test_liveness.py
```

This runs unit and integration tests for the liveness detector. Tests are automatically skipped if the model file is not found.

## Configuration

Edit `app/config.py` to adjust settings:

### Recognition Settings
- `RECOGNITION_THRESHOLD`: Face distance threshold (default: 0.45, tuned for 512-D)
- `EMBEDDING_DIM`: Embedding dimension (512 for InsightFace)
- `PHOTOS_PER_USER`: Number of photos to capture during enrollment (default: 5)

### Liveness Settings (ONNX)
- `LIVENESS_MODEL_PATH`: Path to ONNX liveness model (default: `models/liveness/face_antispoof.onnx`)
- `LIVENESS_PROVIDERS`: ONNX execution providers (None = CPU default, or `["CUDAExecutionProvider", "CPUExecutionProvider"]` for GPU)
- `LIVENESS_SCORE_THRESHOLD`: Liveness score threshold (default: 0.6, higher = stricter, tune with `scripts/tune_liveness.py`)
- `LIVENESS_BATCH`: Batch size for inference (default: 1 for real-time)
- `LIVENESS_IGNORE_EXCEPTIONS`: If False, exceptions return False (fail-closed, recommended)

### Performance Settings
- `CAMERA_INDEX`: Webcam index (default: 0)
- `MAX_FRAME_RATE`: Maximum frames per second to process (default: 10)

## Project Structure

```
Face_Id_Door_Lock/
├── app/
│   ├── main.py          # Main recognition loop with liveness
│   ├── recognizer.py    # InsightFace wrapper (512-D embeddings)
│   ├── liveness_onnx.py # ONNX liveness detection (auto-detects model requirements)
│   ├── capture.py       # Photo capture functionality
│   ├── db.py            # Encoding storage/loading (with migration)
│   ├── utils.py         # Helper functions (crop, normalize, save denied attempts)
│   └── config.py        # Configuration settings
├── models/
│   ├── insightface/     # InsightFace model artifacts (auto-downloaded)
│   └── liveness/        # Liveness ONNX models
├── data/
│   ├── users/           # User enrollment images
│   ├── encodings.pkl    # Face encodings database (512-D)
│   └── logs/            # Authentication logs
│       ├── attempts.jsonl      # Structured event log
│       ├── diagnostics.log    # Model errors and performance metrics
│       └── denied/             # Denied attempts (images + JSON metadata)
├── scripts/
│   ├── enroll.py              # User enrollment tool
│   ├── benchmark_recognition.py  # Recognition threshold tuning
│   ├── tune_liveness.py       # Liveness threshold tuning (ROC/AUC)
│   └── test_camera.py          # Camera test utility
├── tests/
│   ├── test_liveness.py       # Unit and integration tests for liveness
│   └── fixtures/              # Test images (real_face.jpg, spoof_face.jpg)
├── requirements.txt
└── README.md
```

## Security Features

### Liveness Detection
The system uses ML-based liveness detection to prevent spoofing attacks:
- **ONNX Model**: Runs inference on face crops to detect if the face is "live"
- **Fail-Closed**: If liveness detection fails or errors occur, access is denied
- **Threshold Tuning**: Adjust `LIVENESS_SCORE_THRESHOLD` based on your security requirements

### Model Auto-Detection
The liveness detector automatically detects model requirements:
- **Input Shape**: Queries model metadata to determine expected input dimensions
- **Layout Detection**: Auto-detects NCHW vs NHWC data layout
- **Output Normalization**: Handles different output formats (softmax, sigmoid, raw scores)
- **Compatibility**: Works with different ONNX models from hairymax/Face-AntiSpoofing without manual configuration

### Decision Logic
The system uses a strict two-factor approach:
1. **Recognition**: Face must match an enrolled user (distance <= threshold)
2. **Liveness**: ONNX model must classify face as "live" (score >= threshold)

**Authorization**: `if recognition_match AND liveness_is_live: AUTHORIZE else: DENY`

**Fail-Closed Security**: If liveness model is unavailable or any error occurs, access is **DENIED**. This ensures maximum security.

## Troubleshooting

### Camera not opening
- Check that your webcam is connected and not being used by another application
- Try changing `CAMERA_INDEX` in `config.py` (try 1, 2, etc.)
- Run `python scripts/test_camera.py` to verify camera connectivity

### InsightFace model download fails
- Ensure you have internet connection (models download on first run)
- Check disk space (models can be several hundred MB)
- If using proxy, configure it for Python/pip

### Liveness detection not working
- Ensure the ONNX model file exists at the path specified in `config.py` (default: `models/liveness/face_antispoof.onnx`)
- Check that `onnxruntime` is installed correctly: `pip install onnxruntime`
- The system auto-detects model input requirements - no manual configuration needed
- Check `data/logs/diagnostics.log` for error messages
- Verify model is from [hairymax/Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing) `saved_models/` directory
- **Note**: Without a valid liveness model, all access attempts will be DENIED (fail-closed)

### Too many false denials
- Lower `LIVENESS_SCORE_THRESHOLD` (e.g., 0.5 instead of 0.6)
- Ensure good lighting conditions
- Verify liveness model is appropriate for your use case
- Use `benchmark_liveness.py` to tune threshold based on your data

### Too many false accepts
- Raise `LIVENESS_SCORE_THRESHOLD` (e.g., 0.7 instead of 0.6)
- Ensure liveness model is properly trained/tested
- Consider using stricter recognition threshold
- Review denied attempts in `data/logs/denied_attempts/`

### Recognition not working
- Make sure you've enrolled at least one user first
- Try capturing more photos during enrollment
- Adjust `RECOGNITION_THRESHOLD` using `benchmark_recognition.py`

### Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure you're running scripts from the project root directory
- Check that InsightFace models downloaded successfully

## Performance

### Expected Latency
- **Raspberry Pi 4 (CPU)**: ~200-400ms per authentication
- **Desktop CPU**: ~50-150ms per authentication
- **GPU acceleration**: Can reduce latency significantly

### Optimization Tips
- Reduce `MAX_FRAME_RATE` to process fewer frames
- Use smaller liveness model input size if supported
- Process every Nth frame instead of every frame


## Logging

All authentication attempts are logged to `data/logs/attempts.jsonl` with:
- Timestamp
- Frame ID
- Bounding box
- Username (if recognized)
- Recognition distance
- Liveness score
- Final decision (AUTHORIZED/DENIED)

Denied attempts also save face crops to `data/logs/denied_attempts/` for review.

## Security Notes

- This system provides **significantly better security** than the basic version
- Liveness detection helps prevent photo/screen/video spoofing
- However, no system is 100% secure - use appropriate security measures for your use case
- For high-security applications, consider additional factors (PIN, card, etc.)
- Regularly review logs and tune thresholds based on observed false accepts/denies

## License

This project is provided as-is for educational and development purposes.
