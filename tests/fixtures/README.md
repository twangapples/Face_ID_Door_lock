# Test Fixtures

This directory contains test images for liveness detection testing.

## Files

- `real_face.jpg` - Placeholder image representing a real face
- `spoof_face.jpg` - Placeholder image representing a spoof face (photo/screen)

## Note

The current images are simple placeholders. For meaningful tests, replace these with:
- Actual face images captured from a camera (for `real_face.jpg`)
- Photos or screens showing faces (for `spoof_face.jpg`)

The integration test `test_predict_real_vs_spoof` will compare scores between these images to verify the model correctly distinguishes real vs spoof faces.

