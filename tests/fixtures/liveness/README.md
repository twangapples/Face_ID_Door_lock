# Liveness Test Fixtures

Place test images here for benchmarking liveness detection:

- `real/` - Images of real faces (should get high liveness scores)
- `spoof/` - Images of printed photos, phone screens, or videos (should get low liveness scores)

Supported formats: `.jpg`, `.jpeg`, `.png`

## Usage

Run the benchmark script:
```bash
python scripts/benchmark_liveness.py
```

The script will evaluate the liveness model on these images and provide threshold tuning recommendations.

