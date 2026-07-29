# Object Detection using YOLOv8

## Files
- `test.jpg` — sample test image (simple illustrative shapes standing in for person/car/dog)
- `detect.py` — single script: loads YOLOv8n, runs detection on an image, prints detected
  classes + confidence scores, saves an annotated output image; also includes a video
  detection function (commented out by default, since no `video.mp4` is included)
- `requirements.txt`

## Run
```bash
pip install ultralytics opencv-python
python detect.py
```
This detects objects in `test.jpg`, prints results to the console, and saves an annotated
copy to `output.jpg`. On first run, `ultralytics` automatically downloads the pretrained
`yolov8n.pt` weights (~6MB).

To run on your own video, uncomment the last line in `detect.py` and supply your own
`video.mp4`; annotated output is written to `output_video.mp4` and shown live in a window
(press `q` to stop early).

## Example console output
```
Detected Objects:
- Person (0.92)
- Car (0.88)
- Dog (0.85)
```

## Testing note
This code could not be executed in the sandbox used to prepare this submission — installing
`ultralytics` pulls in PyTorch's full GPU/CUDA dependency stack (2GB+), which exceeded the
available disk space here, and CPU-only PyTorch wheels aren't reachable from this
environment's network allowlist. The code itself follows the standard, well-documented
`ultralytics` YOLOv8 API exactly as shown in the official docs, so it should run correctly
in a normal environment with `pip install ultralytics opencv-python` — but I was not able
to confirm actual detection output myself before delivering this. Please test on your end
and let me know if anything needs fixing.

No bonus features included (no webcam mode, saved output video by default, class filtering,
object counting, Streamlit UI, fine-tuning, or CSV logging) per request.
