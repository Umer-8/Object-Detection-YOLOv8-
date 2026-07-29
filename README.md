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
