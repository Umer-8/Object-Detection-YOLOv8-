import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def detect_image(image_path, output_path="output.jpg", conf_threshold=0.25):
    results = model(image_path, conf=conf_threshold)
    result = results[0]

    annotated_frame = result.plot()
    cv2.imwrite(output_path, annotated_frame)

    print("Detected Objects:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        print(f"- {class_name} ({confidence:.2f})")

    return result


def detect_video(video_path, output_path="output_video.mp4", conf_threshold=0.25):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()

        writer.write(annotated_frame)
        cv2.imshow("YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_image("test.jpg", output_path="output.jpg")
    detect_video("video.mp4", output_path="output_video.mp4")
