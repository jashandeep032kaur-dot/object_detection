import gradio as gr
from ultralytics import YOLO
import numpy as np
import cv2

# ----------------------------
# Load YOLO model
# ----------------------------
# You can replace this with your own model, e.g. "best.pt"
MODEL_PATH = "yolov8n.pt"
print(f"Loading YOLO model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# ----------------------------
# Detection function
# ----------------------------
def detect_objects(image):
    """
    Perform object detection on the uploaded image using YOLOv8.
    Returns the image annotated with bounding boxes.
    """
    # Run inference
    results = model(image)

    # Draw bounding boxes
    annotated_image = results[0].plot()  # Ultralytics provides this helper
    return annotated_image

# ----------------------------
# Gradio Interface
# ----------------------------
title = "🦾 YOLOv8 Image Object Detection"
description = (
    "Upload an image to detect common objects using YOLOv8. "
    "You can replace the model file (`yolov8n.pt`) with your own trained YOLO model (e.g. `best.pt`)."
)

iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(label="Upload an Image", type="numpy"),
    outputs=gr.Image(label="Detected Objects"),
    title=title,
    description=description,
    examples=[
        ["https://ultralytics.com/images/zidane.jpg"],
        ["https://ultralytics.com/images/bus.jpg"]
    ],
)

if __name__ == "__main__":
    iface.launch()
