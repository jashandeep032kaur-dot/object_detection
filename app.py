import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# -------------------------------------
# Load YOLO model (replace with your model if trained)
# -------------------------------------
MODEL_PATH = "yolov8n.pt"  # change to "best.pt" if you have a custom model
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully ✅")


# -------------------------------------
# Image detection function
# -------------------------------------
def detect_image(image):
    results = model(image)
    annotated_image = results[0].plot()  # draw boxes and labels
    return annotated_image


# -------------------------------------
# Video detection function
# -------------------------------------
def detect_video(video):
    cap = cv2.VideoCapture(video)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return temp_output.name


# -------------------------------------
# Gradio UI setup
# -------------------------------------
def detect(mode, media):
    """
    Unified detection function:
    mode = "Image" or "Video"
    media = file path
    """
    if mode == "Image":
        return detect_image(media)
    elif mode == "Video":
        return detect_video(media)
    else:
        return "❌ Invalid mode selected."


# -------------------------------------
# Build Gradio interface
# -------------------------------------
title = "🦾 YOLO Object Detection (Image + Video)"
description = (
    "Upload an image or a video and select the mode. "
    "The model will detect common objects using YOLOv8."
)

iface = gr.Interface(
    fn=detect,
    inputs=[
        gr.Radio(choices=["Image", "Video"], label="Choose Input Type"),
        gr.File(label="Upload Image or Video")
    ],
    outputs=gr.components.Image(label="Detected Image", type="numpy"),
    title=title,
    description=description,
    allow_flagging="never"
)


# -------------------------------------
# Dynamic output handling (switching between image/video)
# -------------------------------------
def process_input(mode, file):
    if mode == "Image":
        image = cv2.imread(file.name)
        result = detect_image(image)
        return result
    elif mode == "Video":
        result_path = detect_video(file.name)
        return result_path


iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Radio(choices=["Image", "Video"], label="Input Type", value="Image"),
        gr.File(label="Upload Image or Video")
    ],
    outputs=[
        gr.Image(label="Detected Image"),
        # For videos, display output file
        gr.Video(label="Detected Video")
    ],
    title=title,
    description=description,
)

if __name__ == "__main__":
    iface.launch()
