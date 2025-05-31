from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import torch
from ultralytics import YOLO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define folders
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model once at startup
model = YOLO("best.pt")  # Replace with the path to your trained model
model.fuse()  # Fuse model layers for faster inference

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            # Save uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Run model inference with memory optimization
            with torch.no_grad():
                results = model(img_path)

            # Visualize and save results
            output_img = results[0].plot()
            output_path = os.path.join(RESULT_FOLDER, file.filename)
            cv2.imwrite(output_path, output_img)

            # Extract detected class names and confidence levels
            detections = []
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                confidence = round(box.conf[0].item() * 100, 2)
                detections.append({"class": class_name, "confidence": confidence})

            return render_template("result.html", image_file=file.filename, detections=detections)

    return render_template("index.html")

@app.route("/results/<filename>")
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
